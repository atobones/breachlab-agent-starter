"""The level's SSH session, and the rule about where it may connect.

Transport choice: the system `ssh` binary. No third party SSH library: the
agent uses the same OpenSSH the operator already trusts. The password is typed into ssh's own prompt through a
pseudo-terminal, so it never appears in argv or in the environment. The
remote shell runs without a terminal (`ssh -T`) and reads commands from a
pipe: no echo, no prompt and no line editing to parse around.

Connection rule: a session opens only to a target the TargetGuard authorized,
and the guard authorizes only a Credentials object built from what the
BreachLab API returned for the level being played. Nothing the model says can
name a host.
"""

from __future__ import annotations

import fcntl
import ipaddress
import os
import re
import secrets
import select
import signal
import subprocess
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .sanitize import one_line


class SshError(Exception):
    pass


class SshAuthError(SshError):
    pass


class ForbiddenTarget(SshError):
    """The requested host and port are not the ones the API returned."""


@dataclass(frozen=True)
class Credentials:
    """Connection details of one level, exactly as the API returned them."""

    track: str
    idx: int
    kind: str
    host: str
    port: int
    username: str | None
    password: str | None
    password_source: str
    source_level_idx: int | None
    account_id: str | None

    def __repr__(self) -> str:  # never print the password
        return (
            f"Credentials(track={self.track!r}, idx={self.idx}, kind={self.kind!r}, "
            f"host={self.host!r}, port={self.port}, username={self.username!r}, "
            f"password_source={self.password_source!r})"
        )

    @classmethod
    def from_api(cls, d: Mapping[str, Any], track: str, idx: int) -> "Credentials":
        """Build from the answer of `BreachLab.credentials()`."""
        return cls(
            track=str(d.get("track", track)),
            idx=int(d.get("idx", idx)),
            kind=str(d.get("kind", "")),
            host=str(d.get("host") or ""),
            port=int(d.get("port") or 0),
            username=d.get("username"),
            password=d.get("password"),
            password_source=str(d.get("password_source", "")),
            source_level_idx=d.get("source_level_idx"),
            account_id=d.get("account_id"),
        )


_HOSTNAME = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,62}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,62}[A-Za-z0-9])?)*$")
_USERNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    username: str
    track: str
    idx: int


# The operator's own machine and local networks. Documentation ranges are not
# listed: nothing answers there, and the bundled stub uses one.
_LOCAL_NETS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
        "172.16.0.0/12", "192.168.0.0/16", "::/127", "fc00::/7", "fe80::/10",
    )
]


def _is_private(host: str) -> bool:
    if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local", ".internal", ".lan")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    ip = mapped or ip
    return any(ip.version == n.version and ip in n for n in _LOCAL_NETS)


class TargetGuard:
    """Holds the one target the agent may connect to right now."""

    def __init__(self, allow_private_hosts: bool = False) -> None:
        self._allow_private = allow_private_hosts
        self._current: SshTarget | None = None

    def authorize(self, creds: Credentials) -> SshTarget:
        """Turn API credentials into the single allowed target."""
        if not isinstance(creds, Credentials):
            raise ForbiddenTarget("a target can only come from the BreachLab API")
        if creds.kind != "ssh":
            raise SshError(f"level {creds.idx} is a {creds.kind or 'unknown'} target, this starter plays SSH levels only")
        host = creds.host
        is_ip = True
        try:
            ipaddress.ip_address(host)
        except ValueError:
            is_ip = False
        if not is_ip and not _HOSTNAME.match(host):
            raise ForbiddenTarget("the API returned a host that is not a valid host name or address")
        if not (1 <= creds.port <= 65535):
            raise ForbiddenTarget("the API returned a port out of range")
        if not creds.username or not _USERNAME.match(creds.username):
            raise ForbiddenTarget("the API returned a login name this agent will not pass to ssh")
        if _is_private(host) and not self._allow_private:
            raise ForbiddenTarget(
                f"the API returned {host}, a loopback or private address. Refusing to connect "
                "into the operator's own network. Set [ssh] allow_private_hosts for local development."
            )
        self._current = SshTarget(host, creds.port, creds.username, creds.track, creds.idx)
        return self._current

    def check(self, target: SshTarget) -> None:
        if self._current is None or target != self._current:
            raise ForbiddenTarget(
                f"refusing to connect to {target.host}:{target.port}: it is not the host and port "
                "the BreachLab API returned for the level being played"
            )

    @property
    def current(self) -> SshTarget | None:
        return self._current

    def clear(self) -> None:
        self._current = None


@dataclass
class CommandResult:
    output: str
    exit_code: int | None
    timed_out: bool = False
    session_reset: bool = False


class Session(Protocol):
    def run(self, command: str, timeout: float) -> CommandResult: ...
    def close(self) -> None: ...


SessionFactory = Callable[[SshTarget, str], Session]


class SystemSshSession:
    """One level's shell over the system ssh binary."""

    def __init__(
        self,
        target: SshTarget,
        password: str,
        guard: TargetGuard,
        *,
        known_hosts: str | os.PathLike[str],
        binary: str = "ssh",
        connect_timeout: float = 20.0,
        max_capture_bytes: int = 1_000_000,
    ) -> None:
        self._target = target
        self._password = password
        self._guard = guard
        self._known_hosts = str(known_hosts)
        self._binary = binary
        self._connect_timeout = connect_timeout
        self._max_capture = max_capture_bytes
        self._proc: subprocess.Popen[bytes] | None = None
        self._master: int | None = None
        self._slave: int | None = None
        self._open()

    # ── process ─────────────────────────────────────────────────────────────

    def _argv(self) -> list[str]:
        t = self._target
        opts = {
            # Password login only, asked once, typed by this program.
            "PreferredAuthentications": "password,keyboard-interactive",
            "PubkeyAuthentication": "no",
            "NumberOfPasswordPrompts": "1",
            "BatchMode": "no",
            # A known_hosts file of the agent's own: the first key seen for a
            # box is pinned, a changed key refuses the connection.
            "StrictHostKeyChecking": "accept-new",
            "UserKnownHostsFile": self._known_hosts,
            "GlobalKnownHostsFile": "/dev/null",
            "UpdateHostKeys": "no",
            # Nothing may reach back into the operator's machine or hop on.
            "ClearAllForwardings": "yes",
            "ForwardAgent": "no",
            "ForwardX11": "no",
            "ProxyCommand": "none",
            "ProxyJump": "none",
            "PermitLocalCommand": "no",
            "ControlMaster": "no",
            "ControlPath": "none",
            "EscapeChar": "none",
            "ConnectTimeout": str(int(self._connect_timeout)),
            "ServerAliveInterval": "30",
            "LogLevel": "ERROR",
        }
        argv = [self._binary, "-F", "/dev/null", "-T", "-a", "-x", "-p", str(t.port), "-l", t.username]
        for k, v in opts.items():
            argv += ["-o", f"{k}={v}"]
        return argv + ["--", t.host]

    def _open(self) -> None:
        # Checked on every connect, reconnects included.
        self._guard.check(self._target)
        Path(self._known_hosts).parent.mkdir(parents=True, exist_ok=True)
        master, slave = os.openpty()

        def child_setup() -> None:
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        # A minimal environment: no model keys, no BreachLab token, no agent
        # socket and no askpass helper reach the ssh process.
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C", "LC_ALL": "C"}
        if os.environ.get("HOME"):
            env["HOME"] = os.environ["HOME"]
        try:
            self._proc = subprocess.Popen(
                self._argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=child_setup,
                close_fds=True,
            )
        except OSError as e:
            os.close(master)
            os.close(slave)
            raise SshError(f"cannot start {self._binary}: {e.strerror}") from None
        # The slave stays open here so the master never reports a hangup.
        self._slave = slave
        self._master = master
        os.set_blocking(self._proc.stdout.fileno(), False)
        os.set_blocking(master, False)
        try:
            result = self._exchange("true", self._connect_timeout + 10.0, login=True)
        except SshError:
            self.close()
            raise
        if result.exit_code is None:
            self.close()
            raise SshError("the session opened but the remote shell did not answer")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    proc.wait()
        for name in ("_master", "_slave"):
            fd = getattr(self, name)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, None)

    # ── protocol ────────────────────────────────────────────────────────────

    def _exchange(self, command: str, timeout: float, login: bool = False) -> CommandResult:
        """Send one command, read until its end marker.

        The command travels inside a quoted here-document, so the remote shell
        takes it verbatim, and it runs with stdin closed so it cannot swallow
        the lines that follow. The markers carry a nonce made for this command.
        """
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        nonce = secrets.token_hex(12)
        eof = f"BL_EOF_{nonce}"
        start, end = f"BL_START_{nonce}", f"BL_END_{nonce}"
        if eof in command:
            raise SshError("command rejected")
        script = (
            f"__bl_cmd=$(cat <<'{eof}'\n{command}\n{eof}\n)\n"
            f"printf '%s\\n' '{start}'; eval \"$__bl_cmd\" </dev/null 2>&1; "
            f"printf '\\n%s:%s\\n' '{end}' \"$?\"\n"
        )
        try:
            proc.stdin.write(script.encode("utf-8"))
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise SshError(self._death_reason(b"")) from None

        out_fd = proc.stdout.fileno()
        end_re = re.compile(rb"\n?" + end.encode() + rb":(\d+)\r?\n")
        buf = b""
        tty = b""
        password_sent = not login
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return CommandResult(self._body(buf, start), None, timed_out=True)
            fds = [out_fd] + ([self._master] if login and self._master is not None else [])
            ready, _, _ = select.select(fds, [], [], min(remaining, 0.5))
            if self._master is not None and self._master in ready:
                try:
                    chunk = os.read(self._master, 4096)
                except OSError:
                    chunk = b""
                tty = (tty + chunk)[-4096:]
                if re.search(rb"(?i)pass(word|phrase)[^\n]*:\s*$", tty):
                    if password_sent:
                        raise SshAuthError("the box asked for the password again: login refused")
                    os.write(self._master, self._password.encode("utf-8") + b"\n")
                    password_sent = True
                    tty = b""
            if out_fd in ready:
                try:
                    chunk = os.read(out_fd, 65536)
                except BlockingIOError:
                    chunk = None
                if chunk == b"":
                    raise SshError(self._death_reason(buf + tty))
                if chunk:
                    buf += chunk
                    if len(buf) > self._max_capture:
                        # Keep the head, and enough tail to still see the marker.
                        buf = buf[: self._max_capture // 2] + b"\n[...]\n" + buf[-(self._max_capture // 4):]
                    m = end_re.search(buf)
                    if m:
                        return CommandResult(self._body(buf[: m.start()], start), int(m.group(1)))
            if not ready and proc.poll() is not None:
                raise SshError(self._death_reason(buf + tty))

    @staticmethod
    def _body(buf: bytes, start: str) -> str:
        marker = start.encode() + b"\n"
        at = buf.find(marker)
        body = buf[at + len(marker):] if at >= 0 else b""
        return body.decode("utf-8", errors="replace")

    def _death_reason(self, tail: bytes) -> str:
        text = tail.decode("utf-8", errors="replace")
        if re.search(r"(?i)permission denied|authentication failed|too many authentication", text):
            raise SshAuthError("login refused by the box")
        return "the ssh session ended" + (f": {one_line(text[-300:])}" if text.strip() else "")

    def run(self, command: str, timeout: float) -> CommandResult:
        reset = False
        if self._proc is None or self._proc.poll() is not None:
            self.close()
            self._open()
            reset = True
        try:
            result = self._exchange(command, timeout)
        except SshAuthError:
            self.close()
            raise
        except SshError:
            # The shell is gone, for example the command was `exit`. The next
            # command gets a fresh login.
            self.close()
            return CommandResult("", None, session_reset=True)
        if result.timed_out:
            # Without a terminal there is no interrupt key. Drop the session;
            # the next command logs in again.
            self.close()
            result.session_reset = True
        result.session_reset = result.session_reset or reset
        return result
