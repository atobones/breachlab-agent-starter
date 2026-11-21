"""The real session class, driven against a fake `ssh` program.

The fake is a local script: it asks for a password on the controlling
terminal the way OpenSSH does, then becomes a plain `sh`. No network and no
SSH server are involved.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest

from breachlab_agent.ssh import Credentials, ForbiddenTarget, SshAuthError, SshError, SshTarget, SystemSshSession, TargetGuard

FAKE_SSH = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys, termios
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "argv.json"), "w") as f:
        json.dump({{"argv": sys.argv[1:], "env": sorted(os.environ)}}, f)
    expected = open(os.path.join(here, "password")).read()
    tty = os.open("/dev/tty", os.O_RDWR)
    old = termios.tcgetattr(tty)
    new = termios.tcgetattr(tty)
    new[3] &= ~termios.ECHO
    termios.tcsetattr(tty, termios.TCSANOW, new)
    os.write(tty, b"ghost0@box's password: ")
    typed = b""
    while not typed.endswith(b"\\n"):
        typed += os.read(tty, 1)
    termios.tcsetattr(tty, termios.TCSANOW, old)
    os.close(tty)
    if typed.decode().rstrip("\\n") != expected:
        sys.stderr.write("ghost0@box: Permission denied (password).\\n")
        sys.exit(255)
    os.chdir(os.path.join(here, "home"))
    os.execv("/bin/sh", ["sh"])
    """
)


def creds(host="203.0.113.10", port=2222, username="ghost0", kind="ssh") -> Credentials:
    return Credentials("ghost", 0, kind, host, port, username, "ghost0", "public", None, None)


class Guard(unittest.TestCase):
    def test_only_api_credentials_become_a_target(self) -> None:
        g = TargetGuard()
        t = g.authorize(creds())
        self.assertEqual((t.host, t.port, t.username), ("203.0.113.10", 2222, "ghost0"))
        g.check(t)
        with self.assertRaises(ForbiddenTarget):
            g.authorize({"host": "203.0.113.10", "port": 22, "username": "x", "kind": "ssh"})  # type: ignore[arg-type]

    def test_anything_but_the_current_target_is_refused(self) -> None:
        g = TargetGuard()
        t = g.authorize(creds())
        for other in (
            SshTarget("203.0.113.11", 2222, "ghost0", "ghost", 0),
            SshTarget("203.0.113.10", 22, "ghost0", "ghost", 0),
            SshTarget("203.0.113.10", 2222, "root", "ghost", 0),
        ):
            with self.assertRaises(ForbiddenTarget):
                g.check(other)
        g.authorize(creds(port=2223))  # the next level replaces the target
        with self.assertRaises(ForbiddenTarget):
            g.check(t)
        g.clear()
        with self.assertRaises(ForbiddenTarget):
            g.check(t)

    def test_bad_values_from_the_api_are_refused(self) -> None:
        bad = [
            creds(host="-oProxyCommand=id"), creds(host="a b"), creds(host="box;id"), creds(host=""),
            creds(host="$(id).example"), creds(port=0), creds(port=70000),
            creds(username="-lroot"), creds(username="a b"), creds(username=None), creds(username="x\ny"),
            creds(host="127.0.0.1"), creds(host="localhost"), creds(host="192.168.0.1"), creds(host="10.1.1.1"),
            creds(host="172.16.5.5"), creds(host="169.254.169.254"), creds(host="::1"), creds(host="fd00::1"),
            creds(host="::ffff:127.0.0.1"), creds(host="100.64.0.1"), creds(host="printer.local"),
        ]
        for c in bad:
            with self.subTest(c=c), self.assertRaises(ForbiddenTarget):
                TargetGuard().authorize(c)
        for good in (creds(host="204.168.229.209"), creds(host="box.breachlab.org"), creds(host="2001:db8::1")):
            TargetGuard().authorize(good)
        TargetGuard(allow_private_hosts=True).authorize(creds(host="127.0.0.1"))

    def test_web_levels_are_not_ssh_targets(self) -> None:
        with self.assertRaises(SshError):
            TargetGuard().authorize(creds(kind="web"))


@unittest.skipIf(sys.platform == "win32", "needs a pseudo-terminal")
class SystemSession(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        os.mkdir(os.path.join(d, "home"))
        with open(os.path.join(d, "home", "readme"), "w") as f:
            f.write("flag_from_the_box\n")
        with open(os.path.join(d, "password"), "w") as f:
            f.write("ghost0")
        self.binary = os.path.join(d, "ssh")
        with open(self.binary, "w") as f:
            f.write(FAKE_SSH.format(python=sys.executable))
        os.chmod(self.binary, os.stat(self.binary).st_mode | stat.S_IXUSR)
        self.guard = TargetGuard()
        self.target = self.guard.authorize(creds())
        self.sessions: list[SystemSshSession] = []

    def open(self, password="ghost0", target=None) -> SystemSshSession:
        s = SystemSshSession(target or self.target, password, self.guard,
                             known_hosts=os.path.join(self.tmp.name, "state", "known_hosts"),
                             binary=self.binary, connect_timeout=5)
        self.sessions.append(s)
        self.addCleanup(s.close)
        return s

    def test_login_commands_exit_codes_and_state(self) -> None:
        s = self.open()
        r = s.run("cat readme", 10)
        self.assertEqual((r.output, r.exit_code, r.timed_out), ("flag_from_the_box\n", 0, False))
        self.assertEqual(s.run("ls nope 2>/dev/null; false", 10).exit_code, 1)
        self.assertEqual(s.run("echo to-stderr 1>&2", 10).output, "to-stderr\n")
        s.run("mkdir sub && cd sub && X=kept", 10)
        r = s.run("basename \"$PWD\"; echo $X", 10)
        self.assertEqual(r.output, "sub\nkept\n")

    def test_commands_are_taken_verbatim(self) -> None:
        s = self.open()
        tricky = "printf '%s|' \"a b\" 'c$HOME' \\\\ \"$(echo sub)\" '!x' ; echo\necho second-line"
        r = s.run(tricky, 10)
        self.assertEqual(r.output, "a b|c$HOME|\\|sub|!x|\nsecond-line\n")
        # A command that reads stdin gets none, so it cannot eat what follows.
        self.assertEqual(s.run("cat; echo done", 10).output, "done\n")
        # Text that looks like a marker does not end the capture.
        r = s.run("echo BL_END_000000000000000000000000:0; echo after", 10)
        self.assertEqual(r.output, "BL_END_000000000000000000000000:0\nafter\n")

    def test_wrong_password_is_an_auth_error(self) -> None:
        with self.assertRaises(SshAuthError):
            self.open(password="nope")

    def test_password_is_typed_into_the_prompt_never_passed_along(self) -> None:
        with open(os.path.join(self.tmp.name, "password"), "w") as f:
            f.write("s3cret-level-pass")
        self.open(password="s3cret-level-pass")
        with open(os.path.join(self.tmp.name, "argv.json")) as f:
            seen = json.load(f)
        argv = seen["argv"]
        self.assertNotIn("s3cret-level-pass", " ".join(argv))
        self.assertEqual(argv[-2:], ["--", "203.0.113.10"])
        self.assertEqual(argv[:2], ["-F", "/dev/null"])
        self.assertIn("-T", argv)
        for opt in ("ClearAllForwardings=yes", "ForwardAgent=no", "ProxyCommand=none", "ProxyJump=none",
                    "EscapeChar=none", "PermitLocalCommand=no", "StrictHostKeyChecking=accept-new",
                    "NumberOfPasswordPrompts=1", "PubkeyAuthentication=no", "GlobalKnownHostsFile=/dev/null"):
            self.assertIn(opt, argv)
        self.assertTrue(any(a.startswith("UserKnownHostsFile=" + self.tmp.name) for a in argv))
        allowed = {"PATH", "LANG", "LC_ALL", "HOME", "__CF_USER_TEXT_ENCODING"}  # the last one is added by macOS
        self.assertEqual(set(seen["env"]) - allowed, set())

    def test_model_keys_and_token_do_not_reach_the_ssh_process(self) -> None:
        os.environ["BREACHLAB_TOKEN"] = "blp_test_do_not_leak"
        os.environ["MODEL_API_KEY"] = "test-do-not-leak"
        self.addCleanup(os.environ.pop, "BREACHLAB_TOKEN")
        self.addCleanup(os.environ.pop, "MODEL_API_KEY")
        s = self.open()
        out = s.run("env", 10).output
        self.assertNotIn("do_not_leak", out)
        self.assertNotIn("do-not-leak", out)

    def test_timeout_drops_the_session_and_the_next_command_logs_in_again(self) -> None:
        s = self.open()
        s.run("cd /tmp", 10)
        r = s.run("echo started; sleep 30", 1)
        self.assertTrue(r.timed_out and r.session_reset)
        self.assertIsNone(r.exit_code)
        self.assertEqual(r.output, "started\n")
        r = s.run("basename \"$PWD\"", 10)
        self.assertEqual((r.output, r.exit_code, r.session_reset), ("home\n", 0, True))

    def test_shell_exit_is_survived(self) -> None:
        s = self.open()
        r = s.run("exit 3", 10)
        self.assertTrue(r.session_reset)
        self.assertIsNone(r.exit_code)
        self.assertEqual(s.run("echo back", 10).output, "back\n")

    def test_refuses_to_start_for_a_target_the_guard_does_not_hold(self) -> None:
        other = SshTarget("198.51.100.7", 22, "root", "ghost", 0)
        with self.assertRaises(ForbiddenTarget):
            self.open(target=other)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "argv.json")))

    def test_reconnect_is_refused_once_the_agent_moved_on(self) -> None:
        s = self.open()
        s.run("exit", 10)
        self.guard.authorize(creds(port=2223))
        with self.assertRaises(ForbiddenTarget):
            s.run("echo hi", 10)

    def test_huge_output_is_bounded(self) -> None:
        s = SystemSshSession(self.target, "ghost0", self.guard, known_hosts=os.path.join(self.tmp.name, "kh"),
                             binary=self.binary, connect_timeout=5, max_capture_bytes=20000)
        self.addCleanup(s.close)
        r = s.run("yes AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA | head -n 20000; echo tail-marker", 20)
        self.assertEqual(r.exit_code, 0)
        self.assertLess(len(r.output), 40000)
        self.assertTrue(r.output.endswith("tail-marker\n"))

    def test_missing_binary(self) -> None:
        with self.assertRaises(SshError):
            SystemSshSession(self.target, "x", self.guard, known_hosts=os.path.join(self.tmp.name, "kh"),
                             binary=os.path.join(self.tmp.name, "no-such-ssh"))
