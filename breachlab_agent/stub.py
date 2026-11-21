"""A local stand in for BreachLab: API server, level boxes and a scripted model.

Used by `--dry-run` and by the tests. Nothing here talks to the network
beyond 127.0.0.1, and nothing here is a real level: the flags are made up.
The server follows the v2 contract closely enough to exercise the client:
bearer auth, scopes, problem+json errors with `code` and `doc_url`, rate
limit headers, Retry-After, idempotency keys on run writes, the stop
switches, and an edge that refuses the default Python-urllib User-Agent.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .models import ModelReply, ToolCall, Usage
from .ssh import CommandResult, ForbiddenTarget, SshAuthError, SshTarget, TargetGuard

STUB_TOKEN = "blp_stub_0000000000000000"
STUB_HOST = "203.0.113.10"  # TEST-NET-3, reserved for documentation
STUB_PORT = 2222
DOC = "https://breachlab.org/docs/api/errors/"

_STATUS = {
    "unauthorized": 401, "insufficient_scope": 403, "not_found": 404, "invalid_request": 400,
    "invalid_format": 422, "rate_limited": 429, "cooldown": 429, "already_solved": 409,
    "agents_disabled": 503, "run_closed": 409, "run_limit_reached": 429, "level_locked": 403,
    "play_disabled": 503, "idempotency_key_reused": 422, "internal_error": 500,
    "credentials_unavailable": 503,
}


@dataclass
class StubLevel:
    idx: int
    title: str
    description: str
    flag: str
    files: dict[str, str] = field(default_factory=dict)  # command -> output


def default_levels() -> list[StubLevel]:
    return [
        StubLevel(0, "Stub level 0", "The password for the next level is in a file in your home directory.",
                  "stubflag0_Qm3xT7pLw9", {"ls": "readme\n", "ls -a": ".\n..\nreadme\n", "cat readme": "stubflag0_Qm3xT7pLw9\n"}),
        StubLevel(1, "Stub level 1", "The password is in a hidden file.",
                  "stubflag1_Zk8vB2nRc4", {"ls": "", "ls -a": ".\n..\n.secret\n", "cat .secret": "stubflag1_Zk8vB2nRc4\n"}),
    ]


class StubState:
    def __init__(self, levels: list[StubLevel] | None = None) -> None:
        self.lock = threading.Lock()
        self.levels = levels if levels is not None else default_levels()
        self.token = STUB_TOKEN
        self.scopes = {"read", "submit", "agent", "play"}
        self.is_agent = True
        self.solved: set[int] = set()
        self.runs: dict[str, dict[str, Any]] = {}
        self.idem: dict[str, tuple[str, int, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.agents_disabled = False
        self.play_disabled = False
        # [(path prefix, code, retry_after)], consumed one per matching request.
        self.inject: list[tuple[str, str, int | None]] = []
        self.cred_host: dict[int, tuple[str, int]] = {}
        self.entry_password = "ghost0"
        self.track = {"slug": "ghost", "name": "Ghost I (stub)"}

    def level(self, idx: int) -> StubLevel | None:
        return next((lv for lv in self.levels if lv.idx == idx), None)

    def status_of(self, idx: int) -> str:
        if idx in self.solved:
            return "solved"
        first = min(lv.idx for lv in self.levels)
        return "available" if idx == first or (idx - 1) in self.solved else "locked"


class _Handler(BaseHTTPRequestHandler):
    server_version = "stub"
    protocol_version = "HTTP/1.1"
    state: StubState

    def log_message(self, *args: Any) -> None:
        pass

    # ── plumbing ────────────────────────────────────────────────────────────

    def _send(self, status: int, body: Any, headers: dict[str, str] | None = None, problem: bool = False) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/problem+json" if problem else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Request-Id", "req_stub")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _problem(self, code: str, detail: str = "", retry_after: int | None = None) -> None:
        status = _STATUS[code]
        body: dict[str, Any] = {
            "type": DOC + code, "title": code, "status": status, "code": code,
            "detail": detail or code, "doc_url": DOC + code, "request_id": "req_stub",
        }
        headers = {}
        if retry_after is not None:
            body["retry_after_seconds"] = retry_after
            headers["Retry-After"] = str(retry_after)
        self._send(status, body, headers, problem=True)

    def _handle(self, method: str) -> None:
        st = self.state
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        ua = self.headers.get("User-Agent", "")
        with st.lock:
            st.requests.append({
                "method": method, "path": path, "user_agent": ua,
                "idempotency_key": self.headers.get("Idempotency-Key"), "body": raw.decode("utf-8", "replace"),
            })
            # The edge: an HTML refusal, not a problem document.
            if not ua or ua.lower().startswith("python-urllib"):
                page = b"<html><body>error code: 1010</body></html>"
                self.send_response(403)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if not path.startswith("/api/v2/"):
                return self._problem("not_found")
            path = path[len("/api/v2"):]
            if self.headers.get("Authorization") != f"Bearer {st.token}":
                return self._problem("unauthorized")
            for i, (prefix, code, retry_after) in enumerate(st.inject):
                if path.startswith(prefix):
                    del st.inject[i]
                    return self._problem(code, retry_after=retry_after)
            body: Any = None
            if raw:
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._problem("invalid_request", "Body must be JSON")
            self._route(method, path, body, raw)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    # ── routes ──────────────────────────────────────────────────────────────

    def _need(self, scope: str) -> bool:
        st = self.state
        if scope not in st.scopes:
            self._problem("insufficient_scope")
            return False
        if st.agents_disabled and st.is_agent:
            self._problem("agents_disabled")
            return False
        return True

    def _level_json(self, lv: StubLevel) -> dict[str, Any]:
        status = self.state.status_of(lv.idx)
        return {
            "object": "level", "track": self.state.track["slug"], "idx": lv.idx, "title": lv.title,
            "description": lv.description, "difficulty": "easy", "points": 100, "first_blood_bonus": 0,
            "status": status, "access": "free",
        }

    def _run_json(self, run: dict[str, Any]) -> dict[str, Any]:
        solved = [{"level_idx": i, "points": 100, "solved_at": "2026-01-01T00:00:00.000Z"} for i in run["solved"]]
        return {
            "object": "run", "id": run["id"], "track": run["track"], "model": run["model"],
            "harness": run["harness"], "harness_version": run["harness_version"], "state": run["state"],
            "reported_status": run["reported_status"], "step_count": len(run["steps"]),
            "tokens_used": run["tokens_used"], "cost_usd": run["cost_usd"],
            "result": {"solved_levels": solved, "points": 100 * len(solved)},
        }

    def _route(self, method: str, path: str, body: Any, raw: bytes) -> None:
        st = self.state
        slug = st.track["slug"]

        if method == "GET" and path == "/me":
            if not self._need("read"):
                return
            return self._send(200, {
                "object": "account", "username": "stub-agent", "is_agent": st.is_agent,
                "agent": {"name": "Stub Agent", "model": None} if st.is_agent else None,
                "points": 100 * len(st.solved), "solved": len(st.solved),
            })
        if method == "GET" and path == "/token":
            if not self._need("read"):
                return
            return self._send(200, {
                "object": "token", "scopes": sorted(st.scopes), "created_at": "2026-01-01T00:00:00.000Z",
                "expires_at": "2027-01-01T00:00:00.000Z", "is_agent": st.is_agent,
            })
        if method == "GET" and path == "/me/progress":
            if not self._need("read"):
                return
            nxt = [lv.idx for lv in st.levels if st.status_of(lv.idx) == "available"]
            return self._send(200, {"object": "list", "has_more": False, "next_cursor": None, "prev_cursor": None, "data": [
                {"object": "track_progress", "track": slug, "name": st.track["name"], "access": "free",
                 "level_count": len(st.levels), "solved": len(st.solved), "next_level_idx": min(nxt) if nxt else None},
            ]})
        m = re.fullmatch(rf"/tracks/{slug}/levels/(\d+)", path)
        if method == "GET" and m:
            if not self._need("read"):
                return
            lv = st.level(int(m.group(1)))
            return self._send(200, self._level_json(lv)) if lv else self._problem("not_found", "Level not found")
        if method == "GET" and path == "/tracks":
            if not self._need("read"):
                return
            return self._send(200, {"object": "list", "has_more": False, "next_cursor": None, "prev_cursor": None, "data": [
                {"object": "track", "slug": slug, "name": st.track["name"], "status": "live",
                 "level_count": len(st.levels), "access": "free", "api_playable": True},
            ]})
        if method == "GET" and path == f"/tracks/{slug}/levels":
            if not self._need("read"):
                return
            return self._send(200, {"object": "list", "has_more": False, "next_cursor": None, "prev_cursor": None,
                                    "data": [self._level_json(lv) for lv in st.levels]},
                              {"RateLimit-Limit": "60", "RateLimit-Remaining": "59", "RateLimit-Reset": "60"})
        m = re.fullmatch(rf"/tracks/{slug}/levels/(\d+)/credentials", path)
        if method == "GET" and m:
            if not self._need("play"):
                return
            if st.play_disabled:
                return self._problem("play_disabled")
            lv = st.level(int(m.group(1)))
            if lv is None:
                return self._problem("not_found", "Level not found")
            if st.status_of(lv.idx) == "locked":
                return self._problem("level_locked")
            host, port = st.cred_host.get(lv.idx, (STUB_HOST, STUB_PORT))
            first = lv.idx == min(x.idx for x in st.levels)
            return self._send(200, {
                "object": "level_credentials", "track": slug, "idx": lv.idx, "kind": "ssh", "host": host, "port": port,
                "username": f"ghost{lv.idx}", "command": f"ssh ghost{lv.idx}@{host} -p {port}", "url": None,
                "password": st.entry_password if first else None,
                "password_source": "public" if first else "previous_flag",
                "source_level_idx": None if first else lv.idx - 1, "account_id": None, "access_header": None,
            })
        if method == "POST" and path == "/submit":
            if not self._need("submit"):
                return
            flag = (body or {}).get("flag")
            if not isinstance(flag, str) or not (4 <= len(flag) <= 128):
                return self._problem("invalid_format", "flag must be a 4-128 character string")
            run = None
            if "run_id" in (body or {}):
                run = st.runs.get(body["run_id"])
                if run is None:
                    return self._problem("not_found", "Run not found")
                if run["state"] != "open":
                    return self._problem("run_closed")
            lv = next((x for x in st.levels if x.flag == flag), None)
            if lv is not None and lv.idx in st.solved:
                return self._problem("already_solved")
            if lv is None or st.status_of(lv.idx) != "available":
                if run is not None:
                    run["attempts"].append("incorrect")
                return self._send(200, {"result": "incorrect"})
            st.solved.add(lv.idx)
            if run is not None:
                run["attempts"].append("correct")
                run["solved"].append(lv.idx)
            return self._send(200, {
                "result": "correct", "track": slug, "level_idx": lv.idx, "points_awarded": 100,
                "first_blood": False, "track_completed": len(st.solved) == len(st.levels),
            })
        if method == "POST" and (path == "/runs" or path.startswith("/runs/")):
            if not self._need("agent"):
                return
            key = self.headers.get("Idempotency-Key")
            if key and key in st.idem:
                sig, status, stored = st.idem[key]
                if sig != f"{path}|{raw.decode()}":
                    return self._problem("idempotency_key_reused")
                return self._send(status, stored, {"Idempotent-Replayed": "true"})
            out = self._run_write(path, body or {})
            if out is not None:
                status, payload = out
                if key:
                    st.idem[key] = (f"{path}|{raw.decode()}", status, payload)
                self._send(status, payload)
            return
        m = re.fullmatch(r"/runs/([0-9a-f-]{36})", path)
        if method == "GET" and m:
            if not self._need("read"):
                return
            run = st.runs.get(m.group(1))
            return self._send(200, self._run_json(run)) if run else self._problem("not_found", "Run not found")
        self._problem("not_found")

    def _run_write(self, path: str, body: dict[str, Any]) -> tuple[int, Any] | None:
        st = self.state
        if path == "/runs":
            if body.get("track") != st.track["slug"]:
                self._problem("not_found", "Track not found")
                return None
            model = body.get("model")
            if model is not None and not re.fullmatch(r"[a-z]+/[a-z0-9][a-z0-9._:-]*", str(model)):
                self._problem("invalid_format", "model must be provider/model")
                return None
            run = {
                "id": str(uuid.uuid4()), "track": body["track"], "model": model, "harness": body.get("harness"),
                "harness_version": body.get("harness_version"), "state": "open", "reported_status": None,
                "steps": [], "attempts": [], "solved": [], "tokens_used": None, "cost_usd": None,
            }
            st.runs[run["id"]] = run
            return 201, self._run_json(run)
        m = re.fullmatch(r"/runs/([0-9a-f-]{36})/(steps|finish)", path)
        run = st.runs.get(m.group(1)) if m else None
        if run is None:
            self._problem("not_found", "Run not found")
            return None
        if run["state"] != "open":
            self._problem("run_closed")
            return None
        if m.group(2) == "steps":
            steps = body.get("steps")
            ts_re = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z"
            ok = isinstance(steps, list) and 1 <= len(steps) <= 100
            prev = 0
            for s in steps if ok else []:
                if not (isinstance(s, dict) and isinstance(s.get("seq"), int) and s["seq"] > prev
                        and isinstance(s.get("ts"), str) and re.fullmatch(ts_re, s["ts"])
                        and s.get("type") in ("thought", "command", "output", "submit", "note")
                        and isinstance(s.get("content"), str)):
                    ok = False
                    break
                prev = s["seq"]
            if not ok:
                self._problem("invalid_format", "steps are malformed")
                return None
            last = run["steps"][-1]["seq"] if run["steps"] else 0
            fresh = [s for s in steps if s["seq"] > last]
            run["steps"].extend(fresh)
            return 200, {"object": "step_batch", "accepted": len(fresh), "duplicates": len(steps) - len(fresh),
                         "dropped": 0, "truncated": 0, "step_count": len(run["steps"]), "steps_truncated": False}
        if body.get("status") not in ("solved", "gave_up", "error", "timeout"):
            self._problem("invalid_format", "status must be one of solved, gave_up, error, timeout")
            return None
        run.update(state="finished", reported_status=body["status"],
                   tokens_used=body.get("tokens_used"), cost_usd=body.get("cost_usd"))
        return 200, self._run_json(run)


class StubServer:
    """The stub API on 127.0.0.1 and a free port."""

    def __init__(self, state: StubState | None = None) -> None:
        self.state = state or StubState()
        handler = type("Handler", (_Handler,), {"state": self.state})
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "StubServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# ── boxes ───────────────────────────────────────────────────────────────────


class FakeSshSession:
    """A level box that answers from a table. Opens no connection of any kind."""

    def __init__(self, target: SshTarget, password: str, guard: TargetGuard, state: StubState) -> None:
        guard.check(target)  # the same rule the real session enforces
        level = state.level(target.idx)
        expected = state.entry_password if level and level.idx == min(x.idx for x in state.levels) else (
            state.level(target.idx - 1).flag if level and state.level(target.idx - 1) else None
        )
        if level is None or password != expected:
            raise SshAuthError("login refused by the box")
        self.level = level
        self.commands: list[str] = []
        self.closed = False

    def run(self, command: str, timeout: float) -> CommandResult:
        self.commands.append(command)
        if command in self.level.files:
            return CommandResult(self.level.files[command], 0)
        return CommandResult(f"sh: {command.split()[0] if command.split() else ''}: command not found\n", 127)

    def close(self) -> None:
        self.closed = True


def fake_session_factory(guard: TargetGuard, state: StubState, opened: list[FakeSshSession] | None = None):
    def factory(target: SshTarget, password: str) -> FakeSshSession:
        s = FakeSshSession(target, password, guard, state)
        if opened is not None:
            opened.append(s)
        return s

    return factory


# ── models ──────────────────────────────────────────────────────────────────


class ScriptedModel:
    """Replays a fixed list of replies. For tests."""

    name = "scripted"
    declared_model: str | None = "other/scripted"

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def complete(self, system: str, messages: list[dict[str, Any]]) -> ModelReply:
        self.calls.append((system, [dict(m) for m in messages]))
        if not self._replies:
            return ModelReply(text="GIVE_UP")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return item


_n = 0


def cmd(command: str, tokens: int = 100) -> ModelReply:
    """A reply that runs one command."""
    global _n
    _n += 1
    return ModelReply(tool_calls=[ToolCall(f"call_{_n}", command)], usage=Usage(tokens, 10))


def say(text: str, tokens: int = 100) -> ModelReply:
    return ModelReply(text=text, usage=Usage(tokens, 10))


class DemoModel:
    """The dry run's stand in for a model: it lists files, reads them, reports.

    It is a fixed routine, not a language model, and it solves only the stub
    levels. It exists so a dry run exercises the same loop a real model drives.
    """

    name = "dry-run-demo"
    declared_model: str | None = "other/dry-run-demo"

    def complete(self, system: str, messages: list[dict[str, Any]]) -> ModelReply:
        outputs = [m["text"] for m in messages if m["role"] == "tool"]
        if not outputs:
            return cmd("ls -a", 0)
        if len(outputs) == 1:
            names = [n for n in outputs[0].splitlines()[1:-1] if n not in (".", "..")]
            return cmd(f"cat {names[0]}" if names else "ls", 0)
        body = outputs[-1].splitlines()[1:-1]
        return say(f"FLAG: {body[0].strip()}" if body else "GIVE_UP", 0)


def check_guard_refuses(guard: TargetGuard) -> bool:
    """Dry run self check: a host the API did not return must be refused."""
    try:
        guard.check(SshTarget("198.51.100.7", 22, "root", "ghost", 0))
    except ForbiddenTarget:
        return True
    return False
