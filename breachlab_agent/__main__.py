"""Command line entry point: python -m breachlab_agent"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.parse
from pathlib import Path

from breachlab import ApiError, BreachLab, BreachLabError, NetworkError

from . import HARNESS_NAME, USER_AGENT, __version__
from .agent import KILL_SWITCH_CODES, Agent, MissingScopes, NotAnAgent, RunSummary, describe
from .config import Config, ConfigError, load_config, read_token
from .models import ModelError, build_model
from .sanitize import one_line, redact
from .ssh import SshTarget, SystemSshSession, TargetGuard
from .state import FlagStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_STOPPED_BY_BREACHLAB = 75  # EX_TEMPFAIL: a stop switch is on, try later


def make_logger(secrets: list[str]):
    def log(message: str) -> None:
        print(redact(one_line(message, 400), secrets), file=sys.stderr, flush=True)

    return log


def report(summary: RunSummary, log) -> int:
    log("")
    log(f"stopped: {summary.reason}")
    log(f"reported status: {summary.status}")
    log(f"levels solved in this run: {summary.solved_levels or 'none'}")
    cost = f"{summary.cost_usd:.4f} USD (estimate)" if summary.cost_known else "not tracked, set prices in config.toml"
    log(f"model turns: {summary.steps}, tokens: {summary.tokens}, cost: {cost}, time: {summary.seconds:.0f} s")
    if summary.server_points is not None:
        log(f"points the server scored for this run: {summary.server_points}")
    if summary.run_id:
        log(f"run id: {summary.run_id}")
    if summary.kill_switch:
        log(f"BreachLab has switched this off for now ({summary.kill_switch}). Nothing is wrong on your side. Try again later.")
        return EXIT_STOPPED_BY_BREACHLAB
    return EXIT_ERROR if summary.status == "error" else EXIT_OK


def make_client(base_url: str, token: str, cfg: Config, log) -> BreachLab:
    """The BreachLab SDK client, with this harness's own User-Agent.

    It keeps the token in the Authorization header, on HTTPS only (plain HTTP
    is accepted for localhost, which the bundled stub uses), never follows a
    redirect, and sits through a 429 for as long as Retry-After says, up to
    the wait the operator allows.
    """
    return BreachLab(
        token, base_url,
        timeout=cfg.api.request_timeout_seconds,
        user_agent=USER_AGENT,
        max_retry_wait=cfg.api.max_retry_wait_seconds,
        notify=log,
        auto_token=False,
    )


def dry_run(cfg: Config, log) -> int:
    """The whole loop against the bundled stub: no model, no SSH, no network."""
    from .stub import DemoModel, StubServer, check_guard_refuses, fake_session_factory

    log("dry run: bundled stub API on 127.0.0.1, scripted demo model, simulated boxes")
    cfg.agent.track = "ghost"
    with StubServer() as server, tempfile.TemporaryDirectory() as tmp:
        client = make_client(server.base_url, server.state.token, cfg, log)
        guard = TargetGuard()
        agent = Agent(
            cfg, client, DemoModel(), fake_session_factory(guard, server.state), guard,
            FlagStore(tmp, "dry-run"), log=log,
        )
        summary = agent.run()
        calls = [f"{r['method']} {r['path']}" for r in server.state.requests]
    log("")
    log("API calls made:")
    for line in calls:
        log(f"  {line}")
    log(f"refuses a host the API did not return: {'yes' if check_guard_refuses(guard) else 'NO'}")
    code = report(summary, log)
    ok = code == EXIT_OK and summary.status == "solved" and len(summary.solved_levels) == 2
    log("dry run passed" if ok else "dry run FAILED")
    return EXIT_OK if ok else EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="breachlab-agent", description="Play BreachLab as a declared AI agent.")
    parser.add_argument("--config", default="config.toml", help="path to config.toml (default: ./config.toml)")
    parser.add_argument("--track", help="track slug, overrides the config")
    parser.add_argument("--max-levels", type=int, help="stop after this many levels")
    parser.add_argument("--dry-run", action="store_true", help="run the loop against the bundled stub, no model and no SSH")
    parser.add_argument("--version", action="version", version=f"{HARNESS_NAME} {__version__}")
    args = parser.parse_args(argv)

    # The token, and whatever your model reads its key from: every variable
    # whose name ends in _KEY, _TOKEN or _SECRET is kept out of the log.
    secrets: list[str] = [
        v for k, v in os.environ.items()
        if v and (k == "BREACHLAB_TOKEN" or k.upper().endswith(("_KEY", "_TOKEN", "_SECRET")))
    ]
    log = make_logger(secrets)
    try:
        cfg = load_config(args.config)
        if args.track:
            cfg.agent.track = args.track
        if args.max_levels is not None:
            cfg.agent.max_levels = args.max_levels
        if args.dry_run:
            return dry_run(cfg, log)

        token = read_token()
        model = build_model(cfg.model)
        client = make_client(cfg.api.base_url, token, cfg, log)
    except (ConfigError, ModelError, ValueError, BreachLabError) as e:
        log(f"configuration problem: {e}")
        return EXIT_CONFIG

    state_dir = Path(cfg.agent.state_dir)
    guard = TargetGuard(allow_private_hosts=cfg.ssh.allow_private_hosts)

    def open_session(target: SshTarget, password: str) -> SystemSshSession:
        return SystemSshSession(
            target, password, guard,
            known_hosts=state_dir / "known_hosts",
            binary=cfg.ssh.binary,
            connect_timeout=cfg.ssh.connect_timeout_seconds,
        )

    host = urllib.parse.urlsplit(cfg.api.base_url).hostname or "api"
    log(f"{HARNESS_NAME} {__version__}: track {cfg.agent.track}, model {model.name}")
    if cfg.budgets.max_cost_usd > 0 and cfg.model.price_input_per_mtok <= 0 and cfg.model.price_output_per_mtok <= 0:
        log("note: no price is set for this model, so the cost budget applies only if your model reports a cost. The step, time and token budgets still do.")
    try:
        # The flag store is keyed by API host and agent, resolved on first use.
        me_name = client.me().get("username", "agent")
        agent = Agent(cfg, client, model, open_session, guard, FlagStore(state_dir, f"{host}/{me_name}"), log=log)
        summary = agent.run()
    except (NotAnAgent, MissingScopes) as e:
        log(str(e))
        return EXIT_CONFIG
    except ApiError as e:
        if e.code in KILL_SWITCH_CODES:
            log(f"BreachLab has switched this off for now ({e.code}). Try again later.")
            return EXIT_STOPPED_BY_BREACHLAB
        hint = ""
        if e.code == "unauthorized":
            hint = " Check BREACHLAB_TOKEN: it may be mistyped, expired or revoked."
        elif e.code.startswith("http_"):
            hint = " The request was refused before it reached the API."
        log(f"BreachLab API error: {describe(e)}.{hint}")
        return EXIT_ERROR
    except NetworkError as e:
        log(f"cannot reach the BreachLab API: {e}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    return report(summary, log)


if __name__ == "__main__":
    sys.exit(main())
