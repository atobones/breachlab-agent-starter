"""Shared test wiring: an agent pointed at the stub API and simulated boxes."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from breachlab import BreachLab

from breachlab_agent import USER_AGENT
from breachlab_agent.agent import Agent
from breachlab_agent.config import Config
from breachlab_agent.ssh import TargetGuard
from breachlab_agent.state import FlagStore
from breachlab_agent.stub import ScriptedModel, StubServer, StubState, fake_session_factory


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class StubCase(unittest.TestCase):
    def setUp(self) -> None:
        self.state = StubState()
        self.server = StubServer(self.state)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = FakeClock()
        self.logs: list[str] = []
        self.cfg = Config()
        self.guard = TargetGuard()
        self.sessions: list = []
        self.flags = FlagStore(self.tmp.name, "test")
        # The SDK client waits with breachlab.client.sleep. The fake clock
        # takes its place, so a rate limit wait costs no real time.
        patcher = mock.patch("breachlab.client.sleep", self.clock.sleep)
        patcher.start()
        self.addCleanup(patcher.stop)

    def client(self, **kw) -> BreachLab:
        kw.setdefault("user_agent", USER_AGENT)
        kw.setdefault("notify", self.logs.append)
        kw.setdefault("auto_token", False)
        return BreachLab(self.state.token, self.server.base_url, **kw)

    def agent(self, replies, **client_kw) -> tuple[Agent, ScriptedModel]:
        model = replies if hasattr(replies, "complete") else ScriptedModel(replies)
        agent = Agent(
            self.cfg, self.client(**client_kw), model,
            fake_session_factory(self.guard, self.state, self.sessions),
            self.guard, self.flags, log=self.logs.append, clock=self.clock, sleep=self.clock.sleep,
        )
        return agent, model

    def paths(self, method: str | None = None) -> list[str]:
        return [r["path"] for r in self.state.requests if method in (None, r["method"])]

    def only_run(self) -> dict:
        self.assertEqual(len(self.state.runs), 1)
        return next(iter(self.state.runs.values()))

    def step_text(self) -> str:
        return "\n".join(f"{s['type']}: {s['content']}" for s in self.only_run()["steps"])
