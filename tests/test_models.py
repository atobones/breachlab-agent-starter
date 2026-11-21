"""The model layer: a factory you name, a stub to fill in, a script for wiring checks."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from breachlab_agent.config import ModelConfig
from breachlab_agent.models import ModelError, ModelReply, build_model, price_for, valid_model_key
from breachlab_agent.models import scripted


class FakeModel:
    name = "fake"
    declared_model = None

    def complete(self, system, messages):
        return ModelReply(text="GIVE_UP")


def build_fake(cfg, env):
    return FakeModel()


def build_broken(cfg, env):
    return object()


class Factory(unittest.TestCase):
    def cfg(self, factory: str) -> ModelConfig:
        return ModelConfig(factory=factory)

    def test_no_factory_means_no_model_and_says_what_to_write(self) -> None:
        with self.assertRaises(ModelError) as caught:
            build_model(self.cfg(""), {})
        self.assertTrue(caught.exception.fatal)
        self.assertIn("my_model.py", str(caught.exception))

    def test_a_named_factory_is_called_with_the_table_and_the_environment(self) -> None:
        model = build_model(self.cfg(f"{__name__}:build_fake"), {"X": "1"})
        self.assertIsInstance(model, FakeModel)

    def test_bad_factories_are_configuration_errors(self) -> None:
        for factory in ("os.system", "no_such_module_xyz:build", f"{__name__}:missing", f"{__name__}:build_broken", "a b:c", ":x"):
            with self.assertRaises(ModelError, msg=factory) as caught:
                build_model(self.cfg(factory), {})
            self.assertTrue(caught.exception.fatal, factory)

    def test_declared_model_shape_and_price(self) -> None:
        self.assertTrue(valid_model_key("other/my-model.v2"))
        for bad in ("my-model", "Other/x", "other/" + "x" * 40, "other/ space", "/x", "other/"):
            self.assertFalse(valid_model_key(bad), bad)
        self.assertIsNone(price_for(1000, 1000, 0, 0))
        self.assertAlmostEqual(price_for(1_000_000, 500_000, 2.0, 4.0), 4.0)


class Script(unittest.TestCase):
    def write(self, turns) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(turns, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_runs_the_commands_then_reports_a_line_of_the_last_output(self) -> None:
        path = self.write([{"command": "ls"}, {"command": "cat readme"}, {"flag_from_output": 1}])
        m = scripted.build(ModelConfig(), {scripted.ENV_SCRIPT: path})
        first = m.complete("s", [])
        self.assertEqual([c.command for c in first.tool_calls], ["ls"])
        self.assertEqual(m.complete("s", []).tool_calls[0].command, "cat readme")
        tool = {"role": "tool", "tool_call_id": "x", "text": "<<<SSH_OUTPUT n\n\n  the_value  \nmore\n<<<END n", "is_error": False}
        self.assertEqual(m.complete("s", [tool]).text, "FLAG: the_value")
        self.assertEqual(m.complete("s", []).text, "GIVE_UP")

    def test_a_missing_line_gives_up_instead_of_guessing(self) -> None:
        m = scripted.build(ModelConfig(), {scripted.ENV_SCRIPT: self.write([{"flag_from_output": 3}, {"say": "GIVE_UP"}])})
        self.assertEqual(m.complete("s", []).text, "GIVE_UP")

    def test_bad_scripts_are_refused(self) -> None:
        with self.assertRaises(ModelError):
            scripted.build(ModelConfig(), {})
        for turns in ({"command": "ls"}, [{"shell": "ls"}], [{"command": "a", "say": "b"}], ["ls"]):
            with self.assertRaises(ModelError, msg=str(turns)):
                scripted.build(ModelConfig(), {scripted.ENV_SCRIPT: self.write(turns)})
        with self.assertRaises(ModelError):
            scripted.build(ModelConfig(), {scripted.ENV_SCRIPT: "/nonexistent/script.json"})
