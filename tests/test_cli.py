from __future__ import annotations

import contextlib
import io
import os
import pathlib
import re
import unittest
from unittest import mock

from breachlab_agent.__main__ import EXIT_CONFIG, EXIT_OK, main

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    err = io.StringIO()
    with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, err.getvalue()


# Spelled in pieces so this file does not trip its own check.
VENDOR_WORDS = [
    "anth" + "ropic", "open" + "ai", "clau" + "de", "gp" + "t-", "gem" + "ini", "goo" + "gle", "mist" + "ral",
    "deep" + "seek", "lla" + "ma", "olla" + "ma", "qw" + "en", "x\\.ai", "ali" + "baba", "hai" + "ku", "son" + "net", "op" + "us",
]


class Cli(unittest.TestCase):
    def test_dry_run_needs_no_token_no_model_and_passes(self) -> None:
        code, out = run(["--dry-run", "--config", str(ROOT / "config.toml")], {"PATH": os.environ.get("PATH", "")})
        self.assertEqual(code, EXIT_OK, out)
        for call in ("GET /api/v2/me", "GET /api/v2/tracks", "POST /api/v2/runs", "GET /api/v2/token", "GET /api/v2/me/progress",
                     "GET /api/v2/tracks/ghost/levels/0",
                     "GET /api/v2/tracks/ghost/levels/0/credentials", "POST /api/v2/submit", "/steps", "/finish"):
            self.assertIn(call, out)
        self.assertIn("levels solved in this run: [0, 1]", out)
        self.assertIn("refuses a host the API did not return: yes", out)
        self.assertIn("dry run passed", out)

    def test_missing_token_and_missing_model_are_explained(self) -> None:
        code, out = run(["--config", "/nonexistent.toml"], {})
        self.assertEqual(code, EXIT_CONFIG)
        self.assertIn("BREACHLAB_TOKEN is not set", out)
        code, out = run(["--config", "/nonexistent.toml"], {"BREACHLAB_TOKEN": "blp_abc"})
        self.assertEqual(code, EXIT_CONFIG)
        self.assertIn("no model is wired in", out)
        self.assertIn("my_model.py", out)
        # The shipped stub is a place to write the call, not a model.
        env = {"BREACHLAB_TOKEN": "blp_abc", "BREACHLAB_MODEL_FACTORY": "breachlab_agent.models.my_model:build"}
        from breachlab_agent.models import ModelError, build_model
        from breachlab_agent.config import load_config

        model = build_model(load_config(None, env=env).model, env)
        with self.assertRaises(ModelError) as caught:
            model.complete("system", [])
        self.assertTrue(caught.exception.fatal)
        self.assertIn("not written yet", str(caught.exception))

    def test_no_model_id_is_a_default_anywhere(self) -> None:
        from breachlab_agent.config import Config

        self.assertEqual(Config().model.name, "")
        self.assertNotRegex((ROOT / "config.toml").read_text(encoding="utf-8"), r"(?m)^name\s*=\s*\"[^\"]")
        self.assertEqual(Config().model.factory, "")
        self.assertNotRegex((ROOT / "config.toml").read_text(encoding="utf-8"), r"(?m)^factory\s*=\s*\"[^\"]")
        for path in ROOT.glob("breachlab_agent/**/*.py"):
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"DEFAULT_MODEL", str(path))

    def test_no_vendor_or_model_is_named_anywhere(self) -> None:
        # The starter is neutral: the operator chooses and wires the model.
        names = "|".join(VENDOR_WORDS)
        files = [ROOT / "README.md", ROOT / "config.toml", ROOT / "pyproject.toml"]
        files += [p for p in ROOT.glob("breachlab_agent/**/*.py")]
        for path in files:
            found = re.findall(rf"(?i)\b({names})", path.read_text(encoding="utf-8"))
            self.assertEqual(found, [], str(path))

    def test_the_only_dependency_is_the_sdk(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dependencies = ["breachlab>=0.3.0"]', text)
        self.assertNotIn("optional-dependencies", text)
        for path in ROOT.glob("breachlab_agent/**/*.py"):
            body = path.read_text(encoding="utf-8")
            if path.name != "stub.py":
                self.assertNotRegex(body, r"(?m)^\s*(import|from) urllib\.request", str(path))

    def test_secrets_are_scrubbed_from_log_lines(self) -> None:
        from breachlab_agent.__main__ import make_logger

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            make_logger(["blp_live_token", "sk-model-key"])("sent blp_live_token and sk-model-key\x1b[31m")
        self.assertEqual(err.getvalue(), "sent [redacted] and [redacted]\n")


class Repository(unittest.TestCase):
    def test_readme_copy_rules(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        prose = re.sub(r"`[^`\n]*`", "", prose)
        prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)
        self.assertEqual(re.findall(r"[‒-―]", prose), [])
        self.assertEqual(re.findall(r"\s-{1,2}\s", re.sub(r"(?m)^\s*[-*] ", "", prose)), [])
        self.assertEqual(re.findall(r"[()\[\]]", prose), [])
        self.assertEqual(re.findall(r"[\U0001F000-\U0001FFFF☀-➿]", prose), [])
        self.assertTrue(text.isascii())

    def test_no_secret_shaped_strings_are_committed(self) -> None:
        for path in list(ROOT.glob("*.*")) + list(ROOT.glob("breachlab_agent/**/*.py")):
            if path.is_file() and path.name != "stub.py":
                body = path.read_text(encoding="utf-8", errors="replace")
                self.assertIsNone(re.search(r"blp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9-]{20,}", body), path)
