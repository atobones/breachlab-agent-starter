from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

from breachlab_agent.config import ConfigError, _mini_toml, load_config, parse_toml, read_token
from breachlab_agent.sanitize import clean_output, frame_untrusted, one_line, redact, strip_control, truncate_middle
from breachlab_agent.state import FlagStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Sanitize(unittest.TestCase):
    def test_escape_sequences(self) -> None:
        cases = {
            "\x1b[1;31mbold\x1b[0m": "bold",
            "\x1b]0;title\x07after": "after",
            "\x1b]52;c;aGVsbG8=\x1b\\after": "after",  # clipboard write
            "\x1bPq#0;2;0;0;0\x1b\\after": "after",
            "a\x1b[2J\x1b[Hb": "ab",
            "\x1b[?1049h\x1b[?25lx": "x",
            "tab\tkept\nnewline": "tab\tkept\nnewline",
            "a\r\nb": "a\nb",
            "\x1b": "",
            "unterminated \x1b]0;title": "unterminated ",
        }
        for raw, want in cases.items():
            self.assertEqual(strip_control(raw), want, repr(raw))

    def test_no_control_character_survives_any_byte(self) -> None:
        raw = "".join(chr(i) for i in range(0, 0x100)) + "‮⁦‍﻿"
        out = strip_control(raw)
        for ch in out:
            self.assertTrue(ch in "\t\n" or (ord(ch) >= 0x20 and not 0x7F <= ord(ch) <= 0x9F), repr(ch))
        self.assertNotIn("‮", out)

    def test_truncate(self) -> None:
        self.assertEqual(truncate_middle("short", 100), ("short", False))
        out, cut = truncate_middle("H" * 500 + "T" * 500, 90)
        self.assertTrue(cut)
        self.assertTrue(out.startswith("H" * 60) and out.endswith("T" * 30))
        self.assertIn("910 characters omitted", out)

    def test_strip_happens_before_the_cut(self) -> None:
        out, _ = clean_output("x" * 50 + "\x1b[31m" * 100 + "y" * 50, 80)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("[31m", out)

    def test_frame_nonce_is_fresh(self) -> None:
        a, b = frame_untrusted("SSH_OUTPUT", "x"), frame_untrusted("SSH_OUTPUT", "x")
        self.assertNotEqual(a, b)

    def test_one_line_and_redact(self) -> None:
        self.assertEqual(one_line("a\n\x1b[31mb\x07  c"), "a b c")
        self.assertEqual(redact("key sk-12345 and blp_abc", ["blp_abc", "sk-12345", "", "ab"]), "key [redacted] and [redacted]")


class ConfigTests(unittest.TestCase):
    def test_shipped_config_loads_and_matches_the_defaults(self) -> None:
        cfg = load_config(os.path.join(ROOT, "config.toml"), env={})
        self.assertEqual(cfg, load_config(None, env={}))
        self.assertEqual(cfg.agent.track, "ghost")
        self.assertEqual(cfg.api.base_url, "https://breachlab.org")

    def test_fallback_parser_agrees_with_the_standard_one(self) -> None:
        text = pathlib.Path(ROOT, "config.toml").read_text(encoding="utf-8")
        self.assertEqual(_mini_toml(text), parse_toml(text))

    def test_env_overrides_and_validation(self) -> None:
        cfg = load_config(None, env={"BREACHLAB_TRACK": "ghost-2", "BREACHLAB_API_URL": "http://127.0.0.1:1/api/v2/"})
        # The client takes the site. An address that still names the API prefix keeps working.
        self.assertEqual((cfg.agent.track, cfg.api.base_url), ("ghost-2", "http://127.0.0.1:1"))
        self.assertEqual(load_config(None, env={"BREACHLAB_API_URL": "http://127.0.0.1:1/"}).api.base_url, "http://127.0.0.1:1")
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write("[budgets]\nmax_stepz = 3\n")
        self.addCleanup(os.unlink, f.name)
        with self.assertRaises(ConfigError):
            load_config(f.name, env={})
        self.assertEqual(load_config(None, env={"BREACHLAB_MODEL_FACTORY": "pkg.mod:build"}).model.factory, "pkg.mod:build")

    def test_token_comes_from_the_environment_only(self) -> None:
        self.assertEqual(read_token({"BREACHLAB_TOKEN": " blp_abc "}), "blp_abc")
        for env in ({}, {"BREACHLAB_TOKEN": ""}, {"BREACHLAB_TOKEN": "sk-not-a-breachlab-token"}):
            with self.assertRaises(ConfigError):
                read_token(env)
        text = pathlib.Path(ROOT, "config.toml").read_text(encoding="utf-8").lower()
        for word in ("token =", "api_key", "password"):
            self.assertNotIn(word, text)


class State(unittest.TestCase):
    def test_flags_persist_private_and_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "state")
            FlagStore(sub, "host/agent-a").put("ghost", 0, "flag_a")
            self.assertEqual(FlagStore(sub, "host/agent-a").get("ghost", 0), "flag_a")
            self.assertIsNone(FlagStore(sub, "host/agent-b").get("ghost", 0))
            self.assertIsNone(FlagStore(sub, "host/agent-a").get("ghost", 1))
            self.assertEqual(os.stat(os.path.join(sub, "flags.json")).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(sub).st_mode & 0o777, 0o700)
