"""Configuration: a small config.toml plus environment variables.

Secrets (the BreachLab token and your model's keys) come from the environment only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

DEFAULT_API_URL = "https://breachlab.org"


class ConfigError(Exception):
    pass


@dataclass
class AgentConfig:
    track: str = "ghost"
    # Stop after this many levels in one invocation. 0 means no limit.
    max_levels: int = 0
    # provider/model, recorded on the run. Empty: derived, or the model you
    # declared on the agent in the dashboard.
    declared_model: str = ""
    # Replay solved levels when a flag needed as a password is not in the
    # local state. The API never returns flags.
    recover_missing_flags: bool = True
    state_dir: str = ".breachlab-agent"


@dataclass
class Budgets:
    max_steps: int = 150
    max_steps_per_level: int = 40
    max_minutes: float = 30.0
    max_cost_usd: float = 2.0
    max_total_tokens: int = 2_000_000
    max_consecutive_failures: int = 5
    max_flag_attempts_per_level: int = 3


@dataclass
class ModelConfig:
    # module:callable that returns your model. Empty: no model, the agent
    # explains what to write. There is no default.
    factory: str = ""
    # A label for the log. Your model code may use it as the model id.
    name: str = ""
    max_output_tokens: int = 0  # 0: your model's own default
    request_timeout_seconds: float = 300.0
    # USD per million tokens, for the cost budget. 0: cost is not tracked.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0


@dataclass
class SshConfig:
    binary: str = "ssh"
    command_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 20.0
    # Wait before the one retry of a session that did not open.
    reconnect_delay_seconds: float = 3.0
    max_output_chars: int = 6000
    max_command_chars: int = 2000
    # Development only: lets the agent connect to loopback or private
    # addresses if the API returns one.
    allow_private_hosts: bool = False


@dataclass
class ApiConfig:
    base_url: str = DEFAULT_API_URL
    # Longest single wait the agent accepts from a Retry-After before it
    # stops instead.
    max_retry_wait_seconds: float = 330.0
    request_timeout_seconds: float = 30.0


@dataclass
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    budgets: Budgets = field(default_factory=Budgets)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssh: SshConfig = field(default_factory=SshConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


# ── TOML ────────────────────────────────────────────────────────────────────


def _mini_toml(text: str) -> dict[str, dict[str, Any]]:
    """The subset of TOML that config.toml uses, for Python 3.10.

    Tables, and keys whose values are strings, integers, floats or booleans.
    Python 3.11 and later use the standard library parser instead.
    """
    out: dict[str, dict[str, Any]] = {}
    table: dict[str, Any] | None = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            table = out.setdefault(line[1:-1].strip(), {})
            continue
        if "=" not in line or table is None:
            raise ConfigError(f"config line {n}: expected key = value inside a [table]")
        key, value = (part.strip() for part in line.split("=", 1))
        if value[:1] in "\"'":
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0:
                raise ConfigError(f"config line {n}: unterminated string")
            table[key] = value[1:end]
            continue
        value = value.split("#", 1)[0].strip()
        if value in ("true", "false"):
            table[key] = value == "true"
            continue
        try:
            table[key] = int(value.replace("_", ""))
        except ValueError:
            try:
                table[key] = float(value.replace("_", ""))
            except ValueError:
                raise ConfigError(f"config line {n}: unsupported value {value!r}") from None
    return out


def parse_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        return _mini_toml(text)
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"config file is not valid TOML: {e}") from None


def _apply(section: Any, values: Mapping[str, Any], name: str) -> None:
    known = {f.name: f for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"unknown setting [{name}] {key}")
        current = getattr(section, key)
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"[{name}] {key} must be true or false")
        elif isinstance(current, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"[{name}] {key} must be a number")
            value = type(current)(value)
        elif not isinstance(value, str):
            raise ConfigError(f"[{name}] {key} must be a string")
        setattr(section, key, value)


def load_config(path: str | os.PathLike[str] | None, env: Mapping[str, str] | None = None) -> Config:
    """Read config.toml if it exists, then apply environment overrides."""
    env = os.environ if env is None else env
    cfg = Config()
    if path is not None:
        p = Path(path)
        if p.exists():
            data = parse_toml(p.read_text(encoding="utf-8"))
            for name, values in data.items():
                if not hasattr(cfg, name) or not isinstance(values, dict):
                    raise ConfigError(f"unknown table [{name}] in {p}")
                _apply(getattr(cfg, name), values, name)

    if env.get("BREACHLAB_API_URL"):
        cfg.api.base_url = env["BREACHLAB_API_URL"]
    if env.get("BREACHLAB_TRACK"):
        cfg.agent.track = env["BREACHLAB_TRACK"]
    if env.get("BREACHLAB_MODEL_FACTORY"):
        cfg.model.factory = env["BREACHLAB_MODEL_FACTORY"]
    if env.get("BREACHLAB_DECLARED_MODEL"):
        cfg.agent.declared_model = env["BREACHLAB_DECLARED_MODEL"]

    # The client takes the site, not the API prefix. An older config that
    # names the prefix keeps working.
    cfg.api.base_url = cfg.api.base_url.rstrip("/")
    if cfg.api.base_url.endswith("/api/v2"):
        cfg.api.base_url = cfg.api.base_url[: -len("/api/v2")]
    b = cfg.budgets
    if min(b.max_steps, b.max_steps_per_level, b.max_consecutive_failures, b.max_flag_attempts_per_level) < 1:
        raise ConfigError("[budgets] step, failure and flag attempt limits must be at least 1")
    if b.max_minutes <= 0:
        raise ConfigError("[budgets] max_minutes must be above 0")
    return cfg


def read_token(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    token = env.get("BREACHLAB_TOKEN", "").strip()
    if not token:
        raise ConfigError(
            "BREACHLAB_TOKEN is not set. Mint an agent token in the dashboard "
            "and export it, see the README quickstart."
        )
    if not token.startswith("blp_"):
        raise ConfigError("BREACHLAB_TOKEN does not look like a BreachLab token, it should start with blp_")
    return token
