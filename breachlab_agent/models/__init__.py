"""The one interface the agent loop needs from a model.

A model receives a system prompt and a plain conversation and returns text
plus zero or more calls of the single tool. This starter ships no model: you
write the call in `my_model.py`, name it in `[model] factory`, or pass your
own object straight to `Agent`. The message format is in `my_model.py`.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

TOOL_NAME = "run_command"
TOOL_DESCRIPTION = (
    "Run one shell command in the SSH session of the level you are playing and "
    "return its output and exit code. This is the only tool. It runs on the "
    "level's box, never on the operator's machine. The shell keeps its working "
    "directory between calls. Commands run without a terminal and with stdin "
    "closed, so interactive programs do not work: use non interactive flags."
)
TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The shell command to run on the level's box."}
    },
    "required": ["command"],
    "additionalProperties": False,
}


@dataclass
class ToolCall:
    id: str
    command: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None: the backend cannot price this turn


@dataclass
class ModelReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    truncated: bool = False  # the reply hit the output limit
    served_by: str | None = None  # set when another model than the one asked for answered


class ModelError(Exception):
    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class ModelRefusal(ModelError):
    """The model declined the request. Not retried."""

    def __init__(self, message: str) -> None:
        super().__init__(message, fatal=True)


class Model(Protocol):
    name: str
    # provider/model for the run record, or None when the backend cannot know.
    declared_model: str | None

    def complete(self, system: str, messages: list[dict[str, Any]]) -> ModelReply: ...


_MODEL_KEY = re.compile(r"^[a-z0-9]+/[a-z0-9][a-z0-9._:-]*$")


def valid_model_key(key: str) -> bool:
    """The shape BreachLab takes for a declared model: provider/model, lower
    case, at most 40 characters. Which providers it knows is up to the API."""
    return len(key) <= 40 and bool(_MODEL_KEY.match(key))


def price_for(usage_in: int, usage_out: int, price_in: float, price_out: float) -> float | None:
    if price_in <= 0 and price_out <= 0:
        return None
    return (usage_in * price_in + usage_out * price_out) / 1_000_000


_FACTORY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


def build_model(cfg: Any, env: Mapping[str, str] | None = None) -> Model:
    """Construct the model named by [model] factory.

    The factory is `module:callable`. It is called with the [model] table and
    the environment and returns an object with the `Model` interface.
    """
    env = os.environ if env is None else env
    factory = (cfg.factory or "").strip()
    if not factory:
        raise ModelError(
            "no model is wired in. Write the model call in breachlab_agent/models/my_model.py "
            'and set [model] factory = "breachlab_agent.models.my_model:build", see the README.',
            fatal=True,
        )
    if not _FACTORY.match(factory):
        raise ModelError("[model] factory must look like package.module:callable", fatal=True)
    module_name, attr = factory.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ModelError(f"cannot import the model module {module_name}: {e}", fatal=True) from None
    build = getattr(module, attr, None)
    if not callable(build):
        raise ModelError(f"{module_name} has no callable named {attr}", fatal=True)
    model = build(cfg, env)
    if not callable(getattr(model, "complete", None)) or not isinstance(getattr(model, "name", None), str):
        raise ModelError(f"{factory} did not return a model: it needs a name and a complete() method", fatal=True)
    return model
