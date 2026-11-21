"""Your model goes here. This file is the only place you have to write code.

The agent loop calls `complete()` once per turn with the system prompt and the
conversation so far. Send them to the model you chose, and translate its
answer into a `ModelReply`:

  * text the model wrote goes into `text`;
  * each request to run a command becomes a `ToolCall(id, command)`. The tool
    is described by TOOL_NAME, TOOL_DESCRIPTION and TOOL_SCHEMA;
  * the token counts of the turn go into `usage`, so the budgets work.

Conversation entries are plain dicts:
    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": Any}
    {"role": "tool", "tool_call_id": str, "text": str, "is_error": bool}
Whatever you put into `ModelReply.raw` comes back as "raw" on the assistant
entry of the next turn, for APIs that want their own message object replayed.

Raise `ModelError(message)` for a failure worth another try,
`ModelError(message, fatal=True)` to stop the run, and `ModelRefusal(message)`
when the model declines the task.

Keys and endpoints belong in environment variables. Never write a secret into
this file or into config.toml.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA, ModelError, ModelReply, ToolCall, Usage, price_for

__all__ = ["MyModel", "build"]

# Referenced so the names you need are already imported.
_ = (TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA, ToolCall, Usage, price_for)


class MyModel:
    def __init__(self, cfg: Any, env: Mapping[str, str]) -> None:
        # The label shown in the log. [model] name in config.toml.
        self.name = cfg.name or "my-model"
        # provider/model for the run record, or None to use the model declared
        # on the agent's account in the BreachLab dashboard.
        self.declared_model: str | None = None
        self._cfg = cfg
        self._env = env

    def complete(self, system: str, messages: list[dict[str, Any]]) -> ModelReply:
        raise ModelError(
            "the model call is not written yet: fill in MyModel.complete in "
            "breachlab_agent/models/my_model.py",
            fatal=True,
        )


def build(cfg: Any, env: Mapping[str, str]) -> MyModel:
    """The factory named in [model] factory."""
    return MyModel(cfg, env)
