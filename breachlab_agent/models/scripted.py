"""A fixed script in place of a model. For checking the wiring, not for playing.

It runs the commands of a script file one per turn and then answers. Use it
to check the loop against a real level before spending anything on a model.

    [model]
    factory = "breachlab_agent.models.scripted:build"

    export BREACHLAB_AGENT_SCRIPT=script.json

The script is a JSON list. Each entry is one turn:
    {"command": "ls -la"}        run this command
    {"say": "GIVE_UP"}           answer with this text
    {"flag_from_output": N}      answer FLAG: with line N (from 1) of the last
                                 command output that is not empty
When the script runs out, the answer is GIVE_UP.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from . import ModelError, ModelReply, ToolCall

ENV_SCRIPT = "BREACHLAB_AGENT_SCRIPT"


class ScriptFileModel:
    name = "scripted"
    declared_model: str | None = None

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self._turns = list(turns)
        self._n = 0

    def complete(self, system: str, messages: list[dict[str, Any]]) -> ModelReply:
        if not self._turns:
            return ModelReply(text="GIVE_UP")
        turn = self._turns.pop(0)
        if "command" in turn:
            self._n += 1
            return ModelReply(tool_calls=[ToolCall(f"scripted_{self._n}", str(turn["command"]))])
        if "flag_from_output" in turn:
            outputs = [m["text"] for m in messages if m["role"] == "tool"]
            # The first and last line of a tool message are the frame around
            # the output of the box.
            lines = [ln.strip() for ln in (outputs[-1].splitlines()[1:-1] if outputs else []) if ln.strip()]
            n = int(turn["flag_from_output"])
            if not 1 <= n <= len(lines):
                return ModelReply(text="GIVE_UP")
            return ModelReply(text=f"FLAG: {lines[n - 1]}")
        return ModelReply(text=str(turn.get("say", "GIVE_UP")))


def build(cfg: Any, env: Mapping[str, str]) -> ScriptFileModel:
    path = env.get(ENV_SCRIPT, "").strip()
    if not path:
        raise ModelError(f"{ENV_SCRIPT} is not set: point it at a script file", fatal=True)
    try:
        with open(path, encoding="utf-8") as fh:
            turns = json.load(fh)
    except (OSError, ValueError) as e:
        raise ModelError(f"cannot read the script {path}: {e}", fatal=True) from None
    ok = isinstance(turns, list) and all(
        isinstance(t, dict) and len(t) == 1 and next(iter(t)) in ("command", "say", "flag_from_output") for t in turns
    )
    if not ok:
        raise ModelError("the script must be a JSON list of {command}, {say} or {flag_from_output} entries", fatal=True)
    return ScriptFileModel(turns)
