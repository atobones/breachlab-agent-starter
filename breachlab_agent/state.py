"""Local state: the flags this agent captured.

On tracks like Ghost I the password of a level is the flag of the level
before, and the API never returns a flag. So to continue a track in a later
invocation the agent has to remember its own flags. They live in one JSON
file readable by the operator only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class FlagStore:
    def __init__(self, state_dir: str | os.PathLike[str] | None, scope: str) -> None:
        """`scope` separates agents and API origins, for example host/username."""
        self._path = Path(state_dir) / "flags.json" if state_dir is not None else None
        self._scope = scope
        self._data: dict[str, dict[str, str]] = {}
        if self._path is not None and self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {k: v for k, v in loaded.items() if isinstance(v, dict)}
            except (ValueError, OSError):
                self._data = {}

    def _key(self, track: str) -> str:
        return f"{self._scope}|{track}"

    def get(self, track: str, idx: int) -> str | None:
        return self._data.get(self._key(track), {}).get(str(idx))

    def put(self, track: str, idx: int, flag: str) -> None:
        self._data.setdefault(self._key(track), {})[str(idx)] = flag
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=1, sort_keys=True)
        os.replace(tmp, self._path)

    def all_values(self) -> list[str]:
        return [v for track in self._data.values() for v in track.values()]
