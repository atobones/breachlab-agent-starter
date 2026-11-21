"""Everything a level's box prints is untrusted input.

Before any of it reaches the model, the operator's terminal or a run step, it
goes through this module: control sequences are stripped, the text is cut to a
budget, and it is wrapped in delimiters that carry a random nonce the box
cannot know.
"""

from __future__ import annotations

import re
import secrets

# ANSI escape sequences: CSI, OSC (terminated by BEL or ST), DCS/SOS/PM/APC
# strings, and two-byte escapes.
_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"
    r"|\x1b[@-Z\\-_]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
)
# C0 controls except tab and newline, DEL, C1 controls, and the Unicode
# characters that reorder or hide text (bidi overrides, isolates, zero width).
_CONTROL = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁠-⁩﻿]"
)


def strip_control(text: str) -> str:
    """Remove escape sequences and control characters. Keeps tab and newline."""
    text = text.replace("\r\n", "\n")
    text = _ANSI.sub("", text)
    # A bare carriage return rewrites the line in a terminal. Show both parts.
    text = text.replace("\r", "\n")
    return _CONTROL.sub("", text)


def truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    """Keep the head and the tail of a long text. Returns the text and a flag."""
    if limit <= 0 or len(text) <= limit:
        return text, False
    head = (limit * 2) // 3
    tail = limit - head
    omitted = len(text) - limit
    return (
        f"{text[:head]}\n[... {omitted} characters omitted by the agent ...]\n{text[-tail:]}",
        True,
    )


def clean_output(raw: str, limit: int) -> tuple[str, bool]:
    """Strip, then truncate. The order matters: escapes must not survive a cut."""
    return truncate_middle(strip_control(raw), limit)


def one_line(text: str, limit: int = 200) -> str:
    """A single safe line for the operator's terminal."""
    flat = " ".join(strip_control(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def frame_untrusted(label: str, body: str, attrs: str = "") -> str:
    """Wrap untrusted text in delimiters that carry a fresh random nonce.

    The system prompt tells the model that text inside these delimiters is
    data. The nonce is new for every block, so text printed by the box cannot
    close the block early and continue as if it were the agent speaking.
    """
    nonce = secrets.token_hex(8)
    extra = f" {attrs}" if attrs else ""
    return (
        f"<<<{label} {nonce}{extra}>>>\n"
        f"{body}\n"
        f"<<<END {label} {nonce}>>>"
    )


def redact(text: str, secrets_: list[str]) -> str:
    """Replace known secret values. Longest first so overlaps are covered."""
    for s in sorted({s for s in secrets_ if s and len(s) >= 4}, key=len, reverse=True):
        text = text.replace(s, "[redacted]")
    return text
