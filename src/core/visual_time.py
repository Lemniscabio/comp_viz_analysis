"""Read user-supplied 'visual mixing time' from macOS Finder Comments.

Workflow: in Finder, select a video, ⌘I, type the visual mixing time
into the Comments field as either `8.04` or `visual_time=8.04`. Optional
trailing free-text notes after a `;` are ignored.

Returns NaN when no comment is set, the file is on a non-Apple filesystem,
or the comment cannot be parsed as a float.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Union

_KEY_VAL = re.compile(r"visual_time\s*=\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
_BARE_FLOAT = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*s?\s*$", re.IGNORECASE)


def read_visual_time(path: Union[str, Path]) -> float:
    """Return the visual mixing time in seconds, or NaN if unavailable.

    Reads the macOS kMDItemFinderComment xattr via `mdls`. Accepts either
    `visual_time=8.04` or just `8.04` in the comment.
    """
    try:
        out = subprocess.run(
            ["mdls", "-raw", "-name", "kMDItemFinderComment", str(path)],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return math.nan
    if out.returncode != 0:
        return math.nan
    raw = (out.stdout or "").strip()
    if not raw or raw == "(null)":
        return math.nan

    m = _KEY_VAL.search(raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return math.nan
    m = _BARE_FLOAT.match(raw.split(";", 1)[0])
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return math.nan
    return math.nan
