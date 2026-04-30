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

# Match a time token: either "12.34" / "12" / "12s"  OR  "mm:ss(.ss)".
_TIME_TOKEN = re.compile(
    r"""
    (?:(\d+):)?               # optional leading minutes + colon
    (\d+(?:\.\d+)?)           # seconds (with optional decimal)
    \s*s?                     # optional trailing 's'
    """,
    re.VERBOSE | re.IGNORECASE,
)
_KEY_VAL = re.compile(r"visual_time\s*=\s*(\S[^;]*)", re.IGNORECASE)


def _parse_time(s: str) -> float:
    """Parse a single token like '8.04', '8.04 s', '0:25', '1:23.5'."""
    s = s.strip()
    m = _TIME_TOKEN.fullmatch(s)
    if not m:
        return math.nan
    minutes = int(m.group(1)) if m.group(1) else 0
    seconds = float(m.group(2))
    return minutes * 60.0 + seconds


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
        return _parse_time(m.group(1))
    return _parse_time(raw.split(";", 1)[0])
