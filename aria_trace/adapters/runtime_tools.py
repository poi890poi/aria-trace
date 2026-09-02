"""Locate optional runtimes in an extracted IRIS release tree."""

import sys
from pathlib import Path
from typing import Iterable, Optional, Union


def find_release_tool(
    relative_path: Union[str, Path],
    anchors: Optional[Iterable[Union[str, Path]]] = None,
) -> Optional[Path]:
    """Return a tool from the nearest manifest-bearing release root.

    Packaged applications live below ``apps/<name>`` while importable source
    lives below ``python/``. Walking upward from both locations keeps direct
    executable use and third-party imports independent of the process CWD.
    """
    if anchors is None:
        anchors = (
            Path(sys.executable).resolve().parent,
            Path(__file__).resolve().parent,
            Path.cwd().resolve(),
        )
    seen = set()
    for anchor_value in anchors:
        anchor = Path(anchor_value).resolve()
        for candidate_root in (anchor,) + tuple(anchor.parents):
            key = str(candidate_root).casefold()
            if key in seen:
                continue
            seen.add(key)
            if not (candidate_root / "release-manifest.yaml").is_file():
                continue
            candidate = candidate_root / relative_path
            if candidate.is_file():
                return candidate
    return None
