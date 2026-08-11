from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


def uma_entry_point() -> Path:
    """Absolute path to the installed `uma` console script.

    `sysconfig` reports the script directory of the *running* interpreter, which
    is correct for virtualenvs and base installs on every platform. Deriving it
    from `sys.executable` instead only holds where the interpreter and the
    console scripts share a directory — true on POSIX and inside Windows venvs,
    false for a base Windows install, where `python.exe` sits beside `Scripts/`
    rather than in it. Windows console scripts also carry an `.exe` suffix.
    """
    scripts = Path(sysconfig.get_path("scripts"))
    return scripts / ("uma.exe" if sys.platform == "win32" else "uma")
