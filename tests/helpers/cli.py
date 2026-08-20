from __future__ import annotations

import shutil
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

    That directory is not the only possibility: when the base installation is
    not writable, pip falls back to a `--user` install and the console script
    lands in the *user* scheme's script directory instead. Check that too, then
    fall back to PATH, so the test asserts "is the entry point installed?"
    rather than "is it installed in one specific directory?".

    Returns the interpreter-scheme path when nothing is found, so a failing
    assertion still names the location that was primarily expected.
    """
    name = "uma.exe" if sys.platform == "win32" else "uma"

    primary = Path(sysconfig.get_path("scripts")) / name
    candidates = [primary]

    try:
        user_scheme = sysconfig.get_preferred_scheme("user")
        candidates.append(Path(sysconfig.get_path("scripts", scheme=user_scheme)) / name)
    except (KeyError, ValueError):
        # No user scheme on this platform/interpreter; the other candidates stand.
        pass

    resolved = shutil.which("uma")
    if resolved:
        candidates.append(Path(resolved))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return primary
