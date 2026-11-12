"""Project-root resolution for data, config and cache paths.

The package is installed into a virtualenv, so paths must not be derived from
``__file__``. Resolution order: ``$NIFTY_HOME``, else the nearest ancestor of the
current working directory that contains ``pyproject.toml``, else the cwd.
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    env = os.environ.get("NIFTY_HOME")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return cwd


PROJECT_ROOT = _find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
META_ROOT = DATA_ROOT / "_meta"
