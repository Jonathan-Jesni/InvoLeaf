"""Repository-relative path handling.

Split files are committed and must therefore be machine-independent. Storing an
absolute path such as ``D:/Projects/BTP SEM 7/InvoLeaf/data/...`` would break the moment
the repository is opened on the October ROCm machine, or on any collaborator's checkout.
Roots living inside the repository are stored relative to it and re-resolved on load;
roots outside it (a dataset on another drive) stay absolute, since nothing better is
available.

Local data copies
-----------------
On the cloud instances the repository lives on a network disk whose reads stall for
minutes at a time under load; one ViT-B epoch that normally takes 39 s took 22 min. The
training queue therefore copies the images to the machine's local disk at the start of a
session and sets ``INVOLEAF_DATA_DIR`` to that copy. Roots under ``data/`` are then
resolved there instead. If the variable is unset, or the copy lacks that folder, the
repository path is used as before, so nothing breaks when no copy was made.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_ENV = "INVOLEAF_DATA_DIR"


def to_portable(path: str | Path) -> str:
    """Store as a repo-relative POSIX path when possible."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_root(stored: str | Path) -> Path:
    """Invert :func:`to_portable`, resolving relative roots against the repository.

    A root under ``data/`` is redirected to the local copy named by ``INVOLEAF_DATA_DIR``
    when that copy contains it.
    """
    p = Path(stored)
    if p.is_absolute():
        return p
    local = os.environ.get(DATA_DIR_ENV)
    if local and p.parts and p.parts[0] == "data":
        candidate = Path(local).joinpath(*p.parts[1:])
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / p
