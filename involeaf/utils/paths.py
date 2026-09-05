"""Repository-relative path handling.

Split files are committed and must therefore be machine-independent. Storing an
absolute path such as ``D:/Projects/BTP SEM 7/InvoLeaf/data/...`` would break the moment
the repository is opened on the October ROCm machine, or on any collaborator's checkout.
Roots living inside the repository are stored relative to it and re-resolved on load;
roots outside it (a dataset on another drive) stay absolute, since nothing better is
available.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def to_portable(path: str | Path) -> str:
    """Store as a repo-relative POSIX path when possible."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_root(stored: str | Path) -> Path:
    """Invert :func:`to_portable`, resolving relative roots against the repository."""
    p = Path(stored)
    return p if p.is_absolute() else (REPO_ROOT / p)
