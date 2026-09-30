"""
Records exactly what produced a training run, so any published prediction can
be traced back to (and rebuilt from) its code, dependencies, and data.
"""

import hashlib
import os
import platform
import subprocess
from importlib.metadata import version
from typing import Any, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> Optional[str]:
    if "GITHUB_SHA" in os.environ:
        return os.environ["GITHUB_SHA"]

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def collect_provenance(data_dir: str) -> dict[str, Any]:
    lock_path = os.path.join(REPO_ROOT, "uv.lock")

    return {
        "git_sha": _git_sha(),
        "uv_lock_sha256": sha256_file(lock_path) if os.path.isfile(lock_path) else None,
        "versions": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ["scikit-learn", "pandas", "numpy"]},
        },
        "data_sha256": {
            filename: sha256_file(os.path.join(data_dir, filename))
            for filename in sorted(os.listdir(data_dir))
            if filename.endswith(".csv.gz")
        },
    }
