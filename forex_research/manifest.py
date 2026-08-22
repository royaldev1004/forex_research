from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ResearchConfig
from .logging_utils import get_logger

log = get_logger("manifest")

TRACKED_PACKAGES = ("pandas", "numpy", "pyarrow", "yaml", "pytest")


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "not installed"
    return out


def _git_commit(root: Path) -> dict[str, Any]:
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout.strip()
            return {"commit": commit, "dirty": bool(dirty)}
        return {"commit": None, "reason": "not a git repository"}
    except Exception as exc:  # pragma: no cover
        return {"commit": None, "reason": f"git unavailable: {exc}"}


def build_manifest(
    cfg: ResearchConfig,
    sources: list[dict[str, Any]],
    date_range: dict[str, Any],
    counts: dict[str, Any],
    warnings: list[str],
    blockers: list[str],
    outputs: dict[str, Any],
    timings: dict[str, float],
    project_root: Path,
) -> dict[str, Any]:
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": __import__("forex_research").__version__,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "git": _git_commit(project_root),
        "config": cfg.to_dict(),
        "sources": sources,
        "date_range": date_range,
        "counts": counts,
        "outputs": outputs,
        "timings_seconds": timings,
        "warnings": warnings,
        "blockers": blockers,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info("Wrote run manifest: %s", path)
