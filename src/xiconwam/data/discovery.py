"""Deterministic LeRobot v3 root discovery, without importing training libraries.

Run ``python -m xiconwam.data.discovery ROOT ...`` to inspect a tree before training.
This validates metadata identity, not parquet/video contents or action semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def discover_dataset_dirs(roots) -> list[str]:
    """Expand leaves or arbitrary-depth containers; deduplicate physical roots.

    Input order is preserved, children are sorted, symlink cycles are pruned.
    Each explicit input must contain a dataset. Bad metadata fails loudly.
    Once a dataset is found its data/meta/videos subtree is never traversed.
    """
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    roots = list(roots)
    if not roots:
        raise ValueError("At least one dataset directory is required")
    result, datasets_seen = [], set()

    def fail(error):
        raise error

    for raw in roots:
        root = Path(raw).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        visited, found = set(), False
        for current, children, _ in os.walk(root, followlinks=True, onerror=fail):
            path = Path(current)
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in visited:
                children[:] = []
                continue
            visited.add(identity)
            marker = path / "meta" / "info.json"
            if marker.is_file():
                try:
                    info = json.loads(marker.read_text(encoding="utf-8"))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Invalid dataset metadata: {marker}") from exc
                if not isinstance(info, dict) or info.get("codebase_version") != "v3.0":
                    raise ValueError(f"Expected LeRobot v3.0 metadata at {marker}")
                found = True
                children[:] = []
                if identity not in datasets_seen:
                    datasets_seen.add(identity)
                    result.append(str(path.resolve()))
            else:
                children[:] = sorted(
                    name for name in children if not name.startswith(".")
                )
        if not found:
            raise FileNotFoundError(f"No LeRobot meta/info.json found under {root}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    datasets = []
    for directory in discover_dataset_dirs(args.roots):
        info = json.loads((Path(directory) / "meta/info.json").read_text())
        datasets.append(
            {
                "path": directory,
                **{
                    key: info.get(key)
                    for key in (
                        "codebase_version",
                        "robot_type",
                        "fps",
                        "total_episodes",
                        "total_frames",
                        "features",
                    )
                },
            }
        )
    print(json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
