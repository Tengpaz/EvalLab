#!/usr/bin/env python
"""Validate a configured benchmark dataset."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from eval_bench.utils.config import load_yaml
from eval_bench.utils.logging import log
from eval_bench.utils.registry import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--no-image-read", action="store_true", help="Only check path existence/camera/split structure.")
    args = parser.parse_args()

    dataset = build_dataset(load_yaml(args.dataset_config))
    issues = dataset.validate(check_images=not args.no_image_read)
    for issue in issues:
        extra = f" frame={issue['frame_id']}" if "frame_id" in issue else ""
        log(f"{issue['level'].upper()}: scene={issue['scene_id']}{extra}: {issue['message']}")
        if issue.get("fix"):
            log(f"  fix: {issue['fix']}")
    errors = [issue for issue in issues if issue["level"] == "error"]
    if errors:
        log(f"Validation failed with {len(errors)} error(s).")
        raise SystemExit(1)
    log(f"Validation passed for dataset '{dataset.name}' ({len(issues)} warning(s)).")


if __name__ == "__main__":
    main()
