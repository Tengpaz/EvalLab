#!/usr/bin/env python
"""Print scene/split contents for a configured dataset."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from eval_bench.utils.config import load_yaml
from eval_bench.utils.registry import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--max-scenes", type=int, default=20)
    args = parser.parse_args()

    dataset = build_dataset(load_yaml(args.dataset_config))
    for idx, sample in enumerate(dataset.iter_samples()):
        if idx >= args.max_scenes:
            print(f"... truncated after {args.max_scenes} scenes")
            break
        print(
            f"{sample.dataset_name}/{sample.split_name}/{sample.scene_id}: "
            f"num_inputs={len(sample.input_ids)} inputs={sample.input_ids} "
            f"targets={sample.target_ids} frames={len(sample.frames)}"
        )


if __name__ == "__main__":
    main()
