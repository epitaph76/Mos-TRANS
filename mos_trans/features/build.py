from pathlib import Path
import argparse

import pandas as pd

from mos_trans.features.route_sections import (add_route_signatures, attach_planned_sections, build_route_segments, load_schedule)
from mos_trans.features.delay_context import add_delay_trend_features

SPLITS = (
    "train",
    "test",
    "validate",
)


def build_feature_set(
    input_dir: Path,
    output_dir: Path,
    dataset_path: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:
        print(f"\nBuilding {split}...")

        input_path = (
            input_dir
            / f"{split}_features.parquet"
        )

        frame = pd.read_parquet(
            input_path
        )

        initial_rows = len(frame)
        initial_ids = set(
            frame["sample_id"]
        )

        schedule = load_schedule(
            dataset_path,
            split,
        )

        schedule = add_route_signatures(
            schedule
        )

        segments = build_route_segments(
            schedule
        )

        enriched = attach_planned_sections(
            frame,
            segments,
        )
        enriched = add_delay_trend_features(enriched)

        if len(enriched) != initial_rows:
            raise ValueError(
                f"{split}: изменилось количество "
                f"строк: {initial_rows} → "
                f"{len(enriched)}"
            )

        # Проверяем, что sample_id не изменились.
        enriched_ids = set(
            enriched["sample_id"]
        )

        if enriched_ids != initial_ids:
            raise ValueError(
                f"{split}: изменился набор "
                "sample_id"
            )

        if not enriched[
            "sample_id"
        ].is_unique:
            raise ValueError(
                f"{split}: появились дубли "
                "sample_id"
            )

        output_path = (
            output_dir
            / f"{split}_features.parquet"
        )

        enriched.to_parquet(
            output_path,
            index=False,
        )

        route_coverage = (
            enriched["route_signature"]
            .notna()
            .mean()
        )

        section_coverage = (
            enriched[
                "planned_section_available"
            ].mean()
        )
        previous_delay_coverage = (
            enriched[
                "previous_delay_available"
            ].mean()
        )

        lag_2_coverage = (
            enriched[
                "lag_2_available"
            ].mean()
        )

        print("rows:", len(enriched))
        print(
            "route coverage:",
            round(route_coverage, 4),
        )
        print(
            "planned section coverage:",
            round(section_coverage, 4),
        )
        print(
            "previous delay coverage:",
            round(previous_delay_coverage, 4),
        )

        print(
            "lag 2 coverage:",
            round(lag_2_coverage, 4),
        )
        print(
            "unique routes:",
            enriched[
                "route_signature"
            ].nunique(),
        )
        print(
            "unique sections:",
            enriched[
                "section_id"
            ].nunique(),
        )
        print("saved:", output_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed_route"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/dataset.zip"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_feature_set(
        input_dir=args.input,
        output_dir=args.output,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()