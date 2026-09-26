"""Build the required semicolon CSV from the same predictor used by the API."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import pandas as pd

from mos_trans.inference import Predictor, prediction_frame
from mos_trans.preprocessing.core import DataSource


def build(features_path: Path, dataset: Path, output: Path) -> pd.DataFrame:
    predictor = Predictor()
    prediction = predictor.predict(prediction_frame(features_path, predictor))
    source = DataSource(dataset)
    try:
        if source.archive is not None:
            with source.archive.open("sample_submission.csv") as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig") as stream:
                    template = pd.DataFrame(list(csv.DictReader(stream, delimiter=";")))
        else:
            with (source.path / "sample_submission.csv").open(encoding="utf-8-sig") as stream:
                template = pd.DataFrame(list(csv.DictReader(stream, delimiter=";")))
    finally:
        source.close()
    if template.columns.tolist() != ["sample_id", "prediction"]:
        raise ValueError("Unexpected sample_submission columns")
    if template.sample_id.duplicated().any() or set(template.sample_id) != set(prediction.sample_id):
        raise ValueError("Submission sample IDs do not match validate")
    result = template[["sample_id"]].merge(prediction[["sample_id", "predicted_delay_s"]],
                                           on="sample_id", validate="one_to_one")
    result = result.rename(columns={"predicted_delay_s": "prediction"})
    if len(result) != len(template) or result.prediction.isna().any():
        raise ValueError("Incomplete submission")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep=";", index=False, encoding="utf-8", float_format="%.6f",
                  lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed_route/validate_features.parquet"))
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/integrated_submission/submission.csv"))
    args = parser.parse_args()
    print(f"Saved {len(build(args.features, args.dataset, args.output))} predictions to {args.output}")
