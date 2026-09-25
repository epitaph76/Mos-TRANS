"""Point-in-time and shared-cleaning contract tests."""

import csv
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from mos_trans.preprocessing.core import (
    TrafficIndex,
    Telemetry,
    _features,
    build_dataset,
    clean_traffic_row,
)


NOW = datetime.fromisoformat("2026-01-06 12:00:00")


def packet(packet_id, tr_id, event, receive, speed, lon=37.6, lat=55.7, gps_valid=True):
    return Telemetry(
        packet_id=packet_id, tr_id=tr_id, unit_id="u",
        event_time=datetime.fromisoformat(event), receive_time=datetime.fromisoformat(receive),
        location_valid=gps_valid, gps_valid=gps_valid,
        lon=lon if gps_valid else None, lat=lat if gps_valid else None,
        speed=speed, speed_outlier=False, heading=None, is_hist_data=False,
    )


def feature_for(packets):
    sample = {
        "sample_id": "one", "tr_id": "bus", "T": NOW,
        "target_stop_id": "stop", "target_time_begin": datetime.fromisoformat("2026-01-06 12:12:00"),
        "cur_dev_s": 10.0, "target_delay_s": 30.0, "target_delta_s": 20.0,
    }
    stop = {"lon": 37.61, "lat": 55.7, "address": "A"}
    return _features(sample, stop, [datetime.fromisoformat("2026-01-06 12:12:00")], TrafficIndex(packets))


class CleaningTests(unittest.TestCase):
    def test_bad_speed_and_gps_are_independent(self):
        base = {
            "packet_id": "p", "tr_id": "bus", "unit_id": "u",
            "event_time": "2026-01-06 11:59:00", "receive_time": "2026-01-06 11:59:01",
            "location_valid": "False", "lon": "37.6", "lat": "55.7",
            "speed": "20", "heading": "90", "is_hist_data": "False",
        }
        clean = clean_traffic_row(base)
        self.assertEqual(clean.speed, 20)
        self.assertIsNone(clean.lon)
        self.assertFalse(clean.gps_valid)
        base["speed"] = "368"
        outlier = clean_traffic_row(base)
        self.assertIsNone(outlier.speed)
        self.assertTrue(outlier.speed_outlier)

    def test_future_and_late_packets_do_not_change_features(self):
        current = packet("1", "bus", "2026-01-06 11:59:30", "2026-01-06 11:59:31", 10)
        baseline = feature_for([current])
        added = feature_for([
            current,
            packet("2", "bus", "2026-01-06 12:00:01", "2026-01-06 12:00:01", 100),
            packet("3", "bus", "2026-01-06 11:59:59", "2026-01-06 12:00:01", 95),
            packet("4", "other", "2026-01-06 11:59:59", "2026-01-06 12:00:01", 0),
        ])
        self.assertEqual(baseline, added)

    def test_stale_gps_and_context(self):
        own = packet("1", "bus", "2026-01-06 11:59:30", "2026-01-06 11:59:31", 1)
        near = packet("2", "near", "2026-01-06 11:59:20", "2026-01-06 11:59:21", 0)
        stale = packet("3", "stale", "2026-01-06 11:58:59", "2026-01-06 11:59:00", 40)
        features = feature_for([own, near, stale])
        self.assertEqual(features["context_vehicle_count_500m"], 1)
        self.assertEqual(features["context_speed_mean_500m"], 0)
        self.assertEqual(features["realtime_stopped_fraction_2m"], 1)
        old = feature_for([packet("4", "bus", "2026-01-06 11:57:59", "2026-01-06 11:58:00", 10)])
        self.assertIsNone(old["realtime_distance_to_target_m"])
        self.assertEqual(old["context_vehicle_count_500m"], 0)
        empty = feature_for([])
        self.assertIsNone(empty["realtime_data_age_s"])
        self.assertEqual(empty["realtime_packet_count_2m"], 0)


class PipelineTests(unittest.TestCase):
    def _write_csv(self, root, member, columns, rows):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_split_schedule_zip_and_directory_have_identical_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            traffic_columns = ["packet_id", "tr_id", "unit_id", "event_time", "receive_time", "location_valid", "lon", "lat", "speed", "heading", "is_hist_data"]
            traffic = [{
                "packet_id": "p", "tr_id": "bus", "unit_id": "u", "event_time": "2026-01-06 11:59:30",
                "receive_time": "2026-01-06 11:59:31", "location_valid": "True", "lon": "37.6",
                "lat": "55.7", "speed": "10", "heading": "90", "is_hist_data": "False",
            }]
            for split in ("train", "test", "validate"):
                self._write_csv(source, f"{split}/traffic.csv", traffic_columns, traffic)
            schedule_columns = ["tt_action_item_id", "tr_id", "time_begin", "geom", "building_address", "time_fact_begin"]
            points_columns = ["sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s", "target_delay_s"]
            for split, minute in (("train", 12), ("test", 13), ("validate", 14)):
                planned = f"2026-01-06 12:{minute:02d}:00"
                schedule = [{"tt_action_item_id": "same-stop-id", "tr_id": "bus", "time_begin": planned,
                             "geom": "POINT (37.61 55.7)", "building_address": "A", "time_fact_begin": "2026-01-06 23:59:00"}]
                self._write_csv(source, f"{split}/{'schedule_plan' if split == 'validate' else 'schedule'}.csv", schedule_columns, schedule)
                point = {"sample_id": split, "tr_id": "bus", "T": "2026-01-06 12:00:00",
                         "target_stop_id": "same-stop-id", "target_time_begin": planned,
                         "cur_dev_s": "10", "target_delay_s": "30"}
                member = "validate/points.csv" if split == "validate" else f"labels/labels_{split}.csv"
                cols = points_columns[:-1] if split == "validate" else points_columns
                self._write_csv(source, member, cols, [{key: point[key] for key in cols}])
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as output_zip:
                for path in sorted(source.rglob("*.csv")):
                    output_zip.write(path, path.relative_to(source).as_posix())
            prefixed_archive = root / "dataset-prefixed.zip"
            with zipfile.ZipFile(prefixed_archive, "w") as output_zip:
                for path in sorted(source.rglob("*.csv")):
                    member = Path("dataset") / path.relative_to(source)
                    output_zip.write(path, member.as_posix())
            from_dir = root / "from_dir"
            from_zip = root / "from_zip"
            from_prefixed_zip = root / "from_prefixed_zip"
            self.assertEqual(build_dataset(source, from_dir), {"traffic": 1, "train": 1, "test": 1, "validate": 1})
            self.assertEqual(build_dataset(archive, from_zip), {"traffic": 1, "train": 1, "test": 1, "validate": 1})
            self.assertEqual(build_dataset(prefixed_archive, from_prefixed_zip), {"traffic": 1, "train": 1, "test": 1, "validate": 1})
            for name in ("traffic_clean", "train_samples", "test_samples", "validate_samples", "train_features", "test_features", "validate_features"):
                self.assertTrue(pq.read_table(from_dir / f"{name}.parquet").equals(pq.read_table(from_zip / f"{name}.parquet")))
                self.assertTrue(pq.read_table(from_dir / f"{name}.parquet").equals(pq.read_table(from_prefixed_zip / f"{name}.parquet")))
            train = pq.read_table(from_dir / "train_features.parquet").to_pylist()[0]
            test = pq.read_table(from_dir / "test_features.parquet").to_pylist()[0]
            validate = pq.read_table(from_dir / "validate_features.parquet").to_pylist()[0]
            self.assertEqual((train["schedule_horizon_s"], test["schedule_horizon_s"], validate["schedule_horizon_s"]), (720, 780, 840))
            self.assertEqual(train["target_delta_s"], 20)
            self.assertIsNone(validate["target_delta_s"])
            self.assertFalse(any("fact" in col for col in pq.read_schema(from_dir / "train_features.parquet").names))


if __name__ == "__main__":
    unittest.main()
