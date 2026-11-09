"""Regression checks for Garmin units, identity, sleep records, and missing days."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from twil_demo.ingest import build_tables, ingest


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / "raw"
        self.raw.mkdir()

    def write(self, name, data):
        (self.raw / name).write_text(json.dumps(data), encoding="utf-8")

    def activity(self, activity_id=1, **values):
        return {
            "activityId": activity_id, "startTimeLocal": 1735732800000,
            "startTimeGmt": 1735750800000, "activityType": "running",
            "duration": 3600000, "distance": 1000000,
            "activityTrainingLoad": 75, **values,
        }

    def write_activities(self, *activities):
        self.write("account_1_summarizedActivities.json", [{"summarizedActivitiesExport": list(activities)}])

    def test_units_identity_and_multisport_parent(self):
        child = self.activity(2, parentId=1)
        self.write_activities(self.activity(1, parent=True, activityType="multi_sport"), child, child)
        acts, daily, report = build_tables(self.raw)
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts.iloc[0]["activity_id"], "2")
        self.assertEqual(acts.iloc[0]["duration_min"], 60)
        self.assertEqual(acts.iloc[0]["distance_km"], 10)
        self.assertEqual(acts.iloc[0]["sport"], "run")
        self.assertEqual(acts.iloc[0]["training_load"], 75)
        self.assertEqual(acts.iloc[0]["date"], pd.Timestamp("2025-01-01 12:00"))
        self.assertEqual(report["duplicate_activity_ids_removed"], 1)
        self.assertEqual(report["multisport_parent_totals_excluded"], 1)
        self.assertTrue(pd.isna(daily.iloc[0]["sleep_hours"]))

    def test_main_sleep_naps_duplicates_and_complete_calendar(self):
        self.write_activities(self.activity())
        self.write("account_sleepData.json", [
            {"calendarDate": "2025-01-01", "sleepWindowConfirmationType": "UNCONFIRMED",
             "deepSleepSeconds": 900, "lightSleepSeconds": 900, "remSleepSeconds": 0,
             "sleepEndTimestampGMT": "2025-01-01T15:00:00", "sleepScores": {"overallScore": 10}},
            {"calendarDate": "2025-01-01", "sleepWindowConfirmationType": "ENHANCED_CONFIRMED_FINAL",
             "deepSleepSeconds": 7200, "lightSleepSeconds": 14400, "remSleepSeconds": 3600,
             "sleepEndTimestampGMT": "2025-01-01T12:00:00", "sleepScores": {"overallScore": 80}},
            {"calendarDate": "2025-01-03", "deepSleepSeconds": 3600},
            {"napList": [{"calendarDate": "2025-01-01", "napTimeSec": 1800,
                          "napStartTimestampGMT": "2025-01-01T17:00:00"}]},
        ])
        _, daily, report = build_tables(self.raw)
        self.assertEqual(len(daily), 3)
        self.assertEqual(daily.iloc[0]["sleep_hours"], 7)
        self.assertEqual(daily.iloc[0]["sleep_score"], 80)
        self.assertEqual(daily.iloc[0]["nap_hours"], 0.5)
        self.assertTrue(pd.isna(daily.iloc[1]["sleep_hours"]))
        self.assertTrue(pd.isna(daily.iloc[2]["sleep_hours"]))
        self.assertEqual(report["duplicate_main_sleep_dates_removed"], 1)
        self.assertEqual(report["nap_only_wrapper_records"], 1)

    def test_daily_sources_and_missing_values(self):
        self.write_activities(self.activity(avgHr=-1))
        self.write("UDSFile_test.json", [{"calendarDate": "2025-01-01", "restingHeartRate": 45,
            "allDayStress": {"aggregatorList": [{"type": "AWAKE", "averageStressLevel": 40},
                                                   {"type": "TOTAL", "averageStressLevel": 25}]}}])
        self.write("account_userBioMetrics.json", [{"metaData": {"calendarDate": "2025-01-01T10:00:00"},
                                                    "weight": {"weight": 100000}}])
        self.write("MetricsMaxMetData_test.json", [{"calendarDate": "2025-01-01", "sport": "RUNNING", "vo2MaxValue": 50},
                                                  {"calendarDate": "2025-01-01", "sport": "CYCLING", "vo2MaxValue": 48}])
        acts, daily, _ = build_tables(self.raw)
        self.assertTrue(pd.isna(acts.iloc[0]["avg_hr"]))
        self.assertEqual(daily.iloc[0]["resting_hr"], 45)
        self.assertEqual(daily.iloc[0]["stress_avg"], 25)
        self.assertAlmostEqual(daily.iloc[0]["weight_lb"], 220.462262, places=5)
        self.assertEqual(daily.iloc[0]["vo2max"], 50)
        self.assertEqual(daily.iloc[0]["vo2max_cycling"], 48)
        clean = Path(self.temp.name) / "clean"
        ingest(self.raw, clean)
        self.assertEqual(len(pd.read_parquet(clean / "activities.parquet")), 1)
        self.assertTrue((clean / "ingestion_report.json").exists())

    def test_missing_export_fails_with_help(self):
        with self.assertRaisesRegex(ValueError, "No activities found"):
            build_tables(self.raw)


if __name__ == "__main__":
    unittest.main()
