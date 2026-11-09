import unittest

import numpy as np
import pandas as pd

from twil_demo.metrics import compute_summary, daily_load, filter_activities, weekly_volume


class MetricsTests(unittest.TestCase):
    def activities(self):
        return pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=40, freq="D") + pd.Timedelta(hours=12),
            "sport": ["run"] * 40, "duration_min": [60.] * 40,
            "distance_km": [10.] * 40, "training_load": [100.] * 40,
        })

    def daily(self):
        return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40),
                             **{c: [np.nan] * 40 for c in ["resting_hr", "sleep_hours", "sleep_score", "stress_avg", "weight_lb", "vo2max"]}})

    def test_exact_calendar_range_includes_end_day(self):
        selected = filter_activities(self.activities(), "2026-01-01", "2026-01-28")
        self.assertEqual(len(selected), 28)
        summary = compute_summary(self.activities(), self.daily())
        self.assertEqual(summary["sessions"], 28)
        self.assertEqual(summary["hours"], 28)
        self.assertIsNone(summary["recovery_last_7_days"]["sleep_hours"]["mean"])

    def test_load_ratio_uses_means_and_warmup(self):
        load = daily_load(self.activities())
        self.assertTrue(load["acwr"].iloc[:27].isna().all())
        self.assertEqual(load["acwr"].iloc[27], 1.)

    def test_unknown_session_load_invalidates_day_and_window(self):
        acts = self.activities()
        acts.loc[20, "training_load"] = np.nan
        load = daily_load(acts)
        self.assertTrue(pd.isna(load["training_load"].iloc[20]))
        self.assertTrue(load["acwr"].iloc[27:].isna().all())

    def test_weekly_volume_includes_empty_weeks(self):
        acts = self.activities().iloc[[0, 35]]
        weeks = weekly_volume(acts)
        self.assertGreater(len(weeks), 2)
        self.assertEqual(weeks["run"].iloc[1], 0)

    def test_empty_load_is_supported(self):
        self.assertTrue(daily_load(self.activities().iloc[:0]).empty)

    def test_unknown_duration_is_not_a_zero_or_partial_total(self):
        acts = self.activities()
        acts.loc[39, "duration_min"] = np.nan
        self.assertTrue(pd.isna(weekly_volume(acts)["run"].iloc[-1]))
        self.assertIsNone(compute_summary(acts, self.daily())["hours"])

    def test_transition_only_day_has_zero_load(self):
        acts = self.activities()
        acts.loc[20, "sport"] = "transition"
        acts.loc[20, "training_load"] = np.nan
        self.assertEqual(daily_load(acts)["training_load"].iloc[20], 0.)


if __name__ == "__main__":
    unittest.main()
