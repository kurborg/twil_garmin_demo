import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from twil_demo.build_report import build_report, report_cards
from twil_demo.charts import rolling_daily
from twil_demo import build_report as report_module, charts


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.activities = pd.DataFrame({
            "date": pd.to_datetime(["2026-10-01 08:00", "2026-11-01 08:00", "2026-11-07 08:00"]),
            "sport": ["run", "run", "bike"],
            "duration_min": [600.0, 60.0, 120.0],
            "distance_km": [100.0, 10.0, 40.0],
            "training_load": [100.0, 10.0, np.nan],
        })
        self.daily = pd.DataFrame({"date": pd.date_range("2026-10-01", "2026-11-08")})
        for column in ["resting_hr", "sleep_hours", "weight_lb", "stress_avg", "vo2max"]:
            self.daily[column] = np.nan
        self.daily.loc[self.daily["date"].eq("2026-11-08"), "sleep_hours"] = 8.0

    def test_cards_use_calendar_windows_and_preserve_missing_values(self):
        cards = report_cards(self.activities, self.daily)
        self.assertEqual(cards["Training hours, last 28 days"], "3.0 h")
        self.assertEqual(cards["Sleep, last 7 days"], "8.0 h")
        self.assertEqual(cards["Weight, last 7 days"], "Unavailable")
        self.assertEqual(cards["Load ratio, latest day"], "Unavailable")

    def test_rolling_average_does_not_bridge_sparse_measurements(self):
        sparse = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]), "resting_hr": [40.0, 50.0, 60.0, 70.0]})
        self.assertTrue(rolling_daily(sparse, "resting_hr").isna().all())

    def test_missing_duration_does_not_understate_report_total(self):
        self.activities.loc[1, "duration_min"] = np.nan
        self.assertEqual(report_cards(self.activities, self.daily)["Training hours, last 28 days"], "Unavailable")

    def test_empty_activity_window_reports_zero_hours(self):
        self.activities["date"] = self.activities["date"] - pd.Timedelta(days=90)
        self.assertEqual(report_cards(self.activities, self.daily)["Training hours, last 28 days"], "0.0 h")

    def test_help_does_not_render_or_write(self):
        for module, function in [(charts, "build_charts"), (report_module, "build_report")]:
            with self.subTest(module=module.__name__):
                with patch("sys.argv", [module.__name__, "--help"]), patch.object(module, function) as work:
                    with self.assertRaises(SystemExit) as exited:
                        module.main()
                    self.assertEqual(exited.exception.code, 0)
                    work.assert_not_called()

    def test_html_escapes_observations_and_uses_local_links(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            docs = base / "docs" / "garmin"
            docs.mkdir(parents=True)
            (docs / "patterns.md").write_text('<script>alert("x")</script>', encoding="utf-8")
            (docs / "week_2026-11-02.md").write_text("# Week", encoding="utf-8")
            charts = base / "charts"
            charts.mkdir()
            (charts / "weekly_volume.png").write_bytes(b"image placeholder")
            path = build_report(self.activities, self.daily, base / "dashboard" / "report.html", charts, base / "docs")
            html = path.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("<script>", html)
            self.assertIn('src="../charts/weekly_volume.png"', html)
            self.assertIn('href="../docs/garmin/week_2026-11-02.md"', html)
            self.assertIn("1 / 7", html)
            self.assertNotIn('src="https://', html)


if __name__ == "__main__":
    unittest.main()
