"""Optional UI regression test using only synthetic tables."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Optional dashboard dependencies are not installed")
class DashboardTests(unittest.TestCase):
    def test_sport_filters_and_empty_selection(self):
        from streamlit.testing.v1 import AppTest

        dates = pd.date_range("2026-01-01", periods=40)
        daily = pd.DataFrame({"date": dates, **{name: [np.nan] * 40 for name in ["resting_hr", "sleep_hours", "sleep_score", "stress_avg", "weight_lb", "vo2max"]}})
        acts = pd.DataFrame({"date": [dates[20], dates[30]], "sport": ["run", "bike"],
                             "duration_min": [60., 120.], "distance_km": [10., 40.], "training_load": [100., 100.]})
        with tempfile.TemporaryDirectory() as directory, patch("twil_demo.metrics.load_tables", return_value=(acts, daily)), patch("twil_demo.paths.DOCS", Path(directory)):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/twil_demo/dashboard.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "3.0 h")
            app.multiselect[0].set_value(["run"]).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "1.0 h")
            app.multiselect[0].set_value([]).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "0.0 h")
            self.assertEqual(app.metric[1].value, "0")


if __name__ == "__main__":
    unittest.main()
