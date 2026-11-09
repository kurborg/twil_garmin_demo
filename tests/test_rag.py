import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from twil_demo import ollama, rag_app
from twil_demo.ask_garmin import computed_facts, question_range


class IndexTests(unittest.TestCase):
    def test_index_reuse_retrieval_and_staleness(self):
        with tempfile.TemporaryDirectory() as folder:
            docs = Path(folder) / "docs"
            docs.mkdir()
            (docs / "week.md").write_text("Swim: two hours in the week.")
            index = Path(folder) / "index.npz"
            with patch("twil_demo.ollama.embed", return_value=np.array([[1., 0.]], dtype=np.float32)) as embed:
                self.assertEqual(rag_app.build_index(docs, index), 1)
                self.assertEqual(rag_app.build_index(docs, index), 1)
                self.assertEqual(embed.call_count, 1)
                hits = rag_app.retrieve("swim", docs=docs, index=index)
                self.assertEqual(hits[0][0], "week.md")
                self.assertEqual(hits[0][2], 1.)
                (docs / "week.md").write_text("Changed data")
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    rag_app.retrieve("swim", docs=docs, index=index)

    def test_missing_index_is_not_built_implicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("twil_demo.ollama.embed") as embed:
                with self.assertRaisesRegex(RuntimeError, "No RAG index"):
                    rag_app.retrieve("question", index=Path(folder) / "missing.npz")
                embed.assert_not_called()

    def test_chunk_parameters_and_empty_documents(self):
        with self.assertRaises(ValueError):
            rag_app.chunk("abc", 3, 3)
        self.assertEqual(rag_app.chunk("  "), [])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "No .md or .txt"):
                rag_app.build_index(Path(folder), Path(folder) / "index.npz")

    def test_answer_sends_selected_excerpts_to_generation(self):
        selected = [("garmin/week.md", "Selected evidence: 2 swim hours.", 0.9)]
        with patch.object(rag_app, "retrieve", return_value=selected), patch.object(ollama, "generate", return_value="Two hours [1].") as generate:
            result = rag_app.answer("How was swimming?")
        prompt = generate.call_args.args[0]
        self.assertIn("[1] (garmin/week.md)", prompt)
        self.assertIn("Selected evidence: 2 swim hours.", prompt)
        self.assertIn("QUESTION: How was swimming?", prompt)
        self.assertNotIn("0.9", prompt)
        self.assertIn("[1] garmin/week.md", result)


class OllamaTests(unittest.TestCase):
    def test_generate_posts_local_model_payload(self):
        response = Mock()
        response.json.return_value = {"response": "Hello"}
        with patch.object(ollama.requests, "post", return_value=response) as post:
            self.assertEqual(ollama.generate("A prompt"), "Hello")
        args, kwargs = post.call_args
        self.assertEqual(args, (f"{ollama.URL}/api/generate",))
        self.assertEqual(kwargs["json"]["model"], ollama.CHAT_MODEL)
        self.assertEqual(kwargs["json"]["prompt"], "A prompt")
        self.assertFalse(kwargs["json"]["stream"])
        self.assertEqual(kwargs["timeout"], (10, 600))

    def test_embedding_posts_prefixed_text_and_normalizes(self):
        response = Mock()
        response.json.return_value = {"embeddings": [[3., 4.]]}
        with patch.object(ollama, "EMBED_MODEL", "nomic-embed-text"), patch.object(ollama.requests, "post", return_value=response) as post:
            vector = ollama.embed(["question"], query=True)
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["search_query: question"])
        np.testing.assert_allclose(vector, [[0.6, 0.8]])

    def test_server_invalid_json_is_actionable(self):
        response = Mock()
        response.json.side_effect = ValueError("bad JSON")
        with patch.object(ollama.requests, "post", return_value=response), self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            ollama.generate("test")

    def test_tls_failure_is_actionable(self):
        with patch.object(ollama.requests, "post", side_effect=ollama.requests.exceptions.SSLError("certificate")), self.assertRaisesRegex(RuntimeError, "TLS validation"):
            ollama.generate("test")


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.acts = pd.DataFrame({
            "date": pd.to_datetime(["2026-10-11 12:00", "2026-10-12 12:00", "2026-11-07 23:00", "2026-11-08 12:00"]),
            "sport": ["run", "run", "bike", "run"],
            "duration_min": [900., 60., 120., 30.],
            "distance_km": [100., 10., 50., 5.],
        })
        self.daily = pd.DataFrame({
            "date": pd.date_range("2026-11-01", "2026-11-08"),
            "sleep_hours": [6., 8., None, None, None, None, None, None],
            "resting_hr": [40.] * 8, "weight_lb": [200.] * 8,
        })

    def facts(self, question):
        return json.loads(computed_facts(question, self.acts, self.daily))

    def test_28_day_window_and_sport_and_units(self):
        result = self.facts("How many hours did I run in the last 4 weeks?")
        self.assertEqual(result["date_start"], "2026-10-12")
        self.assertEqual(result["duration"]["value"], 1.5)
        self.assertEqual(result["sports"], ["run"])

    def test_explicit_date_end_includes_whole_day(self):
        result = self.facts("Total bike distance 2026-11-01 to 2026-11-07")
        self.assertEqual(result["distance"]["value"], 50.)

    def test_wellness_reports_observation_coverage(self):
        result = self.facts("Average sleep hours last 8 days")
        self.assertEqual(result["sleep_hours"]["mean"], 7.)
        self.assertEqual(result["sleep_hours"]["observed_days"], 2)
        self.assertEqual(result["sleep_hours"]["calendar_days_in_window"], 8)

    def test_counts_include_transition_records(self):
        self.acts.loc[len(self.acts)] = [pd.Timestamp("2026-11-08 13:00"), "transition", 3., 0.1]
        result = self.facts("How many sessions last 7 days?")
        self.assertEqual(result["session_count"], 3)
        self.assertIn("including transitions", result["count_policy"])

    def test_unsupported_arithmetic_is_rejected(self):
        for question in ["How many days did I rest?", "What was my load ratio?", "Average sleep last month", "Total hours since 2026-01-01", "How many calories last 7 days", "Total hours and distance last 7 days", "Total hours in September", "Average hours per week", "Total hours first 4 weeks"]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                computed_facts(question, self.acts, self.daily)

    def test_invalid_date_window(self):
        with self.assertRaises(ValueError):
            question_range("last 0 weeks", pd.Timestamp("2026-11-08"))

    def test_outside_export_dates_are_not_reported_as_zero(self):
        for question in ["Total hours 2025-01-01 to 2025-02-01", "Total hours last 365 days", "Total distance 2026-11-09 to 2026-11-10"]:
            with self.subTest(question=question), self.assertRaisesRegex(ValueError, "outside the export coverage"):
                self.facts(question)

    def test_unknown_duration_is_not_an_exact_partial_total(self):
        self.acts.loc[3, "duration_min"] = np.nan
        result = self.facts("Total run hours last 4 weeks")
        self.assertIsNone(result["duration"]["value"])
        self.assertEqual(result["duration"]["sessions_with_measurements"], 1)
        self.assertEqual(result["duration"]["sessions_in_window"], 2)


if __name__ == "__main__":
    unittest.main()
