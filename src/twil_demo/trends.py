"""Calculate descriptive trends and write grounded RAG documents."""

import argparse
import json

import pandas as pd

from twil_demo.metrics import compute_summary, daily_load, filter_activities, load_tables, weekly_volume
from twil_demo.paths import DOCS, OUTPUT, ensure_directories


def detect_patterns(activities, daily, summary):
    end = pd.Timestamp(summary["end_date"])
    patterns = []
    recent_start = end - pd.Timedelta(days=27)
    prior_start = recent_start - pd.Timedelta(days=28)
    if not activities.empty and activities["date"].min() <= prior_start:
        recent_swim = filter_activities(activities, recent_start, end, ["swim"])["duration_min"].sum() / 240
        prior_swim = filter_activities(activities, prior_start, recent_start - pd.Timedelta(days=1), ["swim"])["duration_min"].sum() / 240
        if prior_swim - recent_swim >= .5:
            patterns.append(f"Swim volume decreased: {recent_swim:.2f} hours/week in {recent_start.date()} through {end.date()}, compared with {prior_swim:.2f} hours/week in the preceding 28 days.")
    load = daily_load(activities, daily)
    recent = load.loc[(load.index >= end - pd.Timedelta(days=13)) & (load.index <= end), "acwr"]
    if recent.gt(1.3).any():
        patterns.append(f"Load ratio exceeded the descriptive 1.30 threshold on {int(recent.gt(1.3).sum())} days in { (end-pd.Timedelta(days=13)).date()} through {end.date()}; maximum {recent.max():.2f}. This is a rolling-mean metric, not an injury prediction.")
    change = summary["change_vs_7_day_window_28_days_earlier"]
    if change["resting_hr"] is not None and change["sleep_hours"] is not None and change["resting_hr"] >= 2 and change["sleep_hours"] < 0:
        patterns.append(f"Recovery measurements changed: 7-day mean resting heart rate increased {change['resting_hr']:.2f} bpm and sleep changed {change['sleep_hours']:.2f} hours versus the 7-day window 28 days earlier. This describes the measurements and does not establish a cause.")
    # Compare only completed Monday-Sunday weeks, never the current partial week.
    current_monday = end.to_period("W-SUN").start_time
    for offset in (1, 2):
        start = current_monday - pd.Timedelta(weeks=offset)
        finish = start + pd.Timedelta(days=6)
        if not activities.empty and activities["date"].min().normalize() <= start:
            runs = filter_activities(activities, start, finish, ["run"])
            if not runs["duration_min"].ge(90).any():
                patterns.append(f"Long-run check: no recorded run of at least 90 minutes in {start.date()} through {finish.date()}. Whether one was planned is unknown.")
    return patterns


def write_trends(activities, daily):
    ensure_directories()
    summary = compute_summary(activities, daily)
    patterns = detect_patterns(activities, daily, summary)
    summary["patterns"] = patterns
    (OUTPUT / "trends.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    weekly_volume(activities).to_csv(OUTPUT / "weekly_volume.csv")
    (DOCS / "garmin" / "trends.md").write_text(
        f"# Computed training trends\n\nDate range: {summary['start_date']} through {summary['end_date']}. Data as of {summary['data_as_of']}.\n\n"
        + "Source: data/clean/activities.parquet and data/clean/daily.parquet. All values computed in Python.\n\n"
        + "```json\n" + json.dumps(summary, indent=2, allow_nan=False) + "\n```\n", encoding="utf-8")
    (DOCS / "garmin" / "patterns.md").write_text(
        f"# Descriptive pattern checks\n\nData as of {summary['data_as_of']}. Source: output/trends.json.\n\n"
        + "\n".join(f"- {p}" for p in patterns or ["No configured descriptive checks fired."])
        + "\n\nThese checks do not diagnose illness or prescribe training. Missing measurements are not zeros.\n", encoding="utf-8")
    return summary


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(write_trends(*load_tables()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
