"""Shared deterministic calculations for summaries, charts, and questions."""

from pathlib import Path

import numpy as np
import pandas as pd

from twil_demo.paths import CLEAN


def load_tables(clean: Path = CLEAN) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        activities = pd.read_parquet(clean / "activities.parquet")
        daily = pd.read_parquet(clean / "daily.parquet")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Clean tables are missing. Run python -m twil_demo.ingest first."
        ) from exc
    for frame in (activities, daily):
        frame["date"] = pd.to_datetime(frame["date"])
    return activities.sort_values("date"), daily.sort_values("date")


def filter_activities(activities, start=None, end=None, sports=None):
    result = activities.copy()
    if start is not None:
        result = result[result["date"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        result = result[result["date"] < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)]
    if sports is not None:
        result = result[result["sport"].isin(sports)]
    return result


def weekly_volume(activities: pd.DataFrame) -> pd.DataFrame:
    if activities.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="week"))
    work = activities.assign(week=activities["date"].dt.to_period("W-SUN").dt.start_time)
    groups = work.groupby(["week", "sport"])["duration_min"]
    totals = groups.sum(min_count=1).where(groups.count() == groups.size())
    grouped = totals.unstack(fill_value=0)
    weeks = pd.date_range(grouped.index.min(), grouped.index.max(), freq="W-MON")
    return grouped.reindex(weeks, fill_value=0).rename_axis("week") / 60


def daily_load(activities: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rolling mean ratio, distinct from Garmin's native acute/chronic ratio.

    A recorded session with unknown load makes that day's load unknown. Days
    without sessions within the exported activity history are treated as rest.
    """
    if activities.empty:
        return pd.DataFrame(columns=["training_load", "acwr"], index=pd.DatetimeIndex([], name="date"))
    work = activities.assign(day=activities["date"].dt.normalize()).copy()
    transition = work["sport"].eq("transition") & work["training_load"].isna()
    work.loc[transition, "training_load"] = 0.
    grouped = work.groupby("day")["training_load"]
    load = grouped.sum(min_count=1)
    # Transitions don't have an independent load; missing load on other sessions
    # must not silently become a zero or a partial day total.
    unknown = work[work["training_load"].isna() & work["sport"].ne("transition")]["day"]
    load.loc[load.index.isin(unknown)] = np.nan
    end = work["day"].max()
    if daily is not None and not daily.empty:
        end = max(end, daily["date"].max().normalize())
    index = pd.date_range(work["day"].min(), end, freq="D", name="date")
    load = load.reindex(index, fill_value=0)
    chronic = load.rolling(28, min_periods=28).mean()
    ratio = load.rolling(7, min_periods=7).mean() / chronic.where(chronic > 0)
    return pd.DataFrame({"training_load": load, "acwr": ratio})


def number(value, digits=2):
    return None if pd.isna(value) or not np.isfinite(value) else round(float(value), digits)


def total_value(values: pd.Series):
    """A complete total, or None when a recorded measurement is unknown."""
    return None if values.isna().any() else float(values.sum())


def hours_value(values: pd.Series):
    total = total_value(values)
    return None if total is None else number(total / 60)


def compute_summary(activities, daily, start=None, end=None) -> dict:
    """Summarize an inclusive calendar range, defaulting to the latest 28 days."""
    dates = [f["date"].max().normalize() for f in (activities, daily) if not f.empty]
    if not dates:
        raise ValueError("No dated records are available.")
    end = pd.Timestamp(end).normalize() if end is not None else max(dates)
    start = pd.Timestamp(start).normalize() if start is not None else end - pd.Timedelta(days=27)
    if start > end:
        raise ValueError("Start date must be on or before end date.")
    view = filter_activities(activities, start, end)
    days = daily[daily["date"].between(start, end)].copy()
    last_seven = days[days["date"] >= end - pd.Timedelta(days=6)]
    by_sport = {}
    for sport, group in view.groupby("sport"):
        by_sport[sport] = {
            "sessions": len(group),
            "hours": hours_value(group["duration_min"]),
            "distance_km": number(total_value(group["distance_km"])),
            "duration_observed_sessions": int(group["duration_min"].count()),
        }
    latest_load = daily_load(activities, daily).reindex([end])
    recovery = {}
    for column in ("resting_hr", "sleep_hours", "sleep_score", "stress_avg", "weight_lb", "vo2max"):
        values = last_seven[column] if column in last_seven else pd.Series(dtype=float)
        recovery[column] = {"mean": number(values.mean()), "observed_days": int(values.count()), "calendar_days": 7}
    previous = daily[daily["date"].between(end - pd.Timedelta(days=34), end - pd.Timedelta(days=28))]
    changes = {}
    for column in ("resting_hr", "sleep_hours", "weight_lb"):
        now = recovery[column]
        prior_values = previous[column] if column in previous else pd.Series(dtype=float)
        changes[column] = number(last_seven[column].mean() - prior_values.mean()) if now["observed_days"] >= 4 and prior_values.count() >= 4 else None
    return {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "data_as_of": max(dates).date().isoformat(),
        "calendar_days": (end - start).days + 1,
        "sessions": len(view),
        "hours": hours_value(view["duration_min"]),
        "by_sport": by_sport,
        "recovery_last_7_days": recovery,
        "change_vs_7_day_window_28_days_earlier": changes,
        "acwr_latest": number(latest_load["acwr"].iloc[0]),
        "acwr_definition": "7-day mean / 28-day mean of activityTrainingLoad; not Garmin's native ratio",
        "load_observed_sessions": int(view["training_load"].count()),
        "activity_source": "data/clean/activities.parquet",
        "daily_source": "data/clean/daily.parquet",
    }
