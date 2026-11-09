"""Normalize a Garmin account JSON export without extracting its FIT archives."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .paths import CLEAN, RAW


ACTIVITY_COLUMNS = [
    "activity_id", "date", "start_time_utc", "sport", "activity_type", "name",
    "parent_id", "is_parent", "distance_km", "duration_min", "avg_hr", "max_hr",
    "avg_power_w", "training_load", "source_file",
]
DAILY_COLUMNS = [
    "resting_hr", "stress_avg", "sleep_hours", "sleep_score", "nap_hours",
    "weight_lb", "vo2max", "vo2max_running", "vo2max_cycling", "training_readiness",
    "garmin_acute_load", "garmin_chronic_load", "garmin_acwr", "hrv_weekly_average",
]


def number(value):
    """Keep absent, non-finite, and negative Garmin sentinel values missing."""
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) and value >= 0 else float("nan")


def identifier(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return str(int(value))
    return str(value)


def timestamp(value, *, milliseconds=False):
    if value is None:
        return pd.NaT
    if milliseconds or isinstance(value, (float, int)):
        return pd.to_datetime(value, unit="ms", errors="coerce")
    date = pd.to_datetime(value, errors="coerce")
    if not pd.isna(date) and date.tzinfo is not None:
        date = date.tz_localize(None)
    return date


def sport_name(activity_type: str) -> str:
    name = activity_type.lower()
    if "transition" in name:
        return "transition"
    if "swim" in name:
        return "swim"
    if "cycling" in name or "biking" in name or name == "virtual_ride":
        return "bike"
    if "running" in name:
        return "run"
    if "strength" in name:
        return "strength"
    return "other"


def records(raw: Path, pattern: str):
    """Yield source paths and dictionaries; malformed files fail visibly."""
    for path in sorted(raw.rglob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read Garmin JSON {path}: {exc}") from exc
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list or object in {path}")
        for row in data:
            if isinstance(row, dict):
                yield str(path.relative_to(raw)), row


def load_activities(raw: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    invalid = 0
    for source, wrapper in records(raw, "*summarizedActivities.json"):
        for activity in wrapper.get("summarizedActivitiesExport", []):
            date = timestamp(activity.get("startTimeLocal"), milliseconds=True)
            activity_id = identifier(activity.get("activityId"))
            if pd.isna(date) or activity_id is None:
                invalid += 1
                continue
            kind = str(activity.get("activityType") or "other")
            rows.append({
                "activity_id": activity_id,
                "date": date,
                "start_time_utc": pd.to_datetime(activity.get("startTimeGmt"), unit="ms", utc=True),
                "sport": sport_name(kind),
                "activity_type": kind,
                "name": str(activity.get("name") or ""),
                "parent_id": identifier(activity.get("parentId")),
                "is_parent": bool(activity.get("parent")),
                # Garmin JSON uses centimetres and milliseconds, unlike FIT.
                "distance_km": number(activity.get("distance")) / 100_000,
                "duration_min": number(activity.get("duration")) / 60_000,
                "avg_hr": number(activity.get("avgHr")),
                "max_hr": number(activity.get("maxHr")),
                "avg_power_w": number(activity.get("avgPower")),
                "training_load": number(activity.get("activityTrainingLoad")),
                "source_file": source,
            })
    if not rows:
        raise ValueError(
            f"No activities found under {raw}. Expected *summarizedActivities.json "
            "files from the Garmin full account export."
        )
    frame = pd.DataFrame(rows, columns=ACTIVITY_COLUMNS)
    duplicates = int(frame.duplicated("activity_id").sum())
    frame = frame.drop_duplicates("activity_id", keep="last")
    parents_with_children = set(frame["parent_id"].dropna())
    parent_mask = frame["activity_id"].isin(parents_with_children)
    excluded = frame.loc[parent_mask]
    orphan_parents = frame.loc[frame["is_parent"] & ~parent_mask, "activity_id"].tolist()
    child_totals = frame.groupby("parent_id")["duration_min"].sum(min_count=1)
    incomplete = []
    for row in excluded.itertuples():
        child_duration = child_totals.get(row.activity_id, float("nan"))
        if pd.notna(row.duration_min) and pd.notna(child_duration) and child_duration < row.duration_min * 0.95:
            incomplete.append(row.activity_id)
    frame = frame.loc[~parent_mask].sort_values(["date", "activity_id"]).reset_index(drop=True)
    return frame, {
        "raw_activity_records": len(rows),
        "duplicate_activity_ids_removed": duplicates,
        "invalid_activities_skipped": invalid,
        "multisport_parent_totals_excluded": len(excluded),
        "parents_without_children_retained_as_other": orphan_parents,
        "parents_with_possible_missing_legs": incomplete,
        "multisport_policy": "Count child legs and transitions; exclude their parent totals. Retain parents without children as other.",
    }


def daily_candidates(raw: Path) -> tuple[list[dict], dict]:
    rows = []
    counts = {"main_sleep_records": 0, "nap_only_wrapper_records": 0, "records_with_naps": 0}

    def add(source, date, **values):
        parsed = timestamp(date)
        if pd.isna(parsed):
            return
        rows.append({"date": parsed.normalize(), "source_file": source, **values})

    for source, row in records(raw, "UDSFile_*.json"):
        total = next((entry for entry in row.get("allDayStress", {}).get("aggregatorList", [])
                      if entry.get("type") == "TOTAL"), {})
        add(source, row.get("calendarDate"), resting_hr=number(row.get("restingHeartRate")),
            stress_avg=number(total.get("averageStressLevel")))

    for source, row in records(raw, "*sleepData.json"):
        if "calendarDate" not in row and "napList" in row:
            counts["nap_only_wrapper_records"] += 1
        if "calendarDate" in row:
            counts["main_sleep_records"] += 1
            stages = [number(row.get(key)) for key in ("deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds")]
            seconds = sum(stages) if all(pd.notna(value) for value in stages) else float("nan")
            confirmation = str(row.get("sleepWindowConfirmationType") or "")
            add(source, row["calendarDate"], sleep_hours=seconds / 3600,
                sleep_score=number((row.get("sleepScores") or {}).get("overallScore")),
                sleep_rank=2 if "CONFIRMED_FINAL" in confirmation else 1 if "CONFIRMED" in confirmation.split("_") else 0,
                sleep_completeness=sum(pd.notna(value) for value in stages),
                observed_at=timestamp(row.get("sleepEndTimestampGMT")))
        if row.get("napList"):
            counts["records_with_naps"] += 1
            for nap in row["napList"]:
                add(source, nap.get("calendarDate"), nap_hours=number(nap.get("napTimeSec")) / 3600,
                    nap_start=str(nap.get("napStartTimestampGMT") or ""))

    for source, row in records(raw, "*userBioMetrics.json"):
        date = (row.get("metaData") or {}).get("calendarDate")
        weight = row.get("weight")
        if isinstance(weight, dict) and not row.get("userSetNullForWeight"):
            add(source, date, weight_lb=number(weight.get("weight")) / 453.59237,
                observed_at=timestamp(date))

    for source, row in records(raw, "MetricsMaxMetData_*.json"):
        sport = str(row.get("sport") or "").upper()
        column = {"RUNNING": "vo2max_running", "CYCLING": "vo2max_cycling"}.get(sport)
        if column:
            add(source, row.get("calendarDate"), **{column: number(row.get("vo2MaxValue"))},
                observed_at=timestamp(row.get("updateTimestamp")))

    for source, row in records(raw, "TrainingReadinessDTO_*.json"):
        # Use the latest readiness snapshot of each day consistently.
        add(source, row.get("calendarDate"), training_readiness=number(row.get("score")),
            hrv_weekly_average=number(row.get("hrvWeeklyAverage")),
            observed_at=timestamp(row.get("timestampLocal") or row.get("timestamp")))

    for source, row in records(raw, "MetricsAcuteTrainingLoad_*.json"):
        add(source, row.get("calendarDate"),
            garmin_acute_load=number(row.get("dailyTrainingLoadAcute")),
            garmin_chronic_load=number(row.get("dailyTrainingLoadChronic")),
            garmin_acwr=number(row.get("dailyAcuteChronicWorkloadRatio")),
            observed_at=timestamp(row.get("timestamp")))
    return rows, counts


def load_daily(raw: Path, activities: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, report = daily_candidates(raw)
    if rows:
        candidates = pd.DataFrame(rows)
        candidates["observed_at"] = pd.to_datetime(candidates.get("observed_at"), errors="coerce")
        dates = candidates["date"]
        start = min(dates.min(), activities["date"].min().normalize())
        end = max(dates.max(), activities["date"].max().normalize())
    else:
        candidates = pd.DataFrame(columns=["date", "source_file", "observed_at"])
        start, end = activities["date"].min().normalize(), activities["date"].max().normalize()
    frame = pd.DataFrame(index=pd.date_range(start, end, freq="D", name="date"))
    sources = pd.Series("", index=frame.index, dtype="str")
    for column in DAILY_COLUMNS:
        if column == "vo2max" or column not in candidates:
            frame[column] = float("nan")
            continue
        values = candidates.loc[candidates[column].notna()].copy()
        if column in ("sleep_hours", "sleep_score"):
            # Rank the whole sleep record first, so its hours and score agree.
            values = candidates.loc[candidates["sleep_rank"].notna()].copy()
            values = values.sort_values(
                ["sleep_rank", "sleep_completeness", "observed_at", "source_file"],
                na_position="first",
            )
        elif column == "nap_hours":
            values = values.drop_duplicates(["date", "nap_start"])
            frame[column] = values.groupby("date")[column].sum(min_count=1).reindex(frame.index)
            for date, group in values.groupby("date"):
                existing = set(filter(None, sources.loc[date].split(";")))
                sources.loc[date] = ";".join(sorted(existing | set(group["source_file"])))
            continue
        else:
            values = values.sort_values(["observed_at", "source_file"], na_position="first")
        selected = values.drop_duplicates("date", keep="last").set_index("date")
        frame[column] = selected[column].reindex(frame.index)
        for date, source in selected["source_file"].items():
            existing = set(filter(None, sources.loc[date].split(";")))
            existing.add(source)
            sources.loc[date] = ";".join(sorted(existing))
    frame["vo2max"] = frame["vo2max_running"].combine_first(frame["vo2max_cycling"])
    frame["source_files"] = sources
    report["duplicate_main_sleep_dates_removed"] = (
        int(candidates.loc[candidates["sleep_rank"].notna(), "date"].duplicated().sum())
        if "sleep_rank" in candidates else 0
    )
    report["sleep_policy"] = "Prefer confirmed-final, then complete stage data, then latest sleep end; naps remain separate."
    report["readiness_policy"] = "Latest available snapshot per calendar date."
    return frame.reset_index(), report


def build_tables(raw: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = Path(raw)
    if not raw.is_dir():
        raise ValueError(f"Garmin export directory does not exist: {raw}")
    activities, activity_report = load_activities(raw)
    daily, daily_report = load_daily(raw, activities)
    report = {
        **activity_report, **daily_report,
        "activities": len(activities), "daily_calendar_rows": len(daily),
        "activity_date_start": str(activities["date"].min().date()),
        "activity_date_end": str(activities["date"].max().date()),
        "daily_date_start": str(daily["date"].min().date()),
        "daily_date_end": str(daily["date"].max().date()),
        "daily_measurements_available": {column: int(daily[column].notna().sum()) for column in DAILY_COLUMNS},
        "activity_measurements_available": {
            column: int(activities[column].notna().sum())
            for column in ["distance_km", "duration_min", "avg_hr", "max_hr", "avg_power_w", "training_load"]
        },
        "missing_data_policy": "Absent measurements stay null. No forward fill or zero substitution.",
        "date_policy": "Activities use exported local start time; daily records use calendarDate. UTC activity start is retained separately.",
    }
    return activities, daily, report


def ingest(raw: Path = RAW, clean: Path = CLEAN) -> dict:
    activities, daily, report = build_tables(raw)
    clean = Path(clean)
    clean.mkdir(parents=True, exist_ok=True)
    activities.to_parquet(clean / "activities.parquet", index=False)
    daily.to_parquet(clean / "daily.parquet", index=False)
    (clean / "ingestion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW, help="Root of the Garmin account export")
    parser.add_argument("--clean", type=Path, default=CLEAN, help="Destination for the cleaned tables")
    args = parser.parse_args()
    try:
        report = ingest(args.raw, args.clean)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    print(f"Activities: {report['activities']} ({report['activity_date_start']} to {report['activity_date_end']})")
    print(f"Daily calendar: {report['daily_calendar_rows']} days")
    print(f"Wrote {args.clean / 'activities.parquet'} and {args.clean / 'daily.parquet'}")
    if report["parents_with_possible_missing_legs"] or report["parents_without_children_retained_as_other"]:
        print("Check multisport coverage warnings in ingestion_report.json.")


if __name__ == "__main__":
    main()
