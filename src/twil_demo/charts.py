"""Render static charts without contacting Ollama: python -m twil_demo.charts."""

import argparse
import os
from pathlib import Path

import pandas as pd

from twil_demo.metrics import daily_load, load_tables, weekly_volume
from twil_demo.paths import CHARTS, OUTPUT


def _range_label(activities: pd.DataFrame, daily: pd.DataFrame) -> str:
    dates = pd.concat([activities["date"], daily["date"]]).dropna()
    if dates.empty:
        return "No dated records"
    return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"


def rolling_daily(daily: pd.DataFrame, column: str) -> pd.Series:
    """Use calendar windows, requiring four measurements per seven days."""
    values = daily.set_index("date")[column].sort_index()
    return values.rolling("7D", min_periods=4).mean()


def build_charts(
    activities: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    output_dir: Path = CHARTS,
) -> list[Path]:
    if activities is None or daily is None:
        activities, daily = load_tables()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = OUTPUT / ".matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "font.size": 10})
    label = _range_label(activities, daily)
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        paths.append(path)

    weekly = weekly_volume(activities)
    fig, ax = plt.subplots(figsize=(11, 4))
    if not weekly.empty:
        unknown = weekly.isna().any(axis=1)
        # An incomplete discipline makes the total for that week unknown.
        # Keep its calendar position, but omit the whole stacked bar.
        # A positional index prevents pandas from turning weekly dates into
        # period ordinals while our tick positions remain 0..N-1.
        weekly.mask(unknown, axis=0).reset_index(drop=True).plot.bar(stacked=True, ax=ax, width=0.9, colormap="tab20")
        stride = max(1, len(weekly) // 10)
        ticks = list(range(0, len(weekly), stride))
        ax.set_xticks(ticks, [f"{weekly.index[i]:%Y-%m-%d}" for i in ticks], rotation=35, ha="right")
        ax.legend(title="Sport", loc="upper left", bbox_to_anchor=(1.01, 1))
        if unknown.any():
            ax.text(0.01, 0.97, f"{int(unknown.sum())} weeks omitted: unknown session duration", transform=ax.transAxes, va="top", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No activity data", transform=ax.transAxes, ha="center")
    ax.set(title=f"Weekly training volume | {label}", ylabel="Hours", xlabel="Week starting Monday")
    ax.set_ylim(bottom=0)
    save(fig, "weekly_volume.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(daily["date"], daily["sleep_hours"], width=0.9, color="#aecbd5", label="Sleep")
    ax.set(ylabel="Sleep (hours)", title=f"Sleep and resting heart rate | {label}")
    ax.set_ylim(bottom=0)
    right = ax.twinx()
    rhr = rolling_daily(daily, "resting_hr")
    right.plot(rhr.index, rhr, color="#183d59", label="Resting HR, 7-day mean")
    right.set_ylabel("Resting heart rate (bpm)")
    ax.legend(loc="upper left")
    right.legend(loc="upper right")
    fig.autofmt_xdate()
    save(fig, "recovery.png")

    load = daily_load(activities, daily)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if not load.empty and load["acwr"].notna().any():
        ax.plot(load.index, load["acwr"], color="#286c68")
    else:
        ax.text(0.5, 0.5, "Insufficient complete load history for a 28-day ratio", transform=ax.transAxes, ha="center")
    ax.set(title=f"Recorded activity load ratio | {label}", ylabel="7-day mean / 28-day mean", xlabel="Date")
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate()
    save(fig, "acwr.png")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    measured = daily.dropna(subset=["weight_lb"])
    ax.scatter(measured["date"], measured["weight_lb"], s=9, alpha=0.45, color="#97a6ac", label="Measurements")
    weight = rolling_daily(daily, "weight_lb")
    ax.plot(weight.index, weight, color="#183d59", label="7-day mean (at least 4 measurements)")
    ax.set(title=f"Weight | {label}", ylabel="Weight (lb)", xlabel="Date")
    ax.legend()
    fig.autofmt_xdate()
    save(fig, "weight.png")
    return paths


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for path in build_charts():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
