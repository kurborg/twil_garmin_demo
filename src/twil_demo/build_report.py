"""Build an offline HTML report: python -m twil_demo.build_report."""

import argparse
from html import escape
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from twil_demo.metrics import daily_load, load_tables, total_value
from twil_demo.paths import CHARTS, DASHBOARD, DOCS


def format_value(value, suffix: str = "", digits: int = 1) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:,.{digits}f}{suffix}"


def latest_date(activities: pd.DataFrame, daily: pd.DataFrame) -> pd.Timestamp:
    dates = pd.concat([activities["date"], daily["date"]]).dropna()
    if dates.empty:
        raise ValueError("No dated records. Run python -m twil_demo.ingest first.")
    return dates.max().normalize()


def report_cards(activities: pd.DataFrame, daily: pd.DataFrame) -> dict[str, str]:
    end = latest_date(activities, daily)
    recent = daily[(daily["date"] >= end - pd.Timedelta(days=6)) & (daily["date"] <= end)]
    acts28 = activities[(activities["date"] >= end - pd.Timedelta(days=27)) & (activities["date"] < end + pd.Timedelta(days=1))]
    load = daily_load(activities, daily)
    ratio = load["acwr"].get(end, float("nan"))
    minutes = total_value(acts28["duration_min"])
    return {
        "Training hours, last 28 days": format_value(None if minutes is None else minutes / 60, " h"),
        "Sleep, last 7 days": format_value(recent["sleep_hours"].mean(), " h"),
        "Resting HR, last 7 days": format_value(recent["resting_hr"].mean(), " bpm", 0),
        "Weight, last 7 days": format_value(recent["weight_lb"].mean(), " lb"),
        "Load ratio, latest day": format_value(ratio, "", 2),
    }


def build_report(
    activities: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    output_path: Path = DASHBOARD / "report.html",
    charts_dir: Path = CHARTS,
    docs_dir: Path = DOCS,
) -> Path:
    if activities is None or daily is None:
        activities, daily = load_tables()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end = latest_date(activities, daily)
    all_dates = pd.concat([activities["date"], daily["date"]]).dropna()
    cards = "".join(
        f'<article class="card"><span>{escape(name)}</span><strong>{escape(value)}</strong></article>'
        for name, value in report_cards(activities, daily).items()
    )
    figures = []
    for name, title in [
        ("weekly_volume.png", "Volume by sport"),
        ("recovery.png", "Sleep and resting heart rate"),
        ("acwr.png", "Recorded activity load"),
        ("weight.png", "Weight measurements"),
    ]:
        path = charts_dir / name
        if path.exists():
            url = quote(os.path.relpath(path, output_path.parent))
            body = f'<img src="{escape(url, quote=True)}" alt="{escape(title)}">'
        else:
            body = "<p>Chart not generated yet. Run <code>python -m twil_demo.charts</code>.</p>"
        figures.append(f'<section class="panel"><h2>{title}</h2>{body}</section>')

    patterns_path = docs_dir / "garmin" / "patterns.md"
    patterns = patterns_path.read_text(encoding="utf-8") if patterns_path.exists() else "Run python -m twil_demo.trends to generate pattern observations."
    recent = daily[(daily["date"] >= end - pd.Timedelta(days=6)) & (daily["date"] <= end)]
    coverage = "".join(
        f"<tr><td>{escape(label)}</td><td>{recent[column].notna().sum()} / 7</td><td>{daily[column].notna().sum()} / {len(daily)}</td></tr>"
        for column, label in [("sleep_hours", "Sleep"), ("resting_hr", "Resting heart rate"), ("weight_lb", "Weight"), ("stress_avg", "Stress"), ("vo2max", "VO2 max")]
    )
    sources = []
    for path in sorted((docs_dir / "garmin").glob("week_*.md"), reverse=True)[:12]:
        url = quote(os.path.relpath(path, output_path.parent))
        sources.append(f'<li><a href="{escape(url, quote=True)}">{escape(path.stem)}</a></li>')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Garmin training report</title><style>
:root{{color-scheme:light;--ink:#173247;--muted:#576c78;--line:#dce5e8}}
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f5;color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}}header{{margin-bottom:26px}}h1{{font-size:34px;margin:0 0 6px}}h2{{font-size:20px;margin:0 0 16px}}
p{{margin:8px 0;color:var(--muted)}}.eyebrow{{font-size:12px;letter-spacing:2px;font-weight:700;color:#286c68}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:14px;margin-bottom:24px}}
.card,.panel{{background:white;border:1px solid var(--line);border-radius:14px;padding:22px}}
.card span{{display:block;min-height:42px;font-size:13px;color:var(--muted)}}.card strong{{font-size:28px;display:block;margin-top:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.panel img{{display:block;width:100%;height:auto}}.panel{{margin-bottom:20px;overflow:hidden}}
pre{{white-space:pre-wrap;font:14px/1.65 system-ui,sans-serif;margin:0}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid var(--line);padding:10px 6px}}
a{{color:#176b75}}ul{{padding-left:20px}}footer{{font-size:12px;color:var(--muted);margin:24px 0}}
@media(max-width:800px){{main{{padding:18px}}.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
@media print{{body{{background:white}}main{{padding:0}}.panel,.card{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">LOCAL GARMIN REPORT</div><h1>Your training, in context</h1>
<p>Records: {all_dates.min():%Y-%m-%d} to {end:%Y-%m-%d} · {len(activities):,} activities.</p>
<p>Cards end on {end:%Y-%m-%d}, the latest date in the export. Averages use available measurements; coverage appears below.</p></header>
<div class="cards">{cards}</div><div class="grid">{''.join(figures)}</div>
<div class="grid"><section class="panel"><h2>Computed observations</h2><pre>{escape(patterns)}</pre></section>
<section class="panel"><h2>Measurement coverage</h2><table><thead><tr><th>Metric</th><th>Last 7 days</th><th>All days</th></tr></thead><tbody>{coverage}</tbody></table>
<p>Missing measurements remain missing. Seven-day chart averages require at least four measurements. The load ratio compares recorded activity load means over 7 and 28 calendar days; it is separate from Garmin's native ratio.</p></section></div>
<section class="panel"><h2>Recent weekly sources</h2><ul>{''.join(sources) or '<li>No weekly summaries yet. Run python -m twil_demo.summarize.</li>'}</ul></section>
<footer>Computed locally from the cleaned Garmin tables. This report makes no model calls and uses no remote assets. Chart gaps indicate unavailable measurements.</footer>
</main></body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(f"Wrote {build_report()}")


if __name__ == "__main__":
    main()
