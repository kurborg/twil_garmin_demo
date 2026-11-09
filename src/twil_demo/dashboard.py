"""Optional UI: uv sync --extra dashboard; python -m streamlit run src/twil_demo/dashboard.py."""

import pandas as pd

from twil_demo.build_report import format_value, latest_date
from twil_demo.charts import rolling_daily
from twil_demo.metrics import daily_load, load_tables, total_value, weekly_volume
from twil_demo.paths import DOCS


def main() -> None:
    try:
        import streamlit as st
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Install the optional dashboard packages with `uv sync --extra dashboard`, then run "
            "`python -m streamlit run src/twil_demo/dashboard.py`."
        ) from exc

    st.set_page_config(page_title="Garmin training", page_icon="🏊", layout="wide")
    st.title("Your training, in context")
    try:
        activities, daily = load_tables()
        end = latest_date(activities, daily)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    start = pd.concat([activities["date"], daily["date"]]).dropna().min().normalize()
    bounds = st.sidebar.date_input("Date range", (max(start, end - pd.Timedelta(days=89)).date(), end.date()), min_value=start.date(), max_value=end.date())
    if len(bounds) != 2:
        st.info("Choose both a start and an end date.")
        st.stop()
    left, right = [pd.Timestamp(day) for day in bounds]
    sports = st.sidebar.multiselect("Sports", sorted(activities["sport"].dropna().unique()), default=sorted(activities["sport"].dropna().unique()))
    view = activities[(activities["date"] >= left) & (activities["date"] < right + pd.Timedelta(days=1)) & activities["sport"].isin(sports)].copy()
    days = daily[(daily["date"] >= left) & (daily["date"] <= right)].copy()
    recent = days[days["date"] >= max(left, right - pd.Timedelta(days=6))]
    st.caption(f"Showing {left:%Y-%m-%d} through {right:%Y-%m-%d}. Sports filter applies to activity metrics; wellness measurements are daily across sports.")
    columns = st.columns(5)
    minutes = total_value(view["duration_min"])
    for column, name, value in zip(columns, ["Training hours", "Sessions", "Sleep, final 7 days", "Resting HR, final 7 days", "Weight, final 7 days"], [format_value(None if minutes is None else minutes / 60, " h"), str(len(view)), format_value(recent["sleep_hours"].mean(), " h"), format_value(recent["resting_hr"].mean(), " bpm", 0), format_value(recent["weight_lb"].mean(), " lb")]):
        column.metric(name, value)
    st.caption("Final-seven-day wellness cards use available measurements within the selected range.")

    weekly = weekly_volume(view)
    st.subheader("Weekly volume")
    if weekly.empty:
        st.info("No activities match these filters.")
    else:
        unknown = weekly.isna().any(axis=1)
        if unknown.any():
            st.caption(f"{int(unknown.sum())} weeks omitted because at least one session has unknown duration.")
        chart_data = weekly.loc[~unknown].rename_axis("week").reset_index().melt(id_vars="week", var_name="sport", value_name="hours")
        st.plotly_chart(px.bar(chart_data, x="week", y="hours", color="sport", labels={"week": "Week starting Monday", "hours": "Hours"}), width="stretch")

    st.subheader("Sleep and resting heart rate")
    recovery = make_subplots(specs=[[{"secondary_y": True}]])
    recovery.add_trace(go.Bar(x=days["date"], y=days["sleep_hours"], name="Sleep (h)", marker_color="#aecbd5"), secondary_y=False)
    # Compute rolling windows before slicing so the beginning of a view retains its history.
    rhr = rolling_daily(daily, "resting_hr").loc[left:right]
    recovery.add_trace(go.Scatter(x=rhr.index, y=rhr, name="Resting HR, 7-day mean", connectgaps=False), secondary_y=True)
    recovery.update_yaxes(title_text="Sleep (hours)", secondary_y=False, rangemode="tozero")
    recovery.update_yaxes(title_text="Resting heart rate (bpm)", secondary_y=True)
    st.plotly_chart(recovery, width="stretch")

    st.subheader("Recorded activity load ratio")
    sport_history = activities[activities["sport"].isin(sports)]
    load = daily_load(sport_history, daily).loc[left:right]
    if load["acwr"].notna().any():
        st.plotly_chart(px.line(load.reset_index(), x=load.index.name or "index", y="acwr", labels={"acwr": "7-day mean / 28-day mean"}), width="stretch")
    else:
        st.info("Insufficient complete load history for this selection.")
    st.caption("Uses selected sports, with history before the selected start date. This computed ratio is separate from Garmin's native ratio.")

    st.subheader("Measurement coverage")
    coverage = pd.DataFrame({"Metric": ["Sleep", "Resting HR", "Weight", "Stress", "VO2 max"], "Days measured": [int(days[c].notna().sum()) for c in ["sleep_hours", "resting_hr", "weight_lb", "stress_avg", "vo2max"]]})
    coverage["Days selected"] = (right - left).days + 1
    st.dataframe(coverage, hide_index=True, width="stretch")

    st.subheader("Computed observations · full export")
    st.caption("These precomputed observations describe the full export and do not change with the filters above.")
    path = DOCS / "garmin" / "patterns.md"
    st.text(path.read_text(encoding="utf-8") if path.exists() else "Run python -m twil_demo.trends first.")
    st.subheader("Activities")
    st.dataframe(view.sort_values("date", ascending=False), hide_index=True, width="stretch")

    st.subheader("Ask your data · full export")
    st.caption("Questions use the full dataset and saved RAG index, independently of the chart filters. Include dates and sports in your question.")
    with st.form("ask"):
        question = st.text_input("Question", placeholder="How many hours did I run in the last 4 weeks?")
        submitted = st.form_submit_button("Ask TwIL")
    if submitted and question.strip():
        from twil_demo.ask_garmin import ask_garmin

        try:
            with st.spinner("Computing facts and asking local TwIL…"):
                st.write(ask_garmin(question.strip()))
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
