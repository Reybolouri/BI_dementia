import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Care Systems – Dementia Activity Dashboard",
    layout="wide"
)

# --------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Expects two files in the same folder:
      - family_date_aggregate.csv  (steps + indoor motion per family/day)
      - household_accel_day.csv    (accelerometer events per family/day)

    Required columns:
      family_date_aggregate.csv : family_id, dt, steps, motion
      household_accel_day.csv   : family_id, dt, fam_accel_events
    """
    df_steps = pd.read_csv("family_date_aggregate.csv")
    df_accel = pd.read_csv("household_accel_day.csv")

    # Parse datetime
    df_steps["dt"] = pd.to_datetime(df_steps["dt"])
    df_accel["dt"] = pd.to_datetime(df_accel["dt"])

    # Merge on family_id + dt
    df = df_steps.merge(
        df_accel,
        on=["family_id", "dt"],
        how="left"
    )

    # Clean / helper columns
    df["date"] = df["dt"].dt.date
    df["day_of_week"] = df["dt"].dt.day_name()

    # Clip negatives to 0 just in case
    for col in ["steps", "motion", "fam_accel_events"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


df = load_data()

# --------------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------------
st.sidebar.title("Filters")

# Family filter
families = sorted(df["family_id"].unique())
family_choice = st.sidebar.selectbox(
    "Choose a family",
    options=["All families"] + families
)

# Date range filter
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["date"] >= start_date) & (df["date"] <= end_date)
if family_choice != "All families":
    mask &= df["family_id"] == family_choice

filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data available for this selection. Try another family or date range.")
    st.stop()

# --------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------
st.title("Smart Care Systems – Dementia Activity Knowledge Dashboard")
st.caption(
    "Visualizing household daily steps, indoor motion, and accelerometer events "
    "for dementia patients and caregivers."
)

# --------------------------------------------------------------------
# KPI METRICS
# --------------------------------------------------------------------
st.subheader("Summary Metrics")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Number of days", filtered["date"].nunique())
k2.metric("Avg daily steps", f"{filtered['steps'].mean():,.0f}")
k3.metric("Avg indoor motion", f"{filtered['motion'].mean():,.0f}")

if "fam_accel_events" in filtered.columns and filtered["fam_accel_events"].notna().any():
    k4.metric("Avg accel events", f"{filtered['fam_accel_events'].mean():,.0f}")
else:
    k4.metric("Avg accel events", "N/A")

# --------------------------------------------------------------------
# TIME-SERIES VISUALS
# --------------------------------------------------------------------
st.subheader("Daily Activity Over Time")

ts_df = filtered.set_index("dt").sort_index()

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Steps & Indoor Motion (time series)**")
    cols = [c for c in ["steps", "motion"] if c in ts_df.columns]
    if cols:
        st.line_chart(ts_df[cols])
    else:
        st.info("Steps/motion columns not found.")

with c2:
    st.markdown("**Accelerometer Events (time series)**")
    if "fam_accel_events" in ts_df.columns and ts_df["fam_accel_events"].notna().any():
        st.line_chart(ts_df[["fam_accel_events"]])
    else:
        st.info("No accelerometer events available for this selection.")

# --------------------------------------------------------------------
# SCATTER: STEPS VS MOTION
# --------------------------------------------------------------------
st.subheader("Relationship Between Steps and Indoor Motion")

scatter_df = filtered.dropna(subset=["steps", "motion"]).copy()

if scatter_df.empty:
    st.info("Not enough data to show scatter plot of steps vs motion.")
else:
    use_log = st.checkbox(
        "Use log scale (log(steps+1), log(motion+1))",
        value=False
    )
    if use_log:
        scatter_df["log_steps"] = np.log1p(scatter_df["steps"])
        scatter_df["log_motion"] = np.log1p(scatter_df["motion"])
        x_col, y_col = "log_steps", "log_motion"
        x_label, y_label = "log(steps+1)", "log(motion+1)"
    else:
        x_col, y_col = "steps", "motion"
        x_label, y_label = "steps", "motion"

    st.caption(f"Each point = one family-day. Axes: {x_label} vs {y_label}.")
    st.scatter_chart(scatter_df, x=x_col, y=y_col)

    corr = scatter_df[x_col].corr(scatter_df[y_col])
    st.write(f"Estimated correlation between {x_label} and {y_label}: **{corr:.2f}**")

# --------------------------------------------------------------------
# SCATTER: STEPS VS ACCEL EVENTS
# --------------------------------------------------------------------
if "fam_accel_events" in filtered.columns and filtered["fam_accel_events"].notna().any():
    st.subheader("Relationship Between Steps and Accelerometer Events")

    accel_scatter = filtered.dropna(subset=["steps", "fam_accel_events"]).copy()
    if not accel_scatter.empty:
        st.scatter_chart(accel_scatter, x="steps", y="fam_accel_events")
        corr2 = accel_scatter["steps"].corr(accel_scatter["fam_accel_events"])
        st.write(f"Estimated correlation (steps vs accel events): **{corr2:.2f}**")
    else:
        st.info("Not enough overlapping data for steps and accelerometer events.")

# --------------------------------------------------------------------
# DAY-OF-WEEK PATTERNS
# --------------------------------------------------------------------
st.subheader("Day-of-Week Patterns")

dow = (
    filtered.groupby("day_of_week")[["steps", "motion", "fam_accel_events"]]
    .mean()
    .reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
)

st.write("Average activity by day of week:")
st.dataframe(dow)

st.markdown("**Steps & Motion by Day of Week**")
st.bar_chart(dow[["steps", "motion"]])

if "fam_accel_events" in dow.columns and dow["fam_accel_events"].notna().any():
    st.markdown("**Accelerometer Events by Day of Week**")
    st.line_chart(dow[["fam_accel_events"]])

st.caption(
    "Use these patterns in your report to discuss weekly routines, high/low activity days, "
    "and potential links to caregiver schedules."
)

# --------------------------------------------------------------------
# RAW DATA
# --------------------------------------------------------------------
st.subheader("Filtered Data Preview")
st.dataframe(filtered.sort_values(["family_id", "dt"]))
