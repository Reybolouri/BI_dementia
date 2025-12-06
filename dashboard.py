import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Dementia Care Activity Dashboard",
    layout="wide"
)

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():
    # File 1: household-level daily steps and motion
    df1 = pd.read_csv("family_date_aggregate.csv")

    # File 2: accelerometer daily signals
    df2 = pd.read_csv("household_accel_day.csv")

    # Convert DT column to datetime
    df1["dt"] = pd.to_datetime(df1["dt"])
    df2["dt"] = pd.to_datetime(df2["dt"])

    # Merge on family_id + dt
    df = df1.merge(df2, on=["family_id", "dt"], how="left")

    # Add helper columns
    df["date"] = df["dt"].dt.date
    df["day_of_week"] = df["dt"].dt.day_name()

    # Replace negatives or junk values with NaN
    for col in ["steps", "motion", "fam_accel_events"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df

df = load_data()

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")

families = sorted(df["family_id"].unique())
family_choice = st.sidebar.selectbox("Select family:", ["All families"] + families)

min_date, max_date = df["date"].min(), df["date"].max()
start_date, end_date = st.sidebar.date_input(
    "Select date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

mask = (df["date"] >= start_date) & (df["date"] <= end_date)
if family_choice != "All families":
    mask &= df["family_id"] == family_choice
filtered = df.loc[mask]

if filtered.empty:
    st.warning("No data available for this selection.")
    st.stop()

# -------------------------
# TITLE
# -------------------------
st.title("Smart Care Systems – Knowledge Discovery Dashboard")
st.caption("Analyzing steps, indoor motion, and accelerometer signals for dementia caregiver–patient households.")

# -------------------------
# KPI METRICS
# -------------------------
st.subheader("Daily Activity Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Days Included", filtered["date"].nunique())
col2.metric("Avg Daily Steps", f"{filtered['steps'].mean():,.0f}")
col3.metric("Avg Indoor Motion", f"{filtered['motion'].mean():,.0f}")
if "fam_accel_events" in filtered.columns:
    col4.metric("Avg Accelerometer Events", f"{filtered['fam_accel_events'].mean():,.0f}")
else:
    col4.metric("Avg Accelerometer Events", "N/A")

# -------------------------
# TIME SERIES VISUALIZATIONS
# -------------------------
st.subheader("Daily Activity Over Time")

plot_df = filtered.set_index("dt").sort_index()

ts_col1, ts_col2 = st.columns(2)

with ts_col1:
    st.markdown("**Steps & Indoor Motion**")
    st.line_chart(plot_df[["steps", "motion"]])

with ts_col2:
    if "fam_accel_events" in plot_df.columns:
        st.markdown("**Accelerometer Events**")
        st.line_chart(plot_df[["fam_accel_events"]])
    else:
        st.info("No accelerometer data provided.")

# -------------------------
# SCATTER ANALYSIS: STEPS VS MOTION
# -------------------------
st.subheader("Correlation Between Steps and Indoor Motion")

scatter_df = filtered.dropna(subset=["steps", "motion"])

st.scatter_chart(scatter_df, x="steps", y="motion")

corr = scatter_df["steps"].corr(scatter_df["motion"])
st.write(f"**Correlation (steps vs motion): {corr:.2f}**")

# -------------------------
# RELATIONSHIP: STEPS VS ACCELEROMETER EVENTS
# -------------------------
if "fam_accel_events" in filtered.columns:
    st.subheader("Relationship Between Steps and Accelerometer Events")

    acc_df = filtered.dropna(subset=["steps", "fam_accel_events"])
    st.scatter_chart(acc_df, x="steps", y="fam_accel_events")

    corr2 = acc_df["steps"].corr(acc_df["fam_accel_events"])
    st.write(f"**Correlation (steps vs accelerometer events): {corr2:.2f}**")

# -------------------------
# DAY OF WEEK PATTERNS
# -------------------------
st.subheader("Day-of-Week Activity Patterns")

dow = (
    filtered.groupby("day_of_week")[["steps", "motion", "fam_accel_events"]]
    .mean()
    .reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ])
)

st.bar_chart(dow[["steps", "motion"]])

if "fam_accel_events" in dow.columns:
    st.line_chart(dow[["fam_accel_events"]])

# -------------------------
# RAW DATA VIEW
# -------------------------
st.subheader("Filtered Data")
st.dataframe(filtered)

