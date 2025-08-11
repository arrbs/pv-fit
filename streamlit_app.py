import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="PV Fit Tool", layout="wide")

# ---------- Helpers ----------
def _to_datetime(x):
    try:
        return pd.to_datetime(x)
    except:
        return pd.NaT

def _num(x):
    try:
        return float(x)
    except:
        return np.nan

def roll_median(df, col, win_s):
    if win_s <= 0 or df.empty:
        return df
    g = df.set_index("t")[col].rolling(f"{win_s}s", center=True).median()
    out = df.copy()
    out[col] = g.reindex(out.set_index("t").index, method=None).values
    return out.dropna(subset=[col])

def resample_uniform(df, col, every_s):
    if every_s <= 0 or df.empty:
        return df
    t0, t1 = df["t"].min(), df["t"].max()
    idx = pd.date_range(t0, t1, freq=f"{every_s}s")
    s = df.set_index("t")[col].reindex(idx).interpolate(method="time")
    return pd.DataFrame({"t": idx, col: s.values}).dropna()

def ref_window_mean_at_plant(ref_df, plant_df, col_ref="I", win_s=0):
    if win_s <= 0 or ref_df.empty or plant_df.empty:
        return None
    ref = ref_df.set_index("t")[col_ref]
    half = pd.Timedelta(seconds=int(win_s))
    vals = []
    for t in plant_df["t"]:
        seg = ref.loc[t - half : t + half]
        vals.append(float(seg.mean()) if not seg.empty else np.nan)
    out = plant_df[["t"]].copy()
    out[col_ref] = vals
    return out.dropna()

def estimate_lag_seconds(ref_df, plant_df, col_ref="I", col_p="P", max_lag_s=60):
    if max_lag_s <= 0 or ref_df.empty or plant_df.empty:
        return 0
    P = plant_df[["t", col_p]].dropna()
    R = ref_df[["t", col_ref]].dropna()
    Ri = np.interp(P["t"].view("int64"), R["t"].view("int64"), R[col_ref], left=np.nan, right=np.nan)
    mask = np.isfinite(Ri) & np.isfinite(P[col_p].values)
    if mask.sum() < 10:
        return 0
    x = (Ri[mask] - np.nanmean(Ri[mask])) / (np.nanstd(Ri[mask]) + 1e-9)
    y = (P[col_p].values[mask] - np.nanmean(P[col_p].values[mask])) / (np.nanstd(P[col_p].values[mask]) + 1e-9)
    dt = np.median(np.diff(P["t"].values[mask]).astype("timedelta64[s]").astype(float))
    if not np.isfinite(dt) or dt <= 0:
        dt = 2.0
    max_shift = int(max_lag_s / dt)
    best_cc, best_k = -np.inf, 0
    for k in range(-max_shift, max_shift + 1):
        if k < 0:
            cc = np.dot(x[-k:], y[:len(y)+k]) / (len(y)+k)
        elif k > 0:
            cc = np.dot(x[:len(x)-k], y[k:]) / (len(y)-k)
        else:
            cc = np.dot(x, y) / len(y)
        if cc > best_cc:
            best_cc, best_k = cc, k
    return int(round(best_k * dt))

# ---------- UI ----------
st.title("PV Plant vs Reference Panel Fitting Tool")

col1, col2 = st.columns(2)
with col1:
    ref_file = st.file_uploader("Upload Reference Panel CSV", type=["csv"], key="ref")
with col2:
    plant_file = st.file_uploader("Upload Plant Output CSV", type=["csv"], key="plant")

if not (ref_file and plant_file):
    st.stop()

# Read CSVs
ref_df = pd.read_csv(ref_file)
plant_df = pd.read_csv(plant_file)

st.subheader("Column Mapping")
ref_time_col = st.selectbox("Ref: Time column", ref_df.columns)
ref_i_col = st.selectbox("Ref: Isc column", ref_df.columns)
plant_time_col = st.selectbox("Plant: Time column", plant_df.columns)
plant_p_col = st.selectbox("Plant: Power column (W)", plant_df.columns)

st.subheader("Time Alignment Settings")
smooth_s = st.number_input("Rolling median window (s)", min_value=0, value=10, step=2)
resample_s = st.number_input("Uniform resample (s)", min_value=0, value=2, step=1)
max_lag_s = st.number_input("Max lag search (±s)", min_value=0, value=60, step=5)
ref_to_plant_win_s = st.number_input("Ref→Plant window mean ±(s)", min_value=0, value=15, step=5)
join_tol_s = st.number_input("Join tolerance (s)", min_value=0, value=5, step=1)

# ---------- Process ----------
# Prepare raw series
ref = pd.DataFrame({"t": ref_df[ref_time_col].map(_to_datetime),
                    "I": ref_df[ref_i_col].map(_num)}).dropna().sort_values("t")
plant = pd.DataFrame({"t": plant_df[plant_time_col].map(_to_datetime),
                      "P": plant_df[plant_p_col].map(_num)}).dropna().sort_values("t")

# Smoothing
if smooth_s > 0:
    ref = roll_median(ref, "I", smooth_s)
    plant = roll_median(plant, "P", smooth_s)

# Resample
if resample_s > 0:
    ref = resample_uniform(ref, "I", resample_s)
    plant = resample_uniform(plant, "P", resample_s)

# Lag estimation
lag_s = estimate_lag_seconds(ref, plant, "I", "P", max_lag_s=max_lag_s)
if lag_s != 0:
    st.caption(f"Applied lag to reference panel: {lag_s:+d} s")
    ref["t"] = ref["t"] + pd.to_timedelta(lag_s, unit="s")

# Join
if ref_to_plant_win_s > 0:
    ref_at_plant = ref_window_mean_at_plant(ref, plant, "I", win_s=ref_to_plant_win_s)
    joined = plant.merge(ref_at_plant, on="t", how="inner")
else:
    tol = pd.Timedelta(seconds=int(join_tol_s))
    joined = pd.merge_asof(ref, plant, on="t", direction="nearest", tolerance=tol)
    joined = joined.dropna(subset=["P", "I"])

if joined.empty:
    st.error("No matching timestamps found after alignment.")
    st.stop()

# ---------- Quadratic Fit ----------
mask = np.isfinite(joined["I"]) & np.isfinite(joined["P"])
x = joined.loc[mask, "I"].values
y = joined.loc[mask, "P"].values

if len(x) < 3:
    st.error("Not enough valid points to fit.")
    st.stop()

coeffs = np.polyfit(x, y, 2)
poly = np.poly1d(coeffs)
y_pred = poly(x)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

st.subheader("Fit Results")
st.write(f"Quadratic model: `P = {coeffs[0]:.4f}·I² + {coeffs[1]:.4f}·I + {coeffs[2]:.4f}`")
st.write(f"RMSE: {rmse:.2f} W over {len(x)} points")

# ---------- Plot ----------
import altair as alt

chart_data = pd.DataFrame({"I": x, "P": y, "Fit": y_pred})
scatter = alt.Chart(chart_data).mark_circle(size=30, opacity=0.5).encode(
    x=alt.X("I", title="Ref Isc (A)"),
    y=alt.Y("P", title="Plant Power (W)"),
    tooltip=["I", "P", "Fit"]
)
fit_line = alt.Chart(pd.DataFrame({"I": np.linspace(x.min(), x.max(), 200)})).transform_calculate(
    Fit="{} * datum.I * datum.I + {} * datum.I + {}".format(coeffs[0], coeffs[1], coeffs[2])
).mark_line(color="red").encode(
    x="I",
    y="Fit"
)

st.altair_chart(scatter + fit_line, use_container_width=True)
