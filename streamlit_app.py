# streamlit_app.py
# PV Fit Calibrator — Quadratic (through-origin): P ≈ a*I + b*I^2

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="PV Fit Calibrator (Quadratic)", layout="wide")

st.title("PV Fit Calibrator — Quadratic (through-origin)")
st.write("""
Upload **two CSV files**:
1) **Reference CSV** with *Isc (A)* vs *time*  
2) **Plant CSV** with *Power (W)* vs *time*

We align timestamps, filter junk, fit a quadratic model, and provide a Home Assistant snippet.
Now includes an **interactive tool** to remove bad points directly from the scatter plot.
""")

# ---------------- Helpers ----------------
def _num(v):
    try:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        s = str(v).strip().replace(",", ".")
        return float(s)
    except Exception:
        return np.nan

def _to_datetime(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def _guess_time_col(df: pd.DataFrame):
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower():
            return c
    return df.columns[0]

def _guess_current_col(df: pd.DataFrame):
    for c in df.columns:
        if "isc" in c.lower() or "current" in c.lower() or "amps" in c.lower():
            return c
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

def _guess_power_col(df: pd.DataFrame):
    for c in df.columns:
        if "power" in c.lower() or "watt" in c.lower():
            return c
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

@st.cache_data(show_spinner=False)
def read_csv(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(uploaded.getvalue()))

# ---------------- Uploaders ----------------
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("1) Reference CSV (Isc vs Time)")
    ref_file = st.file_uploader("Upload Reference CSV (.csv)", type=["csv"], key="ref")
    ref_df = read_csv(ref_file)

with col_right:
    st.subheader("2) Plant CSV (Power vs Time)")
    plant_file = st.file_uploader("Upload Plant CSV (.csv)", type=["csv"], key="plant")
    plant_df = read_csv(plant_file)

# ---------------- Column mapping ----------------
if not ref_df.empty:
    c1, c2 = st.columns(2)
    with c1:
        ref_time_col = st.selectbox("Ref: time column", ref_df.columns,
                                    index=max(ref_df.columns.get_loc(_guess_time_col(ref_df)), 0))
    with c2:
        ref_i_col = st.selectbox("Ref: Isc current column (A)", ref_df.columns,
                                 index=max(ref_df.columns.get_loc(_guess_current_col(ref_df)), 0))

if not plant_df.empty:
    c3, c4 = st.columns(2)
    with c3:
        plant_time_col = st.selectbox("Plant: time column", plant_df.columns,
                                      index=max(plant_df.columns.get_loc(_guess_time_col(plant_df)), 0))
    with c4:
        plant_p_col = st.selectbox("Plant: power column (W)", plant_df.columns,
                                   index=max(plant_df.columns.get_loc(_guess_power_col(plant_df)), 0))

st.markdown("---")
st.subheader("3) Filters & Alignment")

cA, cB, cC, cD, cE = st.columns(5)
with cA:
    p_min = st.number_input("Pmin (sleep, W)", min_value=0, value=50, step=10)
with cB:
    i_min = st.number_input("Imin (sleep, A)", min_value=0.0, value=0.02, step=0.01, format="%.3f")
with cC:
    p_max = st.number_input("Pmax (PV/MPPT cap, W)", min_value=0, value=6000, step=100)
with cD:
    clip_frac = st.number_input("Clip fraction ×Pmax", min_value=0.0, max_value=1.0, value=0.9, step=0.05, format="%.2f")
with cE:
    join_tol_s = st.number_input("Join tolerance (seconds)", min_value=0, value=60, step=5)

bias_watts = st.number_input("Bias margin (subtract W, then refit)", value=0.0, step=10.0)

# ---------------- Compute ----------------
ready = (not ref_df.empty) and (not plant_df.empty)
if ready:
    ref = pd.DataFrame({"t": ref_df[ref_time_col].map(_to_datetime),
                        "I": ref_df[ref_i_col].map(_num)}).dropna()
    plant = pd.DataFrame({"t": plant_df[plant_time_col].map(_to_datetime),
                          "P": plant_df[plant_p_col].map(_num)}).dropna()

    ref, plant = ref.sort_values("t"), plant.sort_values("t")
    ref["t"], plant["t"] = ref["t"].dt.tz_localize(None), plant["t"].dt.tz_localize(None)

    # Join nearest in time
    tol = pd.Timedelta(seconds=int(join_tol_s))
    joined = pd.merge_asof(ref, plant, on="t", direction="nearest", tolerance=tol).dropna()
    joined["valid"] = (joined["P"] >= p_min) & (joined["I"] >= i_min) & (joined["P"] <= clip_frac * p_max)

    # Keep only valid
    valid = joined.loc[joined["valid"]].copy()

    # ---------------- Interactive bad-point removal ----------------
    if "removed_indices" not in st.session_state:
        st.session_state.removed_indices = set()

    valid_display = valid.drop(index=list(st.session_state.removed_indices), errors="ignore")

    st.subheader("4) Interactive Scatter — Remove Bad Points")
    fig_plotly = px.scatter(valid_display, x="I", y="P",
                            title="Lasso/box select bad points, then click 'Remove Selected'",
                            labels={"I": "Isc (A)", "P": "Power (W)"})
    fig_plotly.update_traces(marker=dict(size=6))
    fig_plotly.update_layout(dragmode="lasso", height=500)

    sel_data = plotly_events(fig_plotly, select_event=True)

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("Remove Selected Points"):
            if sel_data:
                indices_to_remove = {p["pointIndex"] for p in sel_data}
                orig_indices = valid_display.iloc[list(indices_to_remove)].index
                st.session_state.removed_indices.update(orig_indices)
                st.success(f"Removed {len(orig_indices)} points.")
            else:
                st.warning("No points selected.")
    with c_btn2:
        if st.button("Reset Removed Points"):
            st.session_state.removed_indices.clear()
            st.info("All points restored.")

    # Replace valid with cleaned
    valid = valid_display.copy()

    # ---------------- Fit Quadratic ----------------
    def quad_fit(I, P):
        I, P = np.asarray(I), np.asarray(P)
        if len(I) < 3:
            return np.nan, np.nan, np.nan
        x1, x2 = I, I**2
        M = np.array([[np.dot(x1, x1), np.dot(x1, x2)],
                      [np.dot(x1, x2), np.dot(x2, x2)]], dtype=float)
        y = np.array([np.dot(P, x1), np.dot(P, x2)], dtype=float)
        try:
            a, b = np.linalg.solve(M, y)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        yhat = a*I + b*(I**2)
        sst = ((P - P.mean())**2).sum()
        sse = ((P - yhat)**2).sum()
        return a, b, 1 - (sse/sst) if sst > 0 else np.nan

    a_q, b_q, r2_q = quad_fit(valid["I"], valid["P"])
    P_target = np.maximum(valid["P"] - bias_watts, 0.0)
    a_b, b_b, r2_b = quad_fit(valid["I"], P_target)

    st.markdown("### Results")
    st.write(f"Unbiased: a = {a_q:.3f}, b = {b_q:.3f}, R² = {r2_q:.3f}")
    st.write(f"Biased: a = {a_b:.3f}, b = {b_b:.3f}, R² (target) = {r2_b:.3f}")

    st.markdown("#### Home Assistant Snippet (biased coefficients)")
    st.code(
f"""{{% set isc = states('sensor.ref_pv_probe_8266_ref_pv_isc')|float(0) %}}
{{% set a = {a_b:.6f} %}}
{{% set b = {b_b:.6f} %}}
{{% set p = a*isc + b*isc*isc %}}
{{{{ (0 if p < 0 else p) | round(0) }}}}""", language="jinja2")

    # ---------------- Export ----------------
    out = joined[["t", "I", "P", "valid"]].copy()
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset (CSV)", csv_bytes, "pv_fit_joined.csv", "text/csv")

else:
    st.info("Upload both CSVs to proceed.")
