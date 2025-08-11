# streamlit_app.py
# PV Fit Calibrator — Quadratic (through-origin): P ≈ a*I + b*I^2
# Load Ref(Isc vs time) + Plant(Power vs time), align by time, filter,
# fit quadratic, visualize, export joined CSV, and print HA snippets.

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PV Fit Calibrator (Quadratic)", layout="wide")

st.title("PV Fit Calibrator — Quadratic (through-origin)")
st.write("""
Upload **two CSV files**:
1) **Reference CSV** with *Isc (A)* vs *time*  
2) **Plant CSV** with *Power (W)* vs *time*

We align timestamps (nearest within tolerance), exclude sleep/clipping,
fit a **quadratic through-origin model**: **P ≈ a·I + b·I²**, and emit an HA snippet.
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
    cols = list(df.columns)
    if not cols:
        return None
    for c in cols:
        if pd.Series([c]).str.contains(r"(time|timestamp|date)", case=False, regex=True).any():
            return c
    return cols[0]

def _guess_current_col(df: pd.DataFrame):
    cols = list(df.columns)
    for c in cols:
        if pd.Series([c]).str.contains(r"(isc|short.*curr|current|amps?|a\b)", case=False, regex=True).any():
            return c
    return cols[1] if len(cols) > 1 else cols[0]

def _guess_power_col(df: pd.DataFrame):
    cols = list(df.columns)
    for c in cols:
        if pd.Series([c]).str.contains(r"(power|watts?|w\b|ac)", case=False, regex=True).any():
            return c
    return cols[1] if len(cols) > 1 else cols[0]

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
    if not ref_df.empty:
        st.caption(f"Detected columns: {list(ref_df.columns)}")

with col_right:
    st.subheader("2) Plant CSV (Power vs Time)")
    plant_file = st.file_uploader("Upload Plant CSV (.csv)", type=["csv"], key="plant")
    plant_df = read_csv(plant_file)
    if not plant_df.empty:
        st.caption(f"Detected columns: {list(plant_df.columns)}")

# ---------------- Column mapping ----------------
if not ref_df.empty:
    c1, c2 = st.columns(2)
    with c1:
        ref_time_col = st.selectbox(
            "Ref: time column",
            ref_df.columns,
            index=max(ref_df.columns.get_loc(_guess_time_col(ref_df)), 0)
        )
    with c2:
        ref_i_col = st.selectbox(
            "Ref: Isc current column (A)",
            ref_df.columns,
            index=max(ref_df.columns.get_loc(_guess_current_col(ref_df)), 0)
        )

if not plant_df.empty:
    c3, c4 = st.columns(2)
    with c3:
        plant_time_col = st.selectbox(
            "Plant: time column",
            plant_df.columns,
            index=max(plant_df.columns.get_loc(_guess_time_col(plant_df)), 0)
        )
    with c4:
        plant_p_col = st.selectbox(
            "Plant: power column (W)",
            plant_df.columns,
            index=max(plant_df.columns.get_loc(_guess_power_col(plant_df)), 0)
        )

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

st.caption("Valid points: P ≥ Pmin, I ≥ Imin, and P ≤ (clip × Pmax). Timestamps aligned by nearest within tolerance.")

# ---------------- Compute ----------------
ready = (not ref_df.empty) and (not plant_df.empty)
if ready:
    ref = pd.DataFrame({
        "t": ref_df[ref_time_col].map(_to_datetime),
        "I": ref_df[ref_i_col].map(_num),
    }).dropna()

    plant = pd.DataFrame({
        "t": plant_df[plant_time_col].map(_to_datetime),
        "P": plant_df[plant_p_col].map(_num),
    }).dropna()

    ref = ref.sort_values("t")
    plant = plant.sort_values("t")
    if "t" in ref and pd.api.types.is_datetime64_any_dtype(ref["t"]):
        ref["t"] = ref["t"].dt.tz_localize(None)
    if "t" in plant and pd.api.types.is_datetime64_any_dtype(plant["t"]):
        plant["t"] = plant["t"].dt.tz_localize(None)

    tol = pd.Timedelta(seconds=int(join_tol_s))
    joined = pd.merge_asof(ref, plant, on="t", direction="nearest", tolerance=tol)
    joined = joined.dropna(subset=["P", "I"]).copy()

    if len(joined) > 0:
        t0, t1 = joined["t"].iloc[0], joined["t"].iloc[-1]
        overlap_min = (t1 - t0).total_seconds() / 60.0
    else:
        t0 = t1 = None
        overlap_min = 0.0

    clip_limit = clip_frac * float(p_max)
    joined["sleep"] = (joined["P"] < float(p_min)) | (joined["I"] < float(i_min))
    joined["clip"]  = (float(p_max) > 0) & (joined["P"] > clip_limit)
    joined["valid"] = ~(joined["sleep"] | joined["clip"])
    valid = joined.loc[joined["valid"]].copy()

    # ---- Quadratic (through-origin): P ≈ a*I + b*I^2 ----
    def quad_fit_through_origin(I, P):
        I = np.asarray(I); P = np.asarray(P)
        if len(I) < 3:
            return np.nan, np.nan, np.nan
        x1 = I
        x2 = I**2
        S11 = float(np.dot(x1, x1))        # sum I^2
        S12 = float(np.dot(x1, x2))        # sum I^3
        S22 = float(np.dot(x2, x2))        # sum I^4
        T1  = float(np.dot(P,  x1))        # sum P*I
        T2  = float(np.dot(P,  x2))        # sum P*I^2
        M = np.array([[S11, S12],[S12, S22]], dtype=float)
        y = np.array([T1, T2], dtype=float)
        try:
            a, b = np.linalg.solve(M, y)   # P ≈ a*I + b*I^2
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        yhat = a*I + b*(I**2)
        sst = float(((P - P.mean())**2).sum())
        sse = float(((P - yhat)**2).sum())
        r2 = 1 - (sse/sst) if sst > 0 else np.nan
        return a, b, r2

    a_q, b_q, r2_quad = quad_fit_through_origin(valid["I"].values, valid["P"].values)

    # ---- Zero-intercept linear for reference ----
    sumPI = float((valid["P"] * valid["I"]).sum()) if len(valid) > 0 else 0.0
    sumI2 = float((valid["I"] * valid["I"]).sum()) if len(valid) > 0 else 0.0
    sumP2 = float((valid["P"] * valid["P"]).sum()) if len(valid) > 0 else 0.0
    k_zero = (sumPI / sumI2) if sumI2 > 0 else np.nan
    r2_zero = ((sumPI * sumPI) / (sumP2 * sumI2)) if (sumP2 > 0 and sumI2 > 0) else np.nan

    # ---------------- KPIs ----------------
    st.markdown("### 4) Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Joined points", f"{len(joined)}")
    m2.metric("Valid points", f"{len(valid)}")
    ov = f"Start: {t0}\nEnd: {t1}\nDuration: {overlap_min:.1f} min" if t0 is not None else "—"
    m3.write("**Overlap window**"); m3.code(ov)
    guidance = (f"Overlap is ~{overlap_min:.1f} min. Try for ≥30–60 min; 90+ min mixed sun/cloud is best."
                if overlap_min < 30 else f"Overlap is ~{overlap_min:.1f} min. Looks good.")
    m4.write("**Guidance**"); m4.info(guidance)
    if len(valid) < 50:
        st.warning("Very few valid points (<50). Quadratic may be unreliable—collect a longer or more varied window.")

    q1, q2, q3 = st.columns(3)
    q1.write("**a (W/A)**"); q1.subheader(f"{a_q:,.3f}" if np.isfinite(a_q) else "—")
    q2.write("**b (W/A²)**"); q2.subheader(f"{b_q:,.3f}" if np.isfinite(b_q) else "—")
    q3.caption(f"R² (quadratic): {r2_quad:.3f}" if np.isfinite(r2_quad) else "R²: —")

    kcol = st.columns(1)[0]
    kcol.caption(f"Linear reference k: {k_zero:,.3f} W/A (R² {r2_zero:.3f})" if np.isfinite(k_zero) else "Linear reference k: —")

    st.markdown("#### Home Assistant snippet (quadratic)")
    if np.isfinite(a_q) and np.isfinite(b_q):
        st.code(
f"""{{% set isc = states('sensor.ref_pv_probe_8266_ref_pv_isc')|float(0) %}}
{{% set a = {a_q:.6f} %}}
{{% set b = {b_q:.6f} %}}
{{% set p = a*isc + b*isc*isc %}}
{{{{ (0 if p < 0 else p) | round(0) }}}}""", language="jinja2")

    st.markdown("---")

    # ---------------- Scatter ----------------
    st.subheader("Scatter: Plant Power (W) vs Ref Isc (A)")
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.0))
    if len(valid) > 0:
        ax1.scatter(valid["I"], valid["P"], s=12, label="valid")
    if len(joined) > 0 and len(valid) < len(joined):
        ax1.scatter(joined.loc[~joined["valid"], "I"], joined.loc[~joined["valid"], "P"], s=12, marker="x", label="sleep/clip")
    ax1.axhline(clip_limit, linestyle="--", linewidth=1)
    ax1.set_xlabel("Isc (A)"); ax1.set_ylabel("Power (W)")
    # overlay linear & quadratic fits
    if np.isfinite(k_zero) and len(valid) > 0:
        xs_lin = np.linspace(float(valid["I"].min()), float(valid["I"].max()), 2)
        ax1.plot(xs_lin, k_zero*xs_lin, linewidth=2, label="linear fit")
    if np.isfinite(a_q) and np.isfinite(b_q) and len(valid) > 0:
        xi = np.linspace(float(valid["I"].min()), float(valid["I"].max()), 200)
        ax1.plot(xi, a_q*xi + b_q*(xi**2), linewidth=2, label="quadratic fit")
    ax1.legend()
    st.pyplot(fig1)

    # ---------------- Instantaneous slope profile ----------------
    st.subheader("Instantaneous slope profile: median(P/I) vs Isc")
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.0))
    if len(valid) > 0:
        df = valid[["I","P"]].copy()
        df = df[(df["I"]>0) & (df["P"]>0)]
        df["k_inst"] = df["P"] / df["I"]
        bins = 24
        df["bin"] = pd.qcut(df["I"], q=bins, duplicates="drop")
        prof = df.groupby("bin").agg(I_mid=("I","median"), k_med=("k_inst","median")).dropna()
        ax3.plot(prof["I_mid"], prof["k_med"], marker="o", linewidth=1)
        ax3.set_xlabel("Isc (A)")
        ax3.set_ylabel("Median k = P/I (W/A)")
    st.pyplot(fig3)

    # ---------------- Time series (two axes) ----------------
    st.subheader("Time Series (joined order)")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
    if not joined.empty:
        idx = np.arange(len(joined))
        ax2.plot(idx, joined["P"], label="Power (W)")
        ax2.set_xlabel("Joined sample index"); ax2.set_ylabel("Power (W)")
        ax2.grid(True, linewidth=0.3, alpha=0.5)
        ax2b = ax2.twinx()
        ax2b.plot(idx, joined["I"], label="Isc (A)")
        ax2b.set_ylabel("Isc (A)")
        ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")
    st.pyplot(fig2)

    # ---------------- Export cleaned/joined CSV ----------------
    st.markdown("### Export cleaned/joined data")
    if not joined.empty:
        out = joined[["t", "I", "P", "valid", "sleep", "clip"]].copy()
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download joined dataset (CSV)",
            data=csv_bytes,
            file_name="pv_fit_joined.csv",
            mime="text/csv",
        )

else:
    st.info("Upload both CSVs and map their columns to proceed.")
