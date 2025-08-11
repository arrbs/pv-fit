# streamlit_app.py
# PV Fit Calibrator — single-slope + piecewise (two-segment) fit
# Upload Ref(Isc vs time) + Plant(Power vs time), align by time,
# filter sleep/clipping, fit k (zero-intercept) AND piecewise zero-intercept,
# visualize, export joined CSV, and print HA YAML snippets.

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PV Fit Calibrator (Two CSVs → k / piecewise)", layout="wide")

st.title("PV Fit Calibrator (Two CSVs → k / piecewise)")
st.write("""
Upload **two CSV files**:
1) **Reference CSV** with *Isc (A)* vs *time*  
2) **Plant CSV** with *Power (W)* vs *time*

We align timestamps (nearest within tolerance), exclude sleep/clipping,
fit **k** with a through-origin model (**P ≈ k·Isc**), and optionally a **two-segment** through-origin model.
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

st.caption("Valid points must satisfy: P ≥ Pmin, I ≥ Imin, and P ≤ (clip × Pmax). Timestamps are aligned by nearest within the tolerance.")

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

    # ---- single-slope through-origin ----
    sumPI = float((valid["P"] * valid["I"]).sum()) if len(valid) > 0 else 0.0
    sumI2 = float((valid["I"] * valid["I"]).sum()) if len(valid) > 0 else 0.0
    sumP2 = float((valid["P"] * valid["P"]).sum()) if len(valid) > 0 else 0.0
    k_zero = (sumPI / sumI2) if sumI2 > 0 else np.nan
    r2_zero = ((sumPI * sumPI) / (sumP2 * sumI2)) if (sumP2 > 0 and sumI2 > 0) else np.nan

    # ---- OLS with intercept (for reference) ----
    if len(valid) >= 2:
        I = valid["I"].values
        P = valid["P"].values
        I_mean, P_mean = I.mean(), P.mean()
        sxx = ((I - I_mean) ** 2).sum()
        sxy = ((I - I_mean) * (P - P_mean)).sum()
        syy = ((P - P_mean) ** 2).sum()
        k_ols = sxy / sxx if sxx > 0 else np.nan
        a_ols = P_mean - k_ols * I_mean if np.isfinite(k_ols) else np.nan
        r2_ols = (sxy * sxy) / (sxx * syy) if (sxx > 0 and syy > 0) else np.nan
    else:
        k_ols = a_ols = r2_ols = np.nan

    # ---- piecewise through-origin (automated breakpoint) ----
    def piecewise_zero_intercept(I, P, n_grid=40, min_frac=0.10, max_frac=0.90):
        if len(I) < 40:
            return np.nan, np.nan, np.nan, np.nan
        order = np.argsort(I)
        I, P = I[order], P[order]
        n = len(I)
        lo = int(n * min_frac)
        hi = int(n * max_frac)
        idxs = np.linspace(lo, hi, num=min(n_grid, max(hi - lo, 3)), dtype=int)
        best_sse = np.inf
        best = (np.nan, np.nan, np.nan, np.nan)
        sst = ((P - P.mean())**2).sum()
        for j in idxs:
            I1, P1 = I[:j], P[:j]
            I2, P2 = I[j:], P[j:]
            sI2_1 = (I1**2).sum(); sPI_1 = (P1*I1).sum()
            sI2_2 = (I2**2).sum(); sPI_2 = (P2*I2).sum()
            if sI2_1 <= 0 or sI2_2 <= 0:
                continue
            k1 = sPI_1 / sI2_1
            k2 = sPI_2 / sI2_2
            sse = ((P1 - k1*I1)**2).sum() + ((P2 - k2*I2)**2).sum()
            if sse < best_sse:
                r2_pw = 1 - (sse / sst) if sst > 0 else np.nan
                best_sse = sse
                best = (I[j], k1, k2, r2_pw)
        return best

    I_vals = valid["I"].values if len(valid) else np.array([])
    P_vals = valid["P"].values if len(valid) else np.array([])
    I_break, k1_pw, k2_pw, r2_pw = piecewise_zero_intercept(I_vals, P_vals)

    # ---------------- KPIs ----------------
    st.markdown("### 4) Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Joined points", f"{len(joined)}")
    m2.metric("Valid points", f"{len(valid)}")
    ov = f"Start: {t0}\nEnd: {t1}\nDuration: {overlap_min:.1f} min" if t0 is not None else "—"
    m3.write("**Overlap window**"); m3.code(ov)
    guidance = (f"Overlap is ~{overlap_min:.1f} min. Try for ≥30–60 min; 90+ min with mixed sun/cloud is best."
                if overlap_min < 30 else f"Overlap is ~{overlap_min:.1f} min. Looks good.")
    m4.write("**Guidance**"); m4.info(guidance)
    if len(valid) < 50:
        st.warning("Very few valid points (<50). k may be unreliable—collect a longer or more varied window.")

    kcol, ocol, pcol = st.columns(3)
    # single slope
    kcol.write("**k (zero-intercept)**")
    kcol.subheader(f"{k_zero:,.3f} W/A" if np.isfinite(k_zero) else "—")
    kcol.caption(f"R² (zero-intercept): {r2_zero:.3f}" if np.isfinite(r2_zero) else "R²: —")
    # OLS
    ocol.write("**OLS with intercept**")
    ocol.write(f"kₑ: {k_ols:,.3f} W/A" if np.isfinite(k_ols) else "kₑ: —")
    ocol.write(f"a (intercept): {a_ols:,.3f} W" if np.isfinite(a_ols) else "a: —")
    ocol.caption(f"R²: {r2_ols:.3f}" if np.isfinite(r2_ols) else "R²: —")
    # piecewise
    pcol.write("**Piecewise (two-segment, through-origin)**")
    pcol.write(f"I* (break): {I_break:,.3f} A" if np.isfinite(I_break) else "I*: —")
    pcol.write(f"k₁: {k1_pw:,.0f} W/A" if np.isfinite(k1_pw) else "k₁: —")
    pcol.write(f"k₂: {k2_pw:,.0f} W/A" if np.isfinite(k2_pw) else "k₂: —")
    pcol.caption(f"R² (piecewise): {r2_pw:.3f}" if np.isfinite(r2_pw) else "R²: —")

    # ---------- ESPHome & HA snippets ----------
    st.markdown("#### Home Assistant snippet (single k)")
    if np.isfinite(k_zero):
        st.code(
f"""# Template state for 'PV Power Available (expected)'
{{% set isc = states('sensor.ref_pv_probe_8266_ref_pv_isc')|float(0) %}}
{{% set k = {k_zero:.3f} %}}
{{{{ (k * isc) | round(0) }}}}
""", language="jinja2")

    st.markdown("#### Home Assistant snippet (piecewise)")
    if np.isfinite(I_break) and np.isfinite(k1_pw) and np.isfinite(k2_pw):
        st.code(
f"""# Piecewise template (I*={I_break:.3f} A, k1={k1_pw:.0f}, k2={k2_pw:.0f})
{{% set isc = states('sensor.ref_pv_probe_8266_ref_pv_isc')|float(0) %}}
{{% set i_break = {I_break:.6f} %}}
{{% set k1 = {k1_pw:.3f} %}}
{{% set k2 = {k2_pw:.3f} %}}
{{{{ ( (k1 if isc < i_break else k2) * isc ) | round(0) }}}}
""", language="jinja2")

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

    # overlay single k
    if np.isfinite(k_zero) and len(valid) > 0:
        i_min, i_max = float(valid["I"].min()), float(valid["I"].max())
        xs = np.linspace(i_min, i_max, 2)
        ax1.plot(xs, k_zero*xs, linewidth=2, label="fit k")

    # overlay piecewise
    if np.isfinite(I_break) and np.isfinite(k1_pw) and np.isfinite(k2_pw) and len(valid) > 0:
        i_min, i_max = float(valid["I"].min()), float(valid["I"].max())
        xs1 = np.linspace(i_min, I_break, 2)
        xs2 = np.linspace(I_break, i_max, 2)
        ax1.plot(xs1, k1_pw*xs1, linewidth=2, label="fit k₁")
        ax1.plot(xs2, k2_pw*xs2, linewidth=2, label="fit k₂")
        ax1.axvline(I_break, linestyle=":", linewidth=1)

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
        if np.isfinite(I_break):
            ax3.axvline(I_break, linestyle=":", linewidth=1)
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
