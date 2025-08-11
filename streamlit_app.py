# streamlit_app.py
# PV Fit Calibrator (Two CSVs → k for ESPHome)
# Upload Reference(IsC vs time) + Plant(Power vs time), align by time,
# filter sleep/clipping, fit k (zero-intercept), show plots, export joined CSV,
# and emit an ESPHome snippet.

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Page config ----------------
st.set_page_config(page_title="PV Fit Calibrator (Two CSVs → k for ESPHome)", layout="wide")

st.title("PV Fit Calibrator (Two CSVs → k for ESPHome)")
st.write("""
Upload **two CSV files**:
1) **Reference CSV** with *Isc (A)* vs *time*  
2) **Plant CSV** with *Power (W)* vs *time*

The tool aligns timestamps (nearest within tolerance), excludes sleep/clipping, fits
**k** using zero-intercept least squares (**P ≈ k · Isc**), and emits an ESPHome snippet.
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
    # fallback: first column
    return cols[0]

def _guess_current_col(df: pd.DataFrame):
    cols = list(df.columns)
    for c in cols:
        if pd.Series([c]).str.contains(r"(isc|short.*curr|current|amps?|a\b)", case=False, regex=True).any():
            return c
    # fallback: second column if exists
    return cols[1] if len(cols) > 1 else cols[0]

def _guess_power_col(df: pd.DataFrame):
    cols = list(df.columns)
    for c in cols:
        if pd.Series([c]).str.contains(r"(power|watts?|w\b|ac)", case=False, regex=True).any():
            return c
    return cols[1] if len(cols) > 1 else cols[0]

@st.cache_data(show_spinner=False)
def read_csv(uploaded) -> pd.DataFrame:
    """Cache by bytes to avoid hashing UploadedFile objects directly."""
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
    # Build cleaned frames
    ref = pd.DataFrame({
        "t": ref_df[ref_time_col].map(_to_datetime),
        "I": ref_df[ref_i_col].map(_num),
    }).dropna()

    plant = pd.DataFrame({
        "t": plant_df[plant_time_col].map(_to_datetime),
        "P": plant_df[plant_p_col].map(_num),
    }).dropna()

    # Sort and strip tz to avoid merge_asof headaches
    ref = ref.sort_values("t")
    plant = plant.sort_values("t")
    if "t" in ref and pd.api.types.is_datetime64_any_dtype(ref["t"]):
        ref["t"] = ref["t"].dt.tz_localize(None)
    if "t" in plant and pd.api.types.is_datetime64_any_dtype(plant["t"]):
        plant["t"] = plant["t"].dt.tz_localize(None)

    # Merge nearest by time within tolerance
    tol = pd.Timedelta(seconds=int(join_tol_s))
    joined = pd.merge_asof(ref, plant, on="t", direction="nearest", tolerance=tol)

    # Drop NA P/I after join
    joined = joined.dropna(subset=["P", "I"]).copy()

    # Overlap stats
    if len(joined) > 0:
        t0, t1 = joined["t"].iloc[0], joined["t"].iloc[-1]
        overlap_min = (t1 - t0).total_seconds() / 60.0
    else:
        t0 = t1 = None
        overlap_min = 0.0

    # Valid masks
    clip_limit = clip_frac * float(p_max)
    joined["sleep"] = (joined["P"] < float(p_min)) | (joined["I"] < float(i_min))
    joined["clip"]  = (float(p_max) > 0) & (joined["P"] > clip_limit)
    joined["valid"] = ~(joined["sleep"] | joined["clip"])
    valid = joined.loc[joined["valid"]].copy()

    # Fit zero-intercept: k = sum(P*I)/sum(I^2); R²(through-origin)
    sumPI = float((valid["P"] * valid["I"]).sum()) if len(valid) > 0 else 0.0
    sumI2 = float((valid["I"] * valid["I"]).sum()) if len(valid) > 0 else 0.0
    sumP2 = float((valid["P"] * valid["P"]).sum()) if len(valid) > 0 else 0.0
    k_zero = (sumPI / sumI2) if sumI2 > 0 else np.nan
    r2_zero = ((sumPI * sumPI) / (sumP2 * sumI2)) if (sumP2 > 0 and sumI2 > 0) else np.nan

    # OLS with intercept
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

    # ---------------- KPIs ----------------
    st.markdown("### 4) Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Joined points", f"{len(joined)}")
    m2.metric("Valid points", f"{len(valid)}")

    ov = f"Start: {t0}\nEnd: {t1}\nDuration: {overlap_min:.1f} min" if t0 is not None else "—"
    m3.write("**Overlap window**")
    m3.code(ov)

    guidance = (
        f"Overlap is ~{overlap_min:.1f} min. Try for ≥30–60 min; 90+ min with mixed sun/cloud is best."
        if overlap_min < 30 else
        f"Overlap is ~{overlap_min:.1f} min. Looks good."
    )
    m4.write("**Guidance**")
    m4.info(guidance)

    if len(valid) < 50:
        st.warning("Very few valid points (<50). k may be unreliable—collect a longer or more varied window.")

    kcol, ocol, snip = st.columns(3)
    kcol.write("**k (zero-intercept)**")
    kcol.subheader(f"{k_zero:,.3f} W/A" if np.isfinite(k_zero) else "—")
    kcol.caption(f"R² (zero-intercept): {r2_zero:.3f}" if np.isfinite(r2_zero) else "R²: —")

    ocol.write("**OLS with intercept**")
    ocol.write(f"kₑ: {k_ols:,.3f} W/A" if np.isfinite(k_ols) else "kₑ: —")
    ocol.write(f"a (intercept): {a_ols:,.3f} W" if np.isfinite(a_ols) else "a: —")
    ocol.caption(f"R²: {r2_ols:.3f}" if np.isfinite(r2_ols) else "R²: —")

    # ESPHome snippet (based on zero-intercept fit)
    if np.isfinite(k_zero):
        snippet = f'''  - platform: template
    name: "PV Power Available (est)"
    unit_of_measurement: "W"
    update_interval: 2s
    lambda: |-
      const float k = {k_zero:.3f};  // W per A (fitted)
      return k * id(ref_pv_isc).state;'''
        snip.write("**ESPHome snippet**")
        snip.code(snippet, language="yaml")

    st.markdown("---")

    # ---------------- Scatter plot ----------------
    st.subheader("Scatter: Plant Power (W) vs Ref Isc (A)")
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.0))
    if len(valid) > 0:
        ax1.scatter(valid["I"], valid["P"], s=12, label="valid")
    if len(joined) > 0 and len(valid) < len(joined):
        ax1.scatter(joined.loc[~joined["valid"], "I"], joined.loc[~joined["valid"], "P"], s=12, marker="x", label="sleep/clip")
    ax1.axhline(clip_limit, linestyle="--", linewidth=1)
    ax1.set_xlabel("Isc (A)")
    ax1.set_ylabel("Power (W)")
    ax1.legend()
    st.pyplot(fig1)

    # ---------------- Time series plot (two axes) ----------------
    st.subheader("Time Series (joined order)")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
    if not joined.empty:
        idx = np.arange(len(joined))
        p_line, = ax2.plot(idx, joined["P"], label="Power (W)")
        ax2.set_xlabel("Joined sample index")
        ax2.set_ylabel("Power (W)")
        ax2.grid(True, linewidth=0.3, alpha=0.5)

        ax2b = ax2.twinx()
        i_line, = ax2b.plot(idx, joined["I"], label="Isc (A)")
        ax2b.set_ylabel("Isc (A)")

        # Two separate legends (keep it simple)
        ax2.legend(loc="upper left")
        ax2b.legend(loc="upper right")
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
