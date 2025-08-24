# streamlit_app.py
# PV Fit Calibrator — Quadratic (through-origin): P ≈ a*I + b*I^2
# Load Ref(Isc vs time) + Plant(Power vs time), align by time, filter,
# fit quadratic, visualize, export joined CSV, and print HA snippets.

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events

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

with st.expander("Advanced settings"):
    bias_watts = st.number_input("Bias margin (subtract W from targets & refit)", value=0.0, step=10.0, help="Refits to P' = max(P - bias, 0). Produces new a,b.")
    st.caption("Bias reduces predictions on purpose; expect lower R² vs original P.")

st.caption("Valid points: P ≥ Pmin, I ≥ Imin, P ≤ (clip × Pmax), and not manually excluded. Timestamps aligned by nearest within tolerance.")

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
    if pd.api.types.is_datetime64_any_dtype(ref["t"]):
        ref["t"] = ref["t"].dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(plant["t"]):
        plant["t"] = plant["t"].dt.tz_localize(None)

    # ---- Quadratic fit helper ----
    def quad_fit_through_origin(I, P):
        I = np.asarray(I); P = np.asarray(P)
        if len(I) < 3:
            return np.nan, np.nan, np.nan
        x1 = I
        x2 = I**2
        S11 = float(np.dot(x1, x1))
        S12 = float(np.dot(x1, x2))
        S22 = float(np.dot(x2, x2))
        T1  = float(np.dot(P,  x1))
        T2  = float(np.dot(P,  x2))
        M = np.array([[S11, S12],[S12, S22]], dtype=float)
        y = np.array([T1, T2], dtype=float)
        try:
            a, b = np.linalg.solve(M, y)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        yhat = a*I + b*(I**2)
        sst = float(((P - P.mean())**2).sum())
        sse = float(((P - yhat)**2).sum())
        r2 = 1 - (sse/sst) if sst > 0 else np.nan
        return a, b, r2

    def _spearman_rho(x, y):
        # simple Spearman ρ via ranks + Pearson
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        if xr.size < 3: return np.nan
        xm = xr - xr.mean(); ym = yr - yr.mean()
        denom = np.sqrt((xm*xm).sum() * (ym*ym).sum())
        return float((xm*ym).sum()/denom) if denom > 0 else np.nan

    # ---- Lag compensation ----
    st.markdown("### Lag compensation")
    cLag1, cLag2, cLag3 = st.columns(3)
    with cLag1:
        auto_lag = st.checkbox("Auto-detect lag", value=True)
    with cLag2:
        max_lag_s = st.number_input("Max lag search (seconds)", min_value=0, value=120, step=5)
    with cLag3:
        manual_lag_s = st.number_input("Manual lag (seconds, + = plant delayed)", value=0, step=1)
    metric = st.selectbox("Lag optimize metric", ["R² (midrange quadratic)", "Spearman ρ (midrange)"])

    best_lag = manual_lag_s
    if auto_lag:
        lags = np.arange(-max_lag_s, max_lag_s + 1, 5)  # 5s step
        best_score = -np.inf
        for lag in lags:
            shifted = plant.copy()
            shifted["t"] = shifted["t"] + pd.to_timedelta(lag, unit="s")
            temp = pd.merge_asof(
                ref, shifted, on="t", direction="nearest",
                tolerance=pd.Timedelta(seconds=int(join_tol_s))
            ).dropna(subset=["P", "I"])

            # Hard filters (avoid sleep and clipping)
            temp = temp[
                (temp["P"] >= p_min) &
                (temp["I"] >= i_min) &
                (temp["P"] <= clip_frac * p_max)
            ]
            if len(temp) <= 3:
                continue

            # Midrange window to avoid extremes
            q20, q80 = temp["I"].quantile(0.20), temp["I"].quantile(0.80)
            mid = temp[temp["I"].between(q20, q80)]
            if len(mid) <= 3:
                mid = temp

            if metric.startswith("R²"):
                _, _, r2_mid = quad_fit_through_origin(mid["I"], mid["P"])
                score = r2_mid if np.isfinite(r2_mid) else -np.inf
            else:
                rho = _spearman_rho(mid["I"], mid["P"])
                score = rho if np.isfinite(rho) else -np.inf

            if score > best_score:
                best_score = score
                best_lag = lag

        label = "R²" if metric.startswith("R²") else "Spearman ρ"
        st.caption(f"Auto-detected lag: {best_lag} s ({label} {best_score:.3f})")

    # Apply lag before join
    plant["t"] = plant["t"] + pd.to_timedelta(best_lag, unit="s")

    # ---- Join data ----
    tol = pd.Timedelta(seconds=int(join_tol_s))
    joined = pd.merge_asof(ref, plant, on="t", direction="nearest", tolerance=tol)
    joined = joined.dropna(subset=["P", "I"]).copy()

    # ---- Mark valid points ----
    clip_limit = clip_frac * float(p_max)
    joined["sleep"] = (joined["P"] < float(p_min)) | (joined["I"] < float(i_min))
    joined["clip"]  = (float(p_max) > 0) & (joined["P"] > clip_limit)
    joined["auto_valid"] = ~(joined["sleep"] | joined["clip"])
    
    # Initialize session state for manual exclusions
    if "manually_excluded_indices" not in st.session_state:
        st.session_state.manually_excluded_indices = set()
    
    # Create a manual exclusion mask
    joined["manually_excluded"] = joined.index.isin(st.session_state.manually_excluded_indices)
    joined["valid"] = joined["auto_valid"] & ~joined["manually_excluded"]
    valid = joined.loc[joined["valid"]].copy()

    # ---- Fit: un-biased vs biased ----
    # Un-biased fit against actual P
    a_q, b_q, r2_quad = quad_fit_through_origin(valid["I"].values, valid["P"].values)

    # Biased target: push curve down, then refit => new coefficients a_b, b_b
    P_target = np.maximum(valid["P"].values - float(bias_watts), 0.0)
    a_b, b_b, r2_b_target = quad_fit_through_origin(valid["I"].values, P_target)

    # Evaluate biased model against original P, too (expect lower R² by design)
    if np.isfinite(a_b) and np.isfinite(b_b) and len(valid) > 0:
        P_hat_b_on_P = a_b*valid["I"].values + b_b*(valid["I"].values**2)
        sst = float(((valid["P"].values - valid["P"].values.mean())**2).sum())
        sse = float(((valid["P"].values - P_hat_b_on_P)**2).sum())
        r2_b_on_P = 1 - (sse/sst) if sst > 0 else np.nan
    else:
        r2_b_on_P = np.nan

    # Zero-intercept linear reference
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
    num_manually_excluded = len(st.session_state.manually_excluded_indices)
    m3.metric("Manually excluded", f"{num_manually_excluded}")
    if len(joined) > 0:
        t0, t1 = joined["t"].iloc[0], joined["t"].iloc[-1]
        overlap_min = (t1 - t0).total_seconds() / 60.0
    else:
        t0 = t1 = None
        overlap_min = 0.0
    ov = f"Start: {t0}\nEnd: {t1}\nDuration: {overlap_min:.1f} min" if t0 is not None else "—"
    m4.write("**Overlap window**"); m4.code(ov)
    
    # Add guidance in a separate row
    st.markdown("---")
    guidance = (f"Overlap is ~{overlap_min:.1f} min. Try for ≥30–60 min; 90+ min mixed sun/cloud is best."
                if overlap_min < 30 else f"Overlap is ~{overlap_min:.1f} min. Looks good.")
    st.info(guidance)
    if len(valid) < 50:
        st.warning("Very few valid points (<50). Quadratic may be unreliable—collect a longer or more varied window.")

    # Coefficients + metrics
    cfit1, cfit2 = st.columns(2)
    with cfit1:
        st.write("**Un-biased quadratic (P ≈ a·I + b·I²)**")
        st.write(f"a = {a_q:,.3f} W/A")
        st.write(f"b = {b_q:,.3f} W/A²")
        st.caption(f"R² on P: {r2_quad:.3f}" if np.isfinite(r2_quad) else "R² on P: —")
    with cfit2:
        st.write("**Biased quadratic (fit to P' = max(P − bias, 0))**")
        st.write(f"a_b = {a_b:,.3f} W/A" if np.isfinite(a_b) else "a_b = —")
        st.write(f"b_b = {b_b:,.3f} W/A²" if np.isfinite(b_b) else "b_b = —")
        st.caption(
            (f"R² on target P': {r2_b_target:.3f}; R² on original P: {r2_b_on_P:.3f}")
            if np.isfinite(r2_b_target) else "R²: —"
        )

    st.markdown("#### Home Assistant snippet (biased coefficients)")
    if np.isfinite(a_b) and np.isfinite(b_b):
        st.code(
f"""{{% set isc = states('sensor.ref_pv_probe_8266_ref_pv_isc')|float(0) %}}
{{% set a = {a_b:.6f} %}}
{{% set b = {b_b:.6f} %}}
{{% set p = a*isc + b*isc*isc %}}
{{{{ (0 if p < 0 else p) | round(0) }}}}""", language="jinja2")
    else:
        st.caption("Not enough valid data for biased coefficients; showing un-biased snippet instead.")
        if np.isfinite(a_q) and np.isfinite(b_q):
            st.code(
f"""{{% set isc = states('sensor.ref_pv_probe_8266_ref_pv_isc')|float(0) %}}
{{% set a = {a_q:.6f} %}}
{{% set b = {b_q:.6f} %}}
{{% set p = a*isc + b*isc*isc %}}
{{{{ (0 if p < 0 else p) | round(0) }}}}""", language="jinja2")

    st.markdown("---")

    # ---------------- Interactive Scatter ----------------
    st.subheader("Interactive Scatter: Plant Power (W) vs Ref Isc (A)")
    st.caption("💡 Use the box select or lasso select tools to select points for exclusion. Selected points will be removed from the fit.")
    
    # UI controls for manual exclusions
    col_clear, col_info = st.columns([1, 3])
    with col_clear:
        if st.button("Clear Manual Exclusions", type="secondary"):
            st.session_state.manually_excluded_indices = set()
            st.rerun()
    with col_info:
        num_excluded = len(st.session_state.manually_excluded_indices)
        if num_excluded > 0:
            st.info(f"📋 {num_excluded} points manually excluded from fit")

    # Create interactive scatter plot with Plotly
    fig = go.Figure()
    
    # Add valid points
    if len(valid) > 0:
        fig.add_trace(go.Scatter(
            x=valid["I"], 
            y=valid["P"], 
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.7),
            name='Valid points',
            customdata=valid.index,
            hovertemplate='I: %{x:.3f} A<br>P: %{y:.1f} W<br>Index: %{customdata}<extra></extra>'
        ))
    
    # Add automatically excluded points (sleep/clip)  
    auto_excluded = joined.loc[~joined["auto_valid"] & ~joined["manually_excluded"]]
    if len(auto_excluded) > 0:
        fig.add_trace(go.Scatter(
            x=auto_excluded["I"], 
            y=auto_excluded["P"], 
            mode='markers',
            marker=dict(size=6, color='red', symbol='x', opacity=0.7),
            name='Sleep/clip (auto excluded)',
            customdata=auto_excluded.index,
            hovertemplate='I: %{x:.3f} A<br>P: %{y:.1f} W<br>Index: %{customdata}<br>Reason: Sleep/Clip<extra></extra>'
        ))
    
    # Add manually excluded points
    manual_excluded = joined.loc[joined["manually_excluded"]]
    if len(manual_excluded) > 0:
        fig.add_trace(go.Scatter(
            x=manual_excluded["I"], 
            y=manual_excluded["P"], 
            mode='markers',
            marker=dict(size=8, color='orange', symbol='x', opacity=0.8),
            name='Manually excluded',
            customdata=manual_excluded.index,
            hovertemplate='I: %{x:.3f} A<br>P: %{y:.1f} W<br>Index: %{customdata}<br>Reason: Manual exclusion<extra></extra>'
        ))
    
    # Add clip limit line
    if len(joined) > 0:
        i_min_plot = float(joined["I"].min())
        i_max_plot = float(joined["I"].max())
        fig.add_trace(go.Scatter(
            x=[i_min_plot, i_max_plot], 
            y=[clip_limit, clip_limit],
            mode='lines',
            line=dict(dash='dash', color='gray', width=1),
            name=f'Clip limit ({clip_limit:.0f} W)',
            showlegend=True
        ))
    
    # Add fit curves
    if np.isfinite(a_q) and np.isfinite(b_q) and len(valid) > 0:
        xi = np.linspace(float(valid["I"].min()), float(valid["I"].max()), 200)
        yi_quad = a_q*xi + b_q*(xi**2)
        fig.add_trace(go.Scatter(
            x=xi, y=yi_quad,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Quadratic fit'
        ))
    
    if np.isfinite(a_b) and np.isfinite(b_b) and len(valid) > 0:
        xi = np.linspace(float(valid["I"].min()), float(valid["I"].max()), 200)
        yi_biased = a_b*xi + b_b*(xi**2)
        fig.add_trace(go.Scatter(
            x=xi, y=yi_biased,
            mode='lines',
            line=dict(color='blue', width=3, dash='dash'),
            name=f'Biased fit (targets −{bias_watts:.0f} W)'
        ))
    
    # Configure layout
    fig.update_layout(
        xaxis_title="Isc (A)",
        yaxis_title="Power (W)", 
        height=500,
        dragmode='select',  # Enable box select by default
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Handle selection events
    selected_points = plotly_events(
        fig, 
        click_event=False, 
        hover_event=False,
        select_event=True,
        override_height=500,
        key="scatter_plot"
    )
    
    # Process selection to exclude points
    if selected_points:
        # Get indices of selected points
        selected_indices = set()
        for point in selected_points:
            if 'customdata' in point and point['customdata'] is not None:
                selected_indices.add(point['customdata'])
        
        if selected_indices:
            # Only include points that are currently valid (not already excluded)
            valid_selected = selected_indices & set(joined.loc[joined["auto_valid"]].index)
            if valid_selected:
                st.session_state.manually_excluded_indices.update(valid_selected)
                st.rerun()

    # ---------------- Instantaneous slope profile ----------------
    st.subheader("Instantaneous slope profile: median(P/I) vs Isc")
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.0))
    if len(valid) > 0:
        df = valid[["I","P"]].copy()
        df = df[(df["I"] > 0) & (df["P"] > 0)]
        df["k_inst"] = df["P"] / df["I"]
        bins = 24
        df["bin"] = pd.qcut(df["I"], q=bins, duplicates="drop")
        prof = df.groupby("bin").agg(I_mid=("I","median"), k_med=("k_inst","median")).dropna()
        ax3.plot(prof["I_mid"], prof["k_med"], marker="o", linewidth=1)
        ax3.set_xlabel("Isc (A)")
        ax3.set_ylabel("Median k = P/I (W/A)")
    st.pyplot(fig3)

    # ---------------- Time series (joined order) ----------------
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
        out = joined[["t", "I", "P", "valid", "sleep", "clip", "manually_excluded"]].copy()
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download joined dataset (CSV)",
            data=csv_bytes,
            file_name="pv_fit_joined.csv",
            mime="text/csv",
        )

else:
    st.info("Upload both CSVs and map their columns to proceed.")
