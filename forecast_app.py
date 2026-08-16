import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from breeze_connect import BreezeConnect

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecast_pipeline import (
    generate_mock_data,
    fetch_icici_data,
    clean_time_series,
    run_chronos_forecast,
    plot_forecast,
    calculate_indicators,
    backtest_forecast,
    calculate_directional_accuracy,
    generate_signals,
    build_gbdt_features,
    train_gbdt_models,
    forecast_gbdt,
    ensemble_forecast,
    get_nse_trading_days,
    StationaryTransformer,
    HybridTimeSeriesEnsemble,
    ARIMAForecaster,
    _load_credentials,
    TARGET_SYMBOL,
    FORECAST_HORIZON,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECURITY_MASTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SecurityMaster (1)",
    "NSEScripMaster.txt",
)
FO_SECURITY_MASTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SecurityMaster (1)",
    "FONSEScripMaster.txt",
)

CHRONOS_MODELS = {
    "Chronos-T5-Tiny": "amazon/chronos-t5-tiny",
    "Chronos-T5-Small": "amazon/chronos-t5-small",
    "Chronos-T5-Base": "amazon/chronos-t5-base",
    "Chronos-Bolt-Base": "amazon/chronos-bolt-base",
}

ENSEMBLE_METHODS = {
    "Blend (Chronos + GBDT)": "blend",
    "Chronos Only": "chronos_only",
    "GBDT Only": "gbdt_only",
    "Hybrid (Chronos + ARIMA + GBDT)": "hybrid",
}


@st.cache_resource
def load_security_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    column_map = {c: c.strip().strip('"') for c in df.columns}
    df.rename(columns=column_map, inplace=True)

    symbol_col = "ShortName"
    name_col = "CompanyName"

    df["__search__"] = (
        df[symbol_col].str.upper().str.strip()
        + " "
        + df[name_col].str.upper().str.strip()
        + " "
        + df["ExchangeCode"].str.upper().str.strip()
    )
    df["__display__"] = df[symbol_col].str.upper().str.strip()
    return df


@st.cache_resource
def load_fo_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    column_map = {c: c.strip().strip('"') for c in df.columns}
    df.rename(columns=column_map, inplace=True)

    def parse_expiry(val):
        try:
            return pd.to_datetime(val, format="%d-%b-%Y")
        except Exception:
            try:
                return pd.to_datetime(val)
            except Exception:
                return pd.NaT

    df["__expiry_dt__"] = df["ExpiryDate"].apply(parse_expiry)
    df = df[df["__expiry_dt__"].notna()].copy()
    df = df[df["__expiry_dt__"] <= (pd.Timestamp.now() + pd.Timedelta(days=365))].copy()
    df = df[df["__expiry_dt__"] >= (pd.Timestamp.now() - pd.Timedelta(days=90))].copy()

    df["__search__"] = (
        df["ShortName"].str.upper().str.strip()
        + " "
        + df["CompanyName"].str.upper().str.strip()
    )
    df["__display__"] = df["ShortName"].str.upper().str.strip()
    return df


@st.cache_data
def get_data(symbol: str, days: int = 365, interval: str = "1day", exchange_code: str = "NSE", product_type: str = "cash", expiry_date: str = "", strike_price: str = "", right: str = ""):
    df = fetch_icici_data(symbol, days=days, interval=interval, exchange_code=exchange_code, product_type=product_type, expiry_date=expiry_date, strike_price=strike_price, right=right)
    if df is None or df.empty:
        if exchange_code == "NFO" and product_type in ("futures", "options"):
            st.warning(f"No historical data found for {symbol} {product_type} contract. Falling back to underlying {symbol} equity data.")
            df = fetch_icici_data(symbol, days=days, interval=interval, exchange_code="NSE", product_type="cash")
            if df is None:
                df = generate_mock_data(symbol, days=days, interval=interval)
        else:
            df = generate_mock_data(symbol, days=days, interval=interval)
    return clean_time_series(df, interval=interval)


def get_cached_data(symbol: str, days: int = 365, interval: str = "1day", exchange_code: str = "NSE", product_type: str = "cash", expiry_date: str = "", strike_price: str = "", right: str = ""):
    cache_key = f"data_{symbol}_{days}_{interval}_{exchange_code}_{product_type}_{expiry_date}_{strike_price}_{right}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    df = get_data(symbol, days=days, interval=interval, exchange_code=exchange_code, product_type=product_type, expiry_date=expiry_date, strike_price=strike_price, right=right)
    st.session_state[cache_key] = df
    return df


def check_password():
    try:
        expected_password = st.secrets["APP_PASSWORD"]
    except Exception:
        expected_password = os.getenv("APP_PASSWORD", "")

    if not expected_password:
        st.info("No password configured. Set APP_PASSWORD in Streamlit secrets.")
        st.stop()

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("🔒 Login")
        password = st.text_input("Enter password", type="password")
        if st.button("Submit"):
            if password == expected_password:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()


def main():
    st.set_page_config(page_title="NSE Chronos Forecaster", layout="wide")
    check_password()

    with st.sidebar:
        st.subheader("⚙️ Settings")

        api_key, api_secret, session_token = _load_credentials()
        current_override = st.session_state.get("override_session_token", "")
        active_token = current_override or session_token
        token_source = "Session State" if current_override else "Secrets / .env"

        st.caption(f"**Active token source:** {token_source}")
        st.caption(f"**Token preview:** {active_token[:8]}..." if active_token else "**No token set**")

        new_token = st.text_input("ICICI Session Token", value=current_override, type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Token"):
                st.session_state.override_session_token = new_token.strip()
                st.success("Token updated")
                st.rerun()
        with col2:
            if st.button("Test Connection"):
                test_token = new_token.strip() or current_override
                if not test_token:
                    st.warning("Enter a session token first")
                else:
                    with st.spinner("Testing ICICI connection..."):
                        test_breeze = BreezeConnect(api_key=api_key)
                        try:
                            test_breeze.generate_session(api_secret=api_secret, session_token=test_token)
                            st.success("✅ Session token is valid")
                        except Exception as e:
                            st.error(f"❌ Session token rejected: {e}")

        st.divider()
        st.caption("⚠️ On Streamlit Cloud, session state may reset after ~5 min of inactivity. If you see 'Session key expired', re-enter your token.")

    st.title("📈 NSE Stock Forecast — Chronos-T5")

    segment = st.selectbox("Segment", options=["Equity", "Futures", "Options"], index=0)

    if segment == "Equity":
        master_path = SECURITY_MASTER_PATH
        if not os.path.exists(master_path):
            st.error(f"Security master not found at {master_path}")
            st.stop()
        master_df = load_security_master(master_path)
        search = st.text_input("Search stock / company name", placeholder="e.g. RELIANCE, INFY, TATA...").strip().upper()
        if not search:
            st.info("Type a stock name or symbol to search.")
            st.stop()
        matches = master_df[master_df["__search__"].str.contains(search, na=False)].copy()
        if matches.empty:
            st.warning("No matching NSE stock found.")
            st.stop()
        matches["__option__"] = matches["__display__"] + " — " + matches["CompanyName"].str.strip()
        options = matches["__option__"].unique().tolist()
        selected_option = st.selectbox("Select a stock", options=options, index=0)
        symbol = selected_option.split(" — ")[0].strip()
        exchange_code = "NSE"
        product_type = "cash"
        expiry_date = ""
        strike_price = ""
        right = ""
    elif segment == "Futures":
        fo_path = FO_SECURITY_MASTER_PATH
        if not os.path.exists(fo_path):
            st.error(f"F&O security master not found at {fo_path}")
            st.stop()
        fo_df = load_fo_master(fo_path)
        fut_df = fo_df[fo_df["Series"].str.upper() == "FUTURE"].copy()
        search = st.text_input("Search futures contract", placeholder="e.g. RELIANCE, NIFTY...").strip().upper()
        if not search:
            st.info("Type a futures symbol to search.")
            st.stop()
        matches = fut_df[fut_df["__search__"].str.contains(search, na=False)].copy()
        if matches.empty:
            st.warning("No matching NSE futures found.")
            st.stop()
        matches["__option__"] = matches["__display__"] + " — " + matches["ExpiryDate"].str.strip()
        options = matches["__option__"].unique().tolist()
        selected_option = st.selectbox("Select a futures contract", options=options, index=0)
        symbol = selected_option.split(" — ")[0].strip()
        expiry_date = selected_option.split(" — ")[-1].strip()
        exchange_code = "NFO"
        product_type = "futures"
        strike_price = ""
        right = ""
    else:
        fo_path = FO_SECURITY_MASTER_PATH
        if not os.path.exists(fo_path):
            st.error(f"F&O security master not found at {fo_path}")
            st.stop()
        fo_df = load_fo_master(fo_path)
        opt_df = fo_df[fo_df["Series"].str.upper() == "OPTION"].copy()
        search = st.text_input("Search options contract", placeholder="e.g. RELIANCE, NIFTY...").strip().upper()
        if not search:
            st.info("Type an options symbol to search.")
            st.stop()
        matches = opt_df[opt_df["__search__"].str.contains(search, na=False)].copy()
        if matches.empty:
            st.warning("No matching NSE options found.")
            st.stop()

        def display_option_type(val):
            if str(val).upper() == "CE":
                return "Call"
            elif str(val).upper() == "PE":
                return "Put"
            return str(val)

        matches["__option__"] = (
            matches["__display__"]
            + " — "
            + matches["ExpiryDate"].str.strip()
            + " — "
            + matches["StrikePrice"].str.strip()
            + " — "
            + matches["OptionType"].apply(display_option_type)
        )
        options = matches["__option__"].unique().tolist()
        selected_option = st.selectbox("Select an options contract", options=options, index=0)
        parts = selected_option.split(" — ")
        symbol = parts[0].strip()
        expiry_date = parts[1].strip() if len(parts) > 1 else ""
        strike_price = parts[2].strip() if len(parts) > 2 else ""
        raw_right = parts[3].strip() if len(parts) > 3 else ""
        right = "call" if raw_right.upper() == "CE" else ("put" if raw_right.upper() == "PE" else raw_right.lower())
        exchange_code = "NFO"
        product_type = "options"

    with st.form("forecast_form"):
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            days = st.number_input("Lookback days", min_value=30, max_value=3650, value=365, step=30)
        with col2:
            interval = st.selectbox("Interval", options=["1day", "15minute", "1minute"], index=0)
        with col3:
            model_name = st.selectbox("Chronos Model", options=list(CHRONOS_MODELS.keys()), index=1)
        with col4:
            ensemble_method = st.selectbox("Ensemble", options=list(ENSEMBLE_METHODS.keys()), index=0)
        with col5:
            min_rr = st.number_input("Min Risk-Reward", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        with col6:
            max_kelly = st.number_input("Max Kelly %", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
        with col7:
            if interval == "1day":
                horizon = st.number_input("Horizon (days)", min_value=1, max_value=64, value=30, step=1)
            elif interval == "15minute":
                horizon = st.number_input("Horizon (15min bars)", min_value=1, max_value=64, value=26, step=1)
            else:
                horizon = st.number_input("Horizon (1min bars)", min_value=1, max_value=64, value=64, step=1)
        run_btn = st.form_submit_button("Run Forecast", type="primary")

    if not run_btn:
        st.stop()

    model_id = CHRONOS_MODELS[model_name]
    ensemble_key = ENSEMBLE_METHODS[ensemble_method]

    with st.spinner(f"Fetching {symbol} data ({interval}) and running {model_name} + {ensemble_method}..."):
        try:
            series = get_cached_data(symbol, days=int(days), interval=interval, exchange_code=exchange_code, product_type=product_type, expiry_date=expiry_date, strike_price=strike_price, right=right)
            indicators = calculate_indicators(series, interval=interval)
            gbdt_data = build_gbdt_features(series, indicators)
            gbdt_models = train_gbdt_models(gbdt_data, horizon=int(horizon))
            gbdt_forecast = forecast_gbdt(gbdt_models, gbdt_data, horizon=int(horizon))

            transformer = StationaryTransformer(method="log_return")
            stationary_series = transformer.fit_transform(series)
            chronos_forecast = run_chronos_forecast(stationary_series, horizon=int(horizon), model_name=model_id)
            chronos_forecast = transformer.inverse_transform(chronos_forecast)

            arima_df = None
            if ensemble_key == "hybrid":
                arima_model = ARIMAForecaster(order=(1, 0, 1))
                arima_model.fit(stationary_series)
                arima_stationary = arima_model.predict(int(horizon))
                arima_df = transformer.inverse_transform(arima_stationary)

            forecast_df = ensemble_forecast(chronos_forecast, gbdt_forecast, method=ensemble_key, arima_df=arima_df)

            if gbdt_forecast.isna().all().all() and ensemble_key not in ("chronos_only", "hybrid"):
                st.warning("GBDT ensemble skipped due to insufficient data or features. Falling back to Chronos-only forecast.")
                forecast_df = ensemble_forecast(chronos_forecast, gbdt_forecast, method="chronos_only", arima_df=arima_df)

            signals = generate_signals(series, forecast_df, min_risk_reward=float(min_rr), max_kelly_pct=float(max_kelly)/100.0)
            backtest = backtest_forecast(series, horizon=int(horizon), model_name=model_id)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Signal", "Chart", "Indicators", "Data"])

    with tab1:
        st.warning("⚠️ These are READ-ONLY signals. No orders are ever placed automatically.")

        horizon_label = "days" if interval == "1day" else ("15min bars" if interval == "15minute" else "1min bars")
        st.caption(f"Forecast shown is the **endpoint** of the {int(horizon)}-{horizon_label} horizon, not an aggregate return.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{signals['current_price']:.2f}")
        col2.metric("Median Forecast (end of horizon)", f"{signals['median_forecast']:.2f}", delta=f"{signals['forecast_change_pct']:+.2f}%")
        col3.metric("Lower Bound (q10)", f"{signals['lower_bound']:.2f}")
        col4.metric("Upper Bound (q90)", f"{signals['upper_bound']:.2f}")

        st.subheader(f"Signal: {signals['signal']}")
        st.write(signals["reason"])

        if "HOLD" in signals["signal"] and signals["forecast_change_pct"] <= -3.0:
            st.caption(f"⚠️ HOLD Signal Active: Median forecast is negative ({signals['forecast_change_pct']:+.2f}%). Trade setup rejected.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ATR", f"{signals['atr']:.2f}")
        col2.metric("Recommended Stop-Loss", f"{signals['stop_loss']:.2f}")
        if signals["risk_reward"] is not None:
            col3.metric("Risk-Reward", f"{signals['risk_reward']:.2f} : 1", help="Measures potential upside distance vs downside stop-loss distance. Standard rule requires ≥ 2.00 to pass circuit-breaker.")
        else:
            col3.metric("Risk-Reward", "N/A", help="Measures potential upside distance vs downside stop-loss distance. Standard rule requires ≥ 2.00 to pass circuit-breaker.")
        col4.metric("Confidence", f"{signals['confidence']*100:.1f}%", help="Percentage of simulated model trajectories remaining within q10-q90 bounds. Measures forecast stability, NOT guaranteed win rate.")

        st.subheader("Position Sizing (Kelly Criterion)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Probability (raw)", f"{signals['win_prob']*100:.1f}%", help="Model boundary consistency score used in Kelly position sizing math.")
        col2.metric("Win Probability (vol-adj)", f"{signals['vol_adjusted_win_prob']*100:.1f}%", help="Model boundary consistency score used in Kelly position sizing math.")
        col3.metric("Stop Distance", f"{signals['stop_distance_pct']*100:.2f}%", help="Percentage gap between current price and recommended stop-loss price.")
        col4.metric("Kelly Fraction (capped)", f"{signals['kelly_fraction']*100:.2f}%", help="Calculated capital allocation based on confidence and volatility, hard-capped at the Max Kelly % limit.")

        with st.expander("ℹ️ How to interpret these metrics"):
            st.markdown("""
            - **Risk-Reward (≥ 2.00 required):** Upside potential vs downside risk. Ratio of expected gain to maximum loss.
            - **Directional Forecast:** Shows expected endpoint of the forecast horizon, not an aggregate return.
            - **Kelly Sizing:** Dynamic account risk cap based on confidence and stop distance. Hard-capped at the Max Kelly % limit (default 5%).
            - **Confidence:** Measures how tight the model's prediction band is. Lower confidence = wider band = higher uncertainty.
            - **ABNORMAL MOMENTUM:** Price has breached the upper forecast bound. Treat as a warning regime; no position sizing applied.
            """)

        st.caption(f"Optimal position size = {signals['kelly_fraction']*100:.2f}% of capital. Kelly is capped at {signals['max_kelly_pct']*100:.1f}% of capital. Win probability is scaled down by stop-distance volatility. Stop-loss = max(ATR-based, percentile-based). Rules: BUY if median forecast ≥ +3% and q10 above stop-loss. ABNORMAL MOMENTUM if current price > q90. SELL if median forecast ≤ -3%.")

    with tab2:
        import plotly.graph_objects as go
        import tempfile

        last_date = series.index[-1]
        current_price = float(series.iloc[-1])

        if interval == "1day":
            trading_days = get_nse_trading_days(last_date, last_date + timedelta(days=365))
            forecast_dates = trading_days[:len(forecast_df)]
        elif interval == "15minute":
            forecast_dates = pd.date_range(start=last_date + timedelta(minutes=15), periods=len(forecast_df), freq="15min", tz="Asia/Kolkata")
        else:
            forecast_dates = pd.date_range(start=last_date + timedelta(minutes=1), periods=len(forecast_df), freq="1min", tz="Asia/Kolkata")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[last_date],
            y=[current_price],
            mode="markers",
            name="Current Price",
            marker=dict(color="#1f77b4", size=10, symbol="circle"),
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df["q50"].values,
            mode="lines",
            name="Median Forecast",
            line=dict(color="#ff7f0e", width=2),
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df["q90"].values,
            mode="lines",
            name="90th Percentile",
            line=dict(color="#ff7f0e", width=1, dash="dot"),
        ))

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df["q10"].values,
            mode="lines",
            name="10th Percentile",
            line=dict(color="#ff7f0e", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.2)",
        ))

        fig.update_layout(
            title=f"{symbol} — Current to {len(forecast_df)}-{horizon_label} Forecast ({model_name})",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Technical Indicators (latest values)")
        if not indicators.empty:
            latest = indicators.iloc[-1]
            cols = st.columns(len(indicators.columns))
            for i, col in enumerate(indicators.columns):
                cols[i].metric(col.upper(), f"{latest[col]:.4f}")
        else:
            st.info("No indicators available.")

        st.subheader("Backtest Metrics (walk-forward out-of-sample)")
        if backtest.get("mape") is not None:
            col1, col2 = st.columns(2)
            col1.metric("MAPE", f"{backtest['mape']:.2f}%")
            col2.metric("Directional Accuracy", f"{backtest['directional_accuracy']:.2f}%")
            st.caption(backtest.get("note", ""))
        else:
            st.warning("Backtest unavailable: " + str(backtest.get("note", "")))

    with tab4:
        if interval == "1day":
            trading_days = get_nse_trading_days(series.index[-1], series.index[-1] + timedelta(days=365))
            forecast_dates = trading_days[:len(forecast_df)]
        elif interval == "15minute":
            forecast_dates = pd.date_range(start=series.index[-1] + timedelta(minutes=15), periods=len(forecast_df), freq="15min", tz="Asia/Kolkata")
        else:
            forecast_dates = pd.date_range(start=series.index[-1] + timedelta(minutes=1), periods=len(forecast_df), freq="1min", tz="Asia/Kolkata")

        result_df = pd.DataFrame({
            "forecast_date": forecast_dates,
            "q10": forecast_df["q10"].values,
            "q50": forecast_df["q50"].values,
            "q90": forecast_df["q90"].values,
        })
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_forecast.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
