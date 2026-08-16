import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecast_pipeline import (
    generate_mock_data,
    fetch_icici_data,
    clean_time_series,
    run_chronos_forecast,
    calculate_indicators,
    build_gbdt_features,
    train_gbdt_models,
    forecast_gbdt,
    ensemble_forecast,
    get_nse_trading_days,
    StationaryTransformer,
    HybridTimeSeriesEnsemble,
    ARIMAForecaster,
    calculate_directional_accuracy,
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
def get_data_up_to(symbol: str, as_of_date: datetime, days: int = 365, interval: str = "1day", exchange_code: str = "NSE", product_type: str = "cash", expiry_date: str = "", strike_price: str = "", right: str = ""):
    end_date = as_of_date
    start_date = end_date - timedelta(days=days)
    df = fetch_icici_data(symbol, days=days, interval=interval, exchange_code=exchange_code, product_type=product_type, expiry_date=expiry_date, strike_price=strike_price, right=right)
    if df is None or df.empty:
        if exchange_code == "NFO" and product_type in ("futures", "options"):
            st.warning(f"No historical data found for {symbol} {product_type} contract. Falling back to underlying {symbol} equity data.")
            df = fetch_icici_data(symbol, days=days, interval=interval, exchange_code="NSE", product_type="cash")
            if df is None:
                df = generate_mock_data(symbol, days=days, interval=interval)
        else:
            df = generate_mock_data(symbol, days=days, interval=interval)
    df = clean_time_series(df, interval=interval)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    cutoff = pd.Timestamp(end_date)
    if cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    df = df[df.index <= cutoff]
    return df


def get_cached_data_up_to(symbol: str, as_of_date: datetime, days: int = 365, interval: str = "1day", exchange_code: str = "NSE", product_type: str = "cash", expiry_date: str = "", strike_price: str = "", right: str = ""):
    cache_key = f"data_{symbol}_{days}_{interval}_{exchange_code}_{product_type}_{expiry_date}_{strike_price}_{right}_{as_of_date.date()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    df = get_data_up_to(symbol, as_of_date, days=days, interval=interval, exchange_code=exchange_code, product_type=product_type, expiry_date=expiry_date, strike_price=strike_price, right=right)
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
    st.set_page_config(page_title="Chronos Verification", layout="wide")
    check_password()

    with st.sidebar:
        st.subheader("⚙️ Settings")
        current_token = st.session_state.get("override_session_token", "")
        new_token = st.text_input("ICICI Session Token", value=current_token, type="password")
        if st.button("Update Token"):
            st.session_state.override_session_token = new_token.strip()
            st.success("Token updated")
            st.rerun()

    st.title("🔍 Chronos Forecast Verification")

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

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        as_of_date = st.date_input("Forecast as-of date", value=datetime.today() - timedelta(days=60))
    with col2:
        interval = st.selectbox("Interval", options=["1day", "15minute", "1minute"], index=0)
    with col3:
        model_name = st.selectbox("Chronos Model", options=list(CHRONOS_MODELS.keys()), index=1)
    with col4:
        ensemble_method = st.selectbox("Ensemble", options=list(ENSEMBLE_METHODS.keys()), index=0)
    with col5:
        if interval == "1day":
            horizon = st.number_input("Horizon (days)", min_value=1, max_value=64, value=30, step=1)
        elif interval == "15minute":
            horizon = st.number_input("Horizon (15min bars)", min_value=1, max_value=64, value=26, step=1)
        else:
            horizon = st.number_input("Horizon (1min bars)", min_value=1, max_value=64, value=64, step=1)

    run_btn = st.button("Run Verification", type="primary")
    if not run_btn:
        st.stop()

    as_of_datetime = datetime.combine(as_of_date, datetime.min.time())
    with st.spinner(f"Fetching {symbol} history up to {as_of_date} and generating forecast..."):
        try:
            history_cut = get_cached_data_up_to(symbol, as_of_datetime, days=365, interval=interval, exchange_code=exchange_code, product_type=product_type, expiry_date=expiry_date, strike_price=strike_price, right=right)
            indicators = calculate_indicators(history_cut, interval=interval)
            gbdt_data = build_gbdt_features(history_cut, indicators)
            gbdt_models = train_gbdt_models(gbdt_data, horizon=int(horizon))
            gbdt_forecast = forecast_gbdt(gbdt_models, gbdt_data, horizon=int(horizon))

            transformer = StationaryTransformer(method="log_return")
            stationary_series = transformer.fit_transform(history_cut)
            chronos_forecast = run_chronos_forecast(stationary_series, horizon=int(horizon), model_name=CHRONOS_MODELS[model_name])
            chronos_forecast = transformer.inverse_transform(chronos_forecast)

            arima_df = None
            if ENSEMBLE_METHODS[ensemble_method] == "hybrid":
                arima_model = ARIMAForecaster(order=(1, 0, 1))
                arima_model.fit(stationary_series)
                arima_stationary = arima_model.predict(int(horizon))
                arima_df = transformer.inverse_transform(arima_stationary)

            forecast_df = ensemble_forecast(chronos_forecast, gbdt_forecast, method=ENSEMBLE_METHODS[ensemble_method], arima_df=arima_df)

            if gbdt_forecast.isna().all().all() and ENSEMBLE_METHODS[ensemble_method] not in ("chronos_only", "hybrid"):
                st.warning("GBDT ensemble skipped due to insufficient data or features. Falling back to Chronos-only forecast.")
                forecast_df = ensemble_forecast(chronos_forecast, gbdt_forecast, method="chronos_only", arima_df=arima_df)
        except Exception as e:
            st.error(f"Verification failed: {e}")
            st.stop()

    last_history_date = history_cut.index[-1]
    if interval == "1day":
        trading_days = get_nse_trading_days(last_history_date, last_history_date + timedelta(days=365))
        forecast_dates = trading_days[:len(forecast_df)]
    elif interval == "15minute":
        forecast_dates = pd.date_range(start=last_history_date + timedelta(minutes=15), periods=len(forecast_df), freq="15min", tz="Asia/Kolkata")
    else:
        forecast_dates = pd.date_range(start=last_history_date + timedelta(minutes=1), periods=len(forecast_df), freq="1min", tz="Asia/Kolkata")

    actuals = get_cached_data_up_to(symbol, as_of_datetime + timedelta(days=int(horizon) * 2), days=365, interval=interval)
    if actuals.index.tz is not None:
        actuals.index = actuals.index.tz_localize(None)
    actuals = actuals[actuals.index > pd.Timestamp(last_history_date)]
    actuals = actuals.iloc[:int(horizon)]

    tab1, tab2, tab3 = st.tabs(["Chart", "Accuracy", "Data"])

    with tab1:
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[last_history_date],
            y=[float(history_cut.iloc[-1])],
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

        if not actuals.empty:
            fig.add_trace(go.Scatter(
                x=actuals.index,
                y=actuals.values,
                mode="lines",
                name="Actual",
                line=dict(color="#2ca02c", width=2),
            ))

        fig.update_layout(
            title=f"{symbol} Verification: Forecast from {last_history_date.date()} vs Actual",
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

    with tab2:
        st.subheader("Forecast Accuracy vs Actuals")
        if actuals.empty:
            st.warning("No actual data available for the forecast horizon. Accuracy metrics cannot be computed.")
        else:
            predicted = forecast_df["q50"].values[:len(actuals)]
            actual_values = actuals.values[:len(predicted)]

            mape = np.mean(np.abs((actual_values - predicted) / actual_values)) * 100
            rmse = np.sqrt(np.mean((actual_values - predicted) ** 2))
            mae = np.mean(np.abs(actual_values - predicted))

            actual_dir = np.diff(actual_values)
            pred_dir = np.diff(predicted)
            direction_correct = np.sum((actual_dir > 0) & (pred_dir > 0)) + np.sum((actual_dir < 0) & (pred_dir < 0))
            total = len(actual_dir)
            directional_accuracy = (direction_correct / total * 100) if total > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAPE", f"{mape:.2f}%")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("MAE", f"{mae:.2f}")
            col4.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")

            comparison_df = pd.DataFrame({
                "date": actuals.index[:len(predicted)],
                "actual": actual_values,
                "forecast": predicted,
                "error_pct": ((actual_values - predicted) / actual_values * 100),
            })
            st.dataframe(comparison_df, use_container_width=True)
            st.download_button(
                "Download CSV",
                comparison_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{symbol}_verification.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("Forecast Data")
        forecast_out = pd.DataFrame({
            "forecast_date": forecast_dates,
            "q10": forecast_df["q10"].values,
            "q50": forecast_df["q50"].values,
            "q90": forecast_df["q90"].values,
        })
        st.dataframe(forecast_out, use_container_width=True)

        if not actuals.empty:
            st.subheader("Actual Data")
            actual_out = pd.DataFrame({
                "date": actuals.index,
                "actual": actuals.values,
            })
            st.dataframe(actual_out, use_container_width=True)


if __name__ == "__main__":
    main()
