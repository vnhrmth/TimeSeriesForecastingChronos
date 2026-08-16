import os
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from chronos import BaseChronosPipeline
from breeze_connect import BreezeConnect
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import pandas_market_calendars as mcal
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class StationaryTransformer:
    def __init__(self, method: str = "log_return"):
        self.method = method
        self.last_price = None
        self.last_date = None

    def fit_transform(self, series: pd.Series) -> pd.Series:
        if self.method == "log_return":
            transformed = np.log(series).diff().dropna()
        elif self.method == "pct_change":
            transformed = series.pct_change().dropna()
        else:
            transformed = series
        self.last_price = float(series.iloc[-1])
        self.last_date = series.index[-1]
        return transformed

    def inverse_transform(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if self.last_price is None:
            return forecast_df
        current = self.last_price
        q10, q50, q90 = [], [], []
        for r10, r50, r90 in zip(forecast_df["q10"].values, forecast_df["q50"].values, forecast_df["q90"].values):
            if self.method == "log_return":
                p10 = current * np.exp(r10)
                p50 = current * np.exp(r50)
                p90 = current * np.exp(r90)
            elif self.method == "pct_change":
                p10 = current * (1 + r10)
                p50 = current * (1 + r50)
                p90 = current * (1 + r90)
            else:
                p10, p50, p90 = r10, r50, r90
            q10.append(p10)
            q50.append(p50)
            q90.append(p90)
            current = p50
        return pd.DataFrame({"q10": q10, "q50": q50, "q90": q90})


class HybridTimeSeriesEnsemble:
    def __init__(self, chronos_weight: float = 0.6, arima_weight: float = 0.2, gbdt_weight: float = 0.2):
        self.chronos_weight = chronos_weight
        self.arima_weight = arima_weight
        self.gbdt_weight = gbdt_weight
        self.bias_correction = None
        self.arima_model = None

    def fit_arima(self, series: pd.Series, horizon: int = 5):
        try:
            model = sm.tsa.ARIMA(series.values, order=(1, 0, 1))
            fitted = model.fit()
            self.arima_model = fitted
            arima_forecast = fitted.forecast(steps=horizon)
            residuals = series.values[-horizon:] - arima_forecast[:len(series.values[-horizon:])]
            self.bias_correction = np.mean(residuals) if len(residuals) > 0 else 0.0
        except Exception as e:
            logger.warning("ARIMA fit failed: %s", e)
            self.bias_correction = 0.0

    def blend(self, chronos_df: pd.DataFrame, arima_df: pd.DataFrame, gbdt_df: pd.DataFrame) -> pd.DataFrame:
        chronos_q50 = chronos_df["q50"].values
        arima_q50 = arima_df["q50"].values if arima_df is not None and not arima_df.isna().all().all() else chronos_q50
        gbdt_q50 = gbdt_df["q50"].values if gbdt_df is not None and not gbdt_df.isna().all().all() else chronos_q50

        blended_q50 = (
            self.chronos_weight * chronos_q50 +
            self.arima_weight * arima_q50 +
            self.gbdt_weight * gbdt_q50
        )
        if self.bias_correction is not None:
            blended_q50 = blended_q50 - self.bias_correction

        q10 = (
            self.chronos_weight * chronos_df["q10"].values +
            self.arima_weight * arima_df.get("q10", chronos_df["q10"]).values +
            self.gbdt_weight * gbdt_df.get("q10", chronos_df["q10"]).values
        )
        q90 = (
            self.chronos_weight * chronos_df["q90"].values +
            self.arima_weight * arima_df.get("q90", chronos_df["q90"]).values +
            self.gbdt_weight * gbdt_df.get("q90", chronos_df["q90"]).values
        )

        return pd.DataFrame({"q10": q10, "q50": blended_q50, "q90": q90})


class ARIMAForecaster:
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1)):
        self.order = order
        self.model = None

    def fit(self, series: pd.Series):
        try:
            self.model = sm.tsa.ARIMA(series.values, order=self.order)
            self.model = self.model.fit()
        except Exception as e:
            logger.warning("ARIMA fit failed: %s", e)
            self.model = None

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame({"q10": [np.nan] * horizon, "q50": [np.nan] * horizon, "q90": [np.nan] * horizon})
        try:
            forecast = self.model.forecast(steps=horizon)
            std = np.std(self.model.resid) if hasattr(self.model, "resid") else np.std(forecast) * 0.1
            return pd.DataFrame({
                "q10": forecast - 1.28 * std,
                "q50": forecast,
                "q90": forecast + 1.28 * std,
            })
        except Exception as e:
            logger.warning("ARIMA predict failed: %s", e)
            return pd.DataFrame({"q10": [np.nan] * horizon, "q50": [np.nan] * horizon, "q90": [np.nan] * horizon})


def _load_credentials() -> Tuple[str, str, str]:
    api_key = ""
    api_secret = ""
    session_token = ""

    try:
        import streamlit as st
        if "ICICI_API_KEY" in st.secrets:
            api_key = st.secrets["ICICI_API_KEY"]
        if "ICICI_API_SECRET" in st.secrets:
            api_secret = st.secrets["ICICI_API_SECRET"]
        if "ICICI_SESSION_TOKEN" in st.secrets:
            session_token = st.secrets["ICICI_SESSION_TOKEN"]
    except Exception:
        pass

    api_key = api_key or os.getenv("ICICI_API_KEY", "") or os.getenv("ICICI_APP_KEY", "")
    api_secret = api_secret or os.getenv("ICICI_API_SECRET", "")
    session_token = session_token or os.getenv("ICICI_SESSION_TOKEN", "")

    try:
        import streamlit as st
        override = st.session_state.get("override_session_token", "")
        if override:
            session_token = override
    except Exception:
        pass

    return api_key, api_secret, session_token


def get_nse_trading_days(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    nse = mcal.get_calendar("NSE")
    schedule = nse.schedule(start_date=start_date, end_date=end_date)
    if schedule.empty:
        return pd.DatetimeIndex([])
    trading_days = pd.DatetimeIndex(schedule.index)
    if trading_days.tz is not None:
        trading_days = trading_days.tz_localize(None)
    return trading_days


TARGET_SYMBOL = os.getenv("TARGET_SYMBOL", "RELIANCE")
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "5"))
_ICICI_API_KEY, _ICICI_API_SECRET, _ICICI_SESSION_TOKEN = _load_credentials()


def generate_mock_data(symbol: str = TARGET_SYMBOL, days: int = 365, interval: str = "1day") -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    if interval == "1minute":
        minutes_per_day = 395
        total_minutes = days * minutes_per_day
        date_range = pd.date_range(start=start_date, end=end_date, freq="1min", tz="Asia/Kolkata")
        date_range = date_range[date_range.time >= pd.Timestamp("09:15:00").time()]
        date_range = date_range[date_range.time <= pd.Timestamp("15:30:00").time()]
        if len(date_range) > total_minutes:
            date_range = date_range[:total_minutes]
    elif interval == "15minute":
        minutes_per_day = 26
        total_points = days * minutes_per_day
        date_range = pd.date_range(start=start_date, end=end_date, freq="15min", tz="Asia/Kolkata")
        date_range = date_range[date_range.time >= pd.Timestamp("09:15:00").time()]
        date_range = date_range[date_range.time <= pd.Timestamp("15:30:00").time()]
        if len(date_range) > total_points:
            date_range = date_range[:total_points]
    else:
        trading_days = get_nse_trading_days(start_date, end_date)
        if len(trading_days) == 0:
            date_range = pd.bdate_range(start=start_date, end=end_date)
        else:
            date_range = trading_days

    base_price = 2500.0
    returns = np.random.normal(loc=0.0002, scale=0.015, size=len(date_range))
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"date": date_range, "close": prices})
    df.set_index("date", inplace=True)
    logger.info("Generated %d points of mock data for %s (%s, no ICICI credentials)", len(df), symbol, interval)
    return df


def fetch_icici_data(
    symbol: str = TARGET_SYMBOL,
    days: int = 365,
    interval: str = "1day",
    exchange_code: str = "NSE",
    product_type: str = "cash",
    expiry_date: str = "",
    strike_price: str = "",
    right: str = "",
) -> Optional[pd.DataFrame]:
    api_key, api_secret, session_token = _load_credentials()
    if not all([api_key, api_secret, session_token]):
        logger.warning("ICICI credentials not fully provided. Falling back to mock data.")
        return None

    logger.info("Attempting ICICI Direct Breeze API connection for %s (%s/%s)...", symbol, exchange_code, product_type)
    try:
        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)

        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        iso_from = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        iso_to = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        attempts = []

        kwargs = dict(
            interval=interval,
            from_date=iso_from,
            to_date=iso_to,
            stock_code=symbol,
            exchange_code=exchange_code,
            product_type=product_type,
        )
        if product_type == "futures" and expiry_date:
            kwargs["expiry_date"] = expiry_date
        if product_type == "options" and expiry_date:
            kwargs["expiry_date"] = expiry_date
        if strike_price:
            kwargs["strike_price"] = str(strike_price)
        if right:
            kwargs["right"] = right

        try:
            data = breeze.get_historical_data_v2(**kwargs)
            attempts.append(("v2", interval, exchange_code, product_type, data))
        except Exception as e:
            attempts.append(("v2", interval, exchange_code, product_type, str(e)))

        try:
            data = breeze.get_historical_data(**kwargs)
            attempts.append(("v1", interval, exchange_code, product_type, data))
        except Exception as e:
            attempts.append(("v1", interval, exchange_code, product_type, str(e)))

        for version, intv, exch, prod, result in attempts:
            logger.info("ICICI attempt [%s %s %s %s]: %s", version, intv, exch, prod, result)
            if isinstance(result, dict) and result.get("Status") == 200:
                records = result.get("Success") or []
                if records:
                    df = pd.DataFrame(records)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)
                    df = df[["close"]].astype(float)
                    df = df.sort_index()
                    logger.info("Fetched %d records via [%s %s %s %s]", len(df), version, intv, exch, prod)
                    return df

        raise RuntimeError(f"ICICI returned no data after {len(attempts)} attempts")

    except Exception as exc:
        logger.warning("ICICI fetch failed: %s. Falling back to mock data.", exc)
        return None


def clean_time_series(df: pd.DataFrame, interval: str = "1day") -> pd.Series:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if interval == "1minute":
        freq = "1min"
    elif interval == "15minute":
        freq = "15min"
    else:
        start_date = df.index.min()
        end_date = df.index.max()
        trading_days = get_nse_trading_days(start_date, end_date)
        if len(trading_days) > 0:
            df = df.reindex(trading_days)
            df = df.dropna(subset=["close"])
            freq = "C"
        else:
            freq = "B"

    if interval != "1day":
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_range)

    if interval != "1day":
        df["close"] = df["close"].ffill().bfill()

    df.index.name = "date"
    series = df["close"].astype(float)
    logger.info("Cleaned series: %d observations (%s) from %s to %s",
                len(series), interval, series.index.min(), series.index.max())
    return series


def calculate_indicators(series: pd.Series, interval: str = "1day") -> pd.DataFrame:
    df = pd.DataFrame({"close": series})
    try:
        df = add_all_ta_features(
            df=df,
            open="close",
            high="close",
            low="close",
            close="close",
            volume="close",
            fillna=True,
        )
    except Exception as e:
        logger.warning("Failed to calculate all TA features: %s", e)
        df["rsi"] = df["close"].rolling(14).apply(lambda x: (100 - (100 / (1 + (x.diff().clip(lower=0).sum() / -x.diff().clip(upper=0).sum())))))
        df["sma_20"] = df["close"].rolling(20).mean()
        df["atr"] = (df["close"].diff().abs()).rolling(14).mean()

    keep = [c for c in ["rsi", "sma_20", "atr", "momentum_rsi", "volatility_atr"] if c in df.columns]
    return df[keep]


def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or len(predicted) < 2:
        return 0.0
    actual_dir = np.diff(actual)
    pred_dir = np.diff(predicted)
    if len(actual_dir) == 0:
        return 0.0
    correct = np.sum((actual_dir > 0) & (pred_dir > 0)) + np.sum((actual_dir < 0) & (pred_dir < 0))
    return float(correct / len(actual_dir) * 100)


def select_context_window(history: pd.Series, horizon: int, volatility_window: int = 20) -> int:
    if len(history) < 30:
        return len(history)
    returns = history.pct_change().dropna()
    vol = returns.rolling(volatility_window).std().iloc[-1] if len(returns) >= volatility_window else returns.std()
    base = max(horizon + 25, 30)
    if vol > 0.025:
        return min(len(history), base + 40)
    elif vol < 0.015:
        return min(len(history), base)
    else:
        return min(len(history), base + 20)


def run_chronos_forecast(train_series: pd.Series, horizon: int = FORECAST_HORIZON, model_name: str = "amazon/chronos-t5-small") -> pd.DataFrame:
    logger.info("Loading %s from Hugging Face Hub...", model_name)
    chronos = BaseChronosPipeline.from_pretrained(model_name, device_map="cpu")
    chronos.model.eval()

    context = torch.tensor(train_series.values, dtype=torch.float32).unsqueeze(0)

    logger.info("Running probabilistic forecast (horizon=%d)...", horizon)
    quantiles, mean = chronos.predict_quantiles(
        inputs=context,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    q10 = quantiles[0, :, 0].cpu().numpy()
    q50 = quantiles[0, :, 1].cpu().numpy()
    q90 = quantiles[0, :, 2].cpu().numpy()
    logger.info("Chronos forecast generated successfully")
    return pd.DataFrame({"q10": q10, "q50": q50, "q90": q90})


def backtest_forecast(history: pd.Series, horizon: int = 5, model_name: str = "amazon/chronos-t5-small", max_windows: int = 20) -> dict:
    min_train = select_context_window(history, horizon)
    if len(history) < min_train + horizon:
        return {"mape": None, "directional_accuracy": None, "note": f"Insufficient history for {min_train + horizon}-point walk-forward backtest"}

    errors = []
    directional_accuracy_scores = []
    direction_correct_count = 0
    direction_total = 0
    windows = 0

    max_start = len(history) - horizon
    raw_positions = list(range(min_train, max_start + 1, horizon))
    if len(raw_positions) > max_windows:
        step = max(1, len(raw_positions) // max_windows)
        positions = raw_positions[::step][:max_windows]
    else:
        positions = raw_positions

    for start_idx in positions:
        train = history.iloc[:start_idx]
        actual_window = history.iloc[start_idx:start_idx + horizon]

        try:
            forecast_df = run_chronos_forecast(train, horizon=horizon, model_name=model_name)
            predicted = forecast_df["q50"].values[:len(actual_window)]

            actuals = actual_window.values[:len(predicted)]
            mask = actuals != 0
            if mask.any():
                mape = np.mean(np.abs((actuals[mask] - predicted[mask]) / actuals[mask])) * 100
                errors.append(mape)

            dir_acc = calculate_directional_accuracy(actuals, predicted)
            if dir_acc > 0:
                directional_accuracy_scores.append(dir_acc)

            actual_dir = np.diff(actuals)
            pred_dir = np.diff(predicted)
            direction_correct_count += np.sum((actual_dir > 0) & (pred_dir > 0)) + np.sum((actual_dir < 0) & (pred_dir < 0))
            direction_total += len(actual_dir)
            windows += 1
        except Exception:
            pass

    if not errors or direction_total == 0:
        return {"mape": None, "directional_accuracy": None, "note": "Walk-forward backtest produced no valid samples"}

    directional_accuracy = float(np.mean(directional_accuracy_scores)) if directional_accuracy_scores else None

    return {
        "mape": float(np.mean(errors)),
        "directional_accuracy": directional_accuracy,
        "windows": windows,
        "samples": len(errors),
        "note": f"Walk-forward out-of-sample over {windows} expanding windows (horizon={horizon})"
    }


def build_gbdt_features(series: pd.Series, indicators: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"close": series})
    df = df.join(indicators, how="left")
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
    if "sma_20" in df.columns:
        df["price_vs_sma20"] = df["close"] / df["sma_20"] - 1
    else:
        df["price_vs_sma20"] = 0.0
    df = df.dropna()
    return df


def train_gbdt_models(df: pd.DataFrame, horizon: int = 1):
    if df is None or df.empty:
        return None, None, None, [], None

    feature_cols = [c for c in ["rsi", "sma_20", "atr", "momentum_rsi", "volatility_atr",
                                "return_1d", "return_5d", "volatility_20d", "price_vs_sma20"] if c in df.columns]
    if not feature_cols:
        return None, None, None, [], None

    X = df[feature_cols].values
    y = df["close"].shift(-horizon).dropna()
    if len(y) == 0:
        return None, None, None, feature_cols, None
    X = X[:len(y)]

    if len(X) < 10:
        return None, None, None, feature_cols, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lgbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    lgbm.fit(X_scaled, y)

    xgboost = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0)
    xgboost.fit(X_scaled, y)

    gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    gbdt.fit(X_scaled, y)

    return lgbm, xgboost, gbdt, feature_cols, scaler


def forecast_gbdt(models_and_scaler, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    lgbm, xgboost, gbdt, feature_cols, scaler = models_and_scaler
    if lgbm is None or scaler is None or df is None or df.empty or len(df) < horizon:
        return pd.DataFrame({"q10": [np.nan], "q50": [np.nan], "q90": [np.nan]})

    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame({"q10": [np.nan], "q50": [np.nan], "q90": [np.nan]})

    X = df[available_cols].values
    X_scaled = scaler.transform(X[-horizon:])

    preds = []
    for _ in range(horizon):
        p_lgbm = lgbm.predict(X_scaled)[0]
        p_xgb = xgboost.predict(X_scaled)[0]
        p_gbdt = gbdt.predict(X_scaled)[0]
        blended = 0.4 * p_lgbm + 0.4 * p_xgb + 0.2 * p_gbdt
        preds.append(blended)
        X_scaled = np.roll(X_scaled, -1, axis=0)
        X_scaled[-1, :] = X_scaled[-2, :] + np.random.normal(0, 0.01, size=X_scaled.shape[1])

    preds = np.array(preds)
    residuals = np.abs(df["close"].values[1:] - df["close"].values[:-1])
    band = np.percentile(residuals, 80) if len(residuals) > 0 else np.std(df["close"].values) * 0.1

    return pd.DataFrame({
        "q10": preds - 1.28 * band,
        "q50": preds,
        "q90": preds + 1.28 * band,
    })


def ensemble_forecast(chronos_df: pd.DataFrame, gbdt_df: pd.DataFrame, method: str = "blend", arima_df: pd.DataFrame = None) -> pd.DataFrame:
    if method == "chronos_only":
        return chronos_df
    elif method == "gbdt_only":
        return gbdt_df if gbdt_df is not None and not gbdt_df.isna().all().all() else chronos_df
    elif method == "hybrid":
        ensemble = HybridTimeSeriesEnsemble(chronos_weight=0.6, arima_weight=0.2, gbdt_weight=0.2)
        if arima_df is None:
            arima_df = pd.DataFrame({"q10": chronos_df["q10"].values, "q50": chronos_df["q50"].values, "q90": chronos_df["q90"].values})
        return ensemble.blend(chronos_df, arima_df, gbdt_df)
    else:
        if gbdt_df.isna().all().all():
            return chronos_df
        blended = 0.6 * chronos_df["q50"].values + 0.4 * gbdt_df["q50"].values
        q10 = 0.6 * chronos_df["q10"].values + 0.4 * gbdt_df["q10"].values
        q90 = 0.6 * chronos_df["q90"].values + 0.4 * gbdt_df["q90"].values
        return pd.DataFrame({"q10": q10, "q50": blended, "q90": q90})


def generate_signals(history: pd.Series, forecast_df: pd.DataFrame, stop_loss_pct: float = 2.0, atr_multiplier: float = 2.0, kelly_fraction: float = 0.5, min_risk_reward: float = 2.0, max_kelly_pct: float = 0.05) -> dict:
    current_price = float(history.iloc[-1])
    median_forecast = float(forecast_df["q50"].iloc[0])
    upper_bound = float(forecast_df["q90"].iloc[0])
    lower_bound = float(forecast_df["q10"].iloc[0])
    forecast_change_pct = ((median_forecast - current_price) / current_price) * 100

    indicators = calculate_indicators(history)
    atr = float(indicators["atr"].iloc[-1]) if "atr" in indicators.columns and not indicators["atr"].isna().iloc[-1] else current_price * 0.02
    atr_stop = current_price - atr_multiplier * atr
    percentile_stop = lower_bound

    stop_loss = max(atr_stop, percentile_stop)
    risk_reward = None
    if median_forecast > current_price:
        effective_stop = min(stop_loss, current_price * 0.99) if stop_loss >= current_price else stop_loss
        risk_reward = (median_forecast - current_price) / (current_price - effective_stop) if (current_price - effective_stop) > 0 else None
    elif median_forecast < current_price:
        effective_stop = max(stop_loss, current_price * 1.01) if stop_loss <= current_price else stop_loss
        risk_reward = (current_price - median_forecast) / (effective_stop - current_price) if (effective_stop - current_price) > 0 else None

    confidence = max(0.0, min(1.0, 1.0 - (upper_bound - lower_bound) / current_price))
    win_prob = confidence
    loss_prob = 1.0 - win_prob

    stop_distance_pct = (current_price - stop_loss) / current_price if current_price > 0 else 0.0
    vol_adjusted_win_prob = win_prob * min(1.0, stop_distance_pct / 0.02)

    b = risk_reward if risk_reward and risk_reward > 0 else 0.5
    kelly = (vol_adjusted_win_prob * b - loss_prob) / b if b > 0 else 0.0
    kelly = max(0.0, min(1.0, kelly)) * kelly_fraction
    kelly = min(kelly, max_kelly_pct)

    signal = "HOLD"
    reason = "No strong directional edge detected."

    if current_price > upper_bound:
        signal = "ABNORMAL MOMENTUM - DO NOT TRADE"
        reason = (
            f"Current price ({current_price:.2f}) has breached the 90th percentile "
            f"upper bound ({upper_bound:.2f}). Possible spike. "
            f"Treat as WARNING regime; no position sizing applied."
        )
        kelly = 0.0
        risk_reward = None
    elif forecast_change_pct >= 3.0 and lower_bound >= stop_loss:
        if risk_reward is not None and risk_reward < min_risk_reward:
            signal = "HOLD / NO TRADE"
            reason = (
                f"Median forecast is {forecast_change_pct:.2f}% above current price, "
                f"but risk-reward ({risk_reward:.2f}) is below minimum threshold ({min_risk_reward:.2f}). "
                f"Setup rejected."
            )
        else:
            signal = "BUY"
            reason = (
                f"Median forecast is {forecast_change_pct:.2f}% above current price "
                f"and lower bound ({lower_bound:.2f}) is above stop-loss ({stop_loss:.2f}). "
                f"Risk-reward is {risk_reward:.2f}. Volatility-adjusted Kelly: {kelly*100:.2f}%."
            )
    elif forecast_change_pct <= -3.0:
        if risk_reward is not None and risk_reward < min_risk_reward:
            signal = "HOLD / NO TRADE"
            reason = (
                f"Median forecast is {forecast_change_pct:.2f}% below current price, "
                f"but risk-reward ({risk_reward:.2f}) is below minimum threshold ({min_risk_reward:.2f}). "
                f"Setup rejected."
            )
        else:
            signal = "SELL"
            reason = (
                f"Median forecast is {forecast_change_pct:.2f}% below current price. "
                f"Downside pressure expected. Consider short futures or buying puts. "
                f"Risk-reward is {risk_reward:.2f}."
            )

    return {
        "current_price": current_price,
        "median_forecast": median_forecast,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "forecast_change_pct": forecast_change_pct,
        "atr": atr,
        "atr_stop": atr_stop,
        "percentile_stop": percentile_stop,
        "stop_loss": stop_loss,
        "risk_reward": risk_reward,
        "confidence": confidence,
        "win_prob": win_prob,
        "vol_adjusted_win_prob": vol_adjusted_win_prob,
        "loss_prob": loss_prob,
        "stop_distance_pct": stop_distance_pct,
        "kelly_fraction": kelly,
        "max_kelly_pct": max_kelly_pct,
        "signal": signal,
        "reason": reason,
    }


def plot_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    symbol: str = TARGET_SYMBOL,
    save_path: str = "forecast_plot.png",
):
    last_date = history.index[-1]
    trading_days = get_nse_trading_days(last_date, last_date + timedelta(days=365))
    forecast_dates = trading_days[:len(forecast_df)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.index, history.values, label="Historical Close", color="#1f77b4", linewidth=1.5)
    ax.plot(forecast_dates, forecast_df["q50"].values, label="Median Forecast", color="#ff7f0e", linewidth=2)
    ax.fill_between(forecast_dates, forecast_df["q10"].values, forecast_df["q90"].values, color="#ff7f0e", alpha=0.25, label="10th-90th Percentile")

    ax.set_title(f"{symbol} Daily Close + {len(forecast_df)}-Day Probabilistic Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info("Plot saved to %s", save_path)


def main():
    logger.info("Pipeline started for symbol: %s", TARGET_SYMBOL)

    df = fetch_icici_data(TARGET_SYMBOL)
    if df is None:
        df = generate_mock_data(TARGET_SYMBOL)

    series = clean_time_series(df)

    forecast_df = run_chronos_forecast(series)

    last_date = series.index[-1]
    trading_days = get_nse_trading_days(last_date, last_date + timedelta(days=365))
    forecast_dates = trading_days[:len(forecast_df)]

    result_df = pd.DataFrame({
        "forecast_date": forecast_dates,
        "q10": forecast_df["q10"].values,
        "q50": forecast_df["q50"].values,
        "q90": forecast_df["q90"].values,
    })
    csv_path = "forecast_results.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info("Forecast CSV saved to %s", csv_path)

    plot_forecast(series, forecast_df)
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
