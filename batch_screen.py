#!/usr/bin/env python3
"""
Batch NSE stock screener: loops through selected symbols and identifies
stocks with high expected directional move using Chronos + GBDT ensemble.

Outputs:
  - batch_results.csv      : all screened stocks
  - batch_top_movers.csv   : stocks with |expected_move| >= threshold
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

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
    generate_signals,
    TARGET_SYMBOL,
    FORECAST_HORIZON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SECURITY_MASTER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SecurityMaster (1)",
    "NSEScripMaster.txt",
)
DEFAULT_NIFTY250_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SecurityMaster (1)",
    "nifty250.csv",
)
DEFAULT_NIFTY250_MAPPED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nifty250_mapped.csv",
)

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_HORIZON = 30
DEFAULT_INTERVAL = "1day"
DEFAULT_MODEL = "amazon/chronos-t5-tiny"
DEFAULT_ENSEMBLE = "chronos_only"
DEFAULT_MOVE_THRESHOLD = 3.0
DEFAULT_MAX_WORKERS = 1


def load_security_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    column_map = {c: c.strip().strip('"') for c in df.columns}
    df.rename(columns=column_map, inplace=True)
    df["__search__"] = (
        df["ShortName"].str.upper().str.strip()
        + " "
        + df["CompanyName"].str.upper().str.strip()
        + " "
        + df["ExchangeCode"].str.upper().str.strip()
    )
    df["__display__"] = df["ShortName"].str.upper().str.strip()
    return df


def get_symbols_from_master(master_df: pd.DataFrame, limit: Optional[int] = None) -> List[str]:
    symbols = master_df["__display__"].dropna().unique().tolist()
    symbols = sorted(symbols)
    if limit:
        symbols = symbols[:limit]
    return symbols


def load_nifty250_symbols(path: str) -> List[str]:
    df = pd.read_csv(path, dtype=str)
    symbol_col = None
    for candidate in ["Symbol", "TradingSymbol", "ShortName"]:
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        raise ValueError(f"Could not find Symbol column in {path}. Columns: {df.columns.tolist()}")
    symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
    symbols = sorted([s for s in symbols if s])
    return symbols


def load_mapped_nifty250(path: str) -> List[str]:
    df = pd.read_csv(path, dtype=str)
    symbol_col = None
    for candidate in ["icici_shortname", "ICICIShortName", "ShortName", "symbol"]:
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        raise ValueError(f"Could not find symbol column in {path}. Columns: {df.columns.tolist()}")
    symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
    symbols = sorted([s for s in symbols if s])
    return symbols


def build_nifty250_to_icici_map(nifty250_path: str, master_df: pd.DataFrame) -> dict:
    nifty_df = pd.read_csv(nifty250_path, dtype=str)
    symbol_col = None
    for candidate in ["Symbol", "TradingSymbol", "ShortName"]:
        if candidate in nifty_df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        raise ValueError(f"Could not find Symbol column in {nifty250_path}. Columns: {nifty_df.columns.tolist()}")

    nifty_df = nifty_df.copy()
    nifty_df["__nifty_symbol__"] = nifty_df[symbol_col].str.upper().str.strip()
    nifty_df["__company_clean__"] = nifty_df["Company Name"].str.upper().str.strip()

    mapping = {}
    unmatched = []

    master_lookup = master_df.set_index("__display__")
    master_by_company = master_df.set_index(master_df["CompanyName"].str.upper().str.strip())

    for _, row in nifty_df.iterrows():
        nifty_sym = row["__nifty_symbol__"]
        company = row["__company_clean__"]

        if nifty_sym in master_lookup.index:
            mapping[nifty_sym] = nifty_sym
            continue

        matched = False
        for company_key in master_by_company.index:
            if company and company_key and (company in company_key or company_key in company):
                icici_sym = master_by_company.loc[company_key, "ShortName"]
                if isinstance(icici_sym, pd.Series):
                    icici_sym = icici_sym.iloc[0]
                mapping[nifty_sym] = str(icici_sym).strip().upper()
                matched = True
                break

        if not matched:
            mapping[nifty_sym] = nifty_sym
            unmatched.append(nifty_sym)

    if unmatched:
        logger.warning("Could not map %d NIFTY250 symbols to ICICI master: %s", len(unmatched), unmatched[:20])

    return mapping


def screen_stock(
    symbol: str,
    days: int,
    interval: str,
    horizon: int,
    model_name: str,
    ensemble_method: str,
) -> dict:
    try:
        df = fetch_icici_data(symbol, days=days, interval=interval)
        if df is None:
            df = generate_mock_data(symbol, days=days, interval=interval)
        series = clean_time_series(df, interval=interval)

        indicators = calculate_indicators(series, interval=interval)
        gbdt_data = build_gbdt_features(series, indicators)
        gbdt_models = train_gbdt_models(gbdt_data, horizon=horizon)
        gbdt_forecast = forecast_gbdt(gbdt_models, gbdt_data, horizon=horizon)
        chronos_forecast = run_chronos_forecast(series, horizon=horizon, model_name=model_name)
        forecast_df = ensemble_forecast(chronos_forecast, gbdt_forecast, method=ensemble_method)

        current_price = float(series.iloc[-1])
        median_forecast = float(forecast_df["q50"].iloc[0])
        forecast_change_pct = ((median_forecast - current_price) / current_price) * 100 if current_price != 0 else 0.0

        signals = generate_signals(series, forecast_df, min_risk_reward=2.0, max_kelly_pct=0.05)

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "median_forecast": round(median_forecast, 2),
            "expected_move_pct": round(forecast_change_pct, 2),
            "signal": signals["signal"],
            "risk_reward": round(signals["risk_reward"], 2) if signals["risk_reward"] is not None else None,
            "kelly_pct": round(signals["kelly_fraction"] * 100, 2),
            "status": "OK",
        }
    except Exception as e:
        logger.warning("Failed for %s: %s", symbol, e)
        return {
            "symbol": symbol,
            "current_price": None,
            "median_forecast": None,
            "expected_move_pct": None,
            "status": f"ERROR: {e}",
        }


def run_batch(
    symbols: List[str],
    days: int,
    interval: str,
    horizon: int,
    model_name: str,
    ensemble_method: str,
    move_threshold: float,
    output_dir: str,
) -> pd.DataFrame:
    results = []
    total = len(symbols)
    logger.info("Screening %d stocks...", total)

    for idx, symbol in enumerate(symbols, start=1):
        logger.info("[%d/%d] Screening %s...", idx, total, symbol)
        res = screen_stock(symbol, days, interval, horizon, model_name, ensemble_method)
        results.append(res)

        if idx % 10 == 0 or idx == total:
            partial_df = pd.DataFrame(results)
            partial_path = os.path.join(output_dir, "batch_results_partial.csv")
            partial_df.to_csv(partial_path, index=False)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="expected_move_pct", ascending=False, na_position="last")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Batch NSE stock directional move screener")
    parser.add_argument("--master", default=DEFAULT_SECURITY_MASTER, help="Path to NSEScripMaster.txt")
    parser.add_argument("--nifty250", default=DEFAULT_NIFTY250_CSV, help="Path to nifty250.csv")
    parser.add_argument("--mapped-csv", default=DEFAULT_NIFTY250_MAPPED, help="Path to pre-mapped nifty250_mapped.csv")
    parser.add_argument("--use-nifty250", action="store_true", help="Use nifty250.csv symbols instead of security master")
    parser.add_argument("--use-mapped", action="store_true", help="Use pre-mapped nifty250_mapped.csv for ICICI symbols")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stocks to screen")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Lookback days")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, choices=["1day", "15minute", "1minute"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chronos model name")
    parser.add_argument("--ensemble", default=DEFAULT_ENSEMBLE, choices=["blend", "chronos_only", "gbdt_only"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_MOVE_THRESHOLD, help="Min |expected_move_pct| to flag")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    if args.use_mapped:
        if not os.path.exists(args.mapped_csv):
            logger.error("Mapped NIFTY250 CSV not found: %s", args.mapped_csv)
            sys.exit(1)
        symbols = load_mapped_nifty250(args.mapped_csv)
        logger.info("Loaded %d ICICI-mapped symbols from %s", len(symbols), args.mapped_csv)
    elif args.use_nifty250:
        if not os.path.exists(args.nifty250):
            logger.error("NIFTY250 CSV not found: %s", args.nifty250)
            sys.exit(1)
        raw_symbols = load_nifty250_symbols(args.nifty250)
        if not os.path.exists(args.master):
            logger.error("Security master not found: %s", args.master)
            sys.exit(1)
        master_df = load_security_master(args.master)
        symbol_map = build_nifty250_to_icici_map(args.nifty250, master_df)
        symbols = [symbol_map.get(s, s) for s in raw_symbols]
        logger.info("Loaded %d NIFTY250 symbols and mapped to ICICI master", len(symbols))
    else:
        if not os.path.exists(args.master):
            logger.error("Security master not found: %s", args.master)
            sys.exit(1)
        master_df = load_security_master(args.master)
        symbols = get_symbols_from_master(master_df, limit=args.limit)
        logger.info("Loaded %d unique symbols from master", len(symbols))

    if args.limit and not args.use_nifty250:
        symbols = symbols[:args.limit]
    elif args.limit and args.use_nifty250:
        symbols = symbols[:args.limit]

    results_df = run_batch(
        symbols=symbols,
        days=args.days,
        interval=args.interval,
        horizon=args.horizon,
        model_name=args.model,
        ensemble_method=args.ensemble,
        move_threshold=args.threshold,
        output_dir=args.output_dir,
    )

    results_path = os.path.join(args.output_dir, "batch_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("Full results saved to %s", results_path)

    top_movers = results_df[
        (results_df["status"] == "OK")
        & (results_df["expected_move_pct"].notna())
        & (results_df["expected_move_pct"].abs() >= args.threshold)
    ].copy()

    top_movers_path = os.path.join(args.output_dir, "batch_top_movers.csv")
    top_movers.to_csv(top_movers_path, index=False)
    logger.info("Top movers (|move| >= %.1f%%) saved to %s", args.threshold, top_movers_path)

    print("\n" + "=" * 60)
    print(f"BATCH SCREENING COMPLETE")
    print(f"Total screened : {len(results_df)}")
    print(f"Successful     : {len(results_df[results_df['status'] == 'OK'])}")
    print(f"Failed         : {len(results_df[results_df['status'] != 'OK'])}")
    print(f"High-move flags: {len(top_movers)}")
    print("=" * 60)

    if not top_movers.empty:
        print("\nTOP BULLISH EXPECTED MOVES:")
        bullish = top_movers[top_movers["expected_move_pct"] > 0].head(10)
        for _, row in bullish.iterrows():
            print(f"  {row['symbol']:12s}  {row.get('signal',''):18s}  current={row['current_price']:.2f}  forecast={row['median_forecast']:.2f}  move={row['expected_move_pct']:+.2f}%")

        print("\nTOP BEARISH EXPECTED MOVES (Short Sell Candidates):")
        bearish = top_movers[top_movers["expected_move_pct"] < 0].tail(10).iloc[::-1]
        for _, row in bearish.iterrows():
            print(f"  {row['symbol']:12s}  {row.get('signal',''):18s}  current={row['current_price']:.2f}  forecast={row['median_forecast']:.2f}  move={row['expected_move_pct']:+.2f}%")
    else:
        print("\nNo stocks found with |expected move| >= {:.1f}%".format(args.threshold))

    print("\nOutput files:")
    print(f"  - {results_path}")
    print(f"  - {top_movers_path}")


if __name__ == "__main__":
    main()
