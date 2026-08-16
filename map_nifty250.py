#!/usr/bin/env python3
"""
Map NIFTY250 company names to ICICI Direct security master symbols.
Outputs:
  - nifty250_mapped.csv      : company name -> ICICI ShortName mapping
  - nifty250_unmapped.csv    : companies that could not be mapped
"""

import os
import sys
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE_DIR, "SecurityMaster (1)", "NSEScripMaster.txt")
NIFTY250_PATH = os.path.join(BASE_DIR, "SecurityMaster (1)", "nifty250.csv")
OUTPUT_MAPPED = os.path.join(BASE_DIR, "nifty250_mapped.csv")
OUTPUT_UNMAPPED = os.path.join(BASE_DIR, "nifty250_unmapped.csv")


def load_equity_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df[df["Series"].str.upper() == "EQ"].copy()
    df["__company_clean__"] = df["CompanyName"].str.upper().str.strip()
    df["__company_normalized__"] = df["__company_clean__"].apply(normalize_company)
    df["__shortname_clean__"] = df["ShortName"].str.upper().str.strip()
    return df


def load_nifty250(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["__company_clean__"] = df["Company Name"].str.upper().str.strip()
    df["__symbol_clean__"] = df["Symbol"].str.upper().str.strip()
    return df


def normalize_company(name: str) -> str:
    name = name.upper().strip()
    name = name.replace(".", "")
    name = name.replace(",", "")
    name = name.replace(" LTD", " LIMITED")
    name = name.replace(" LTD", " LIMITED")
    name = " ".join(name.split())
    return name


def map_symbols(nifty_df: pd.DataFrame, master_df: pd.DataFrame):
    master_indexed = master_df.set_index("__company_normalized__")
    master_by_symbol = master_df.set_index("__shortname_clean__")

    mapped = []
    unmapped = []

    for _, row in nifty_df.iterrows():
        company = row["__company_clean__"]
        nifty_symbol = row["__symbol_clean__"]
        normalized_company = normalize_company(company)

        if normalized_company in master_indexed.index:
            icici_row = master_indexed.loc[normalized_company]
            if isinstance(icici_row, pd.DataFrame):
                icici_row = icici_row.iloc[0]
            mapped.append({
                "company_name": row["Company Name"],
                "nifty_symbol": nifty_symbol,
                "icici_shortname": icici_row["ShortName"],
                "match_type": "company_normalized",
            })
            continue

        symbol_match = None
        if nifty_symbol in master_by_symbol.index:
            symbol_match = master_by_symbol.loc[nifty_symbol]
            if isinstance(symbol_match, pd.DataFrame):
                symbol_match = symbol_match.iloc[0]

        if symbol_match is not None:
            mapped.append({
                "company_name": row["Company Name"],
                "nifty_symbol": nifty_symbol,
                "icici_shortname": symbol_match["ShortName"],
                "match_type": "symbol_exact",
            })
            continue

        normalized_words = set(normalized_company.split())
        best_match = None
        best_score = 0

        for master_normalized in master_indexed.index:
            master_words = set(master_normalized.split())
            overlap = len(normalized_words & master_words)
            if overlap > best_score:
                best_score = overlap
                best_match = master_normalized

        if best_match and best_score >= 2:
            icici_row = master_indexed.loc[best_match]
            if isinstance(icici_row, pd.DataFrame):
                icici_row = icici_row.iloc[0]
            mapped.append({
                "company_name": row["Company Name"],
                "nifty_symbol": nifty_symbol,
                "icici_shortname": icici_row["ShortName"],
                "match_type": "company_word_overlap",
            })
            continue

        unmapped.append({
            "company_name": row["Company Name"],
            "nifty_symbol": nifty_symbol,
        })

    mapped_df = pd.DataFrame(mapped)
    unmapped_df = pd.DataFrame(unmapped)
    return mapped_df, unmapped_df


def main():
    if not os.path.exists(MASTER_PATH):
        logger.error("Security master not found: %s", MASTER_PATH)
        sys.exit(1)

    if not os.path.exists(NIFTY250_PATH):
        logger.error("NIFTY250 CSV not found: %s", NIFTY250_PATH)
        sys.exit(1)

    logger.info("Loading equity master...")
    master_df = load_equity_master(MASTER_PATH)
    logger.info("Loaded %d equity instruments", len(master_df))

    logger.info("Loading NIFTY250 list...")
    nifty_df = load_nifty250(NIFTY250_PATH)
    logger.info("Loaded %d NIFTY250 stocks", len(nifty_df))

    logger.info("Mapping symbols...")
    mapped_df, unmapped_df = map_symbols(nifty_df, master_df)

    mapped_df.to_csv(OUTPUT_MAPPED, index=False)
    unmapped_df.to_csv(OUTPUT_UNMAPPED, index=False)

    logger.info("Mapped %d stocks to ICICI symbols", len(mapped_df))
    logger.info("Unmapped %d stocks", len(unmapped_df))
    logger.info("Saved mapped list to %s", OUTPUT_MAPPED)
    logger.info("Saved unmapped list to %s", OUTPUT_UNMAPPED)

    print("\n" + "=" * 60)
    print("MAPPING COMPLETE")
    print(f"Total NIFTY250 stocks : {len(nifty_df)}")
    print(f"Mapped successfully    : {len(mapped_df)}")
    print(f"Unmapped               : {len(unmapped_df)}")
    print("=" * 60)

    if not unmapped_df.empty:
        print("\nUNMAPPED STOCKS:")
        for _, row in unmapped_df.iterrows():
            print(f"  {row['nifty_symbol']:12s}  {row['company_name']}")


if __name__ == "__main__":
    main()
