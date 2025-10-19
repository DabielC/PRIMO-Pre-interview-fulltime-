"""Find top 10 spenders per month from a large order dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# --- Configuration ---
INPUT_FILE = Path("../large_order_data_2024/large_order_data_10M_2024.parquet")
OUTPUT_FILE = Path("../large_order_data_2024/top_10_spenders_per_month.parquet")
CHUNK_SIZE = 1_000_000  # Tune based on available memory
PARQUET_COMPRESSION = "snappy"


def _load_chunked_summaries() -> List[pd.DataFrame]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found at {INPUT_FILE}. Run `generate_data.py` first."
        )

    summaries: List[pd.DataFrame] = []

    parquet_file = pq.ParquetFile(INPUT_FILE)
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
    total_batches = (
        (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE if total_rows is not None else None
    )

    batches = parquet_file.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["customer_no", "amount", "transaction_datetime"],
    )

    for batch in tqdm(batches, total=total_batches, desc="Processing batches", unit="batch"):
        chunk = batch.to_pandas()
        chunk["transaction_datetime"] = pd.to_datetime(chunk["transaction_datetime"])
        chunk["customer_no"] = chunk["customer_no"].astype("string")
        chunk["amount"] = chunk["amount"].astype("int64")
        chunk["year_month"] = chunk["transaction_datetime"].dt.to_period("M")
        grouped = (
            chunk.groupby(["year_month", "customer_no"], as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "total_spent"})
        )
        summaries.append(grouped)

    return summaries


def find_top_spenders() -> pd.DataFrame:
    summaries = _load_chunked_summaries()
    if not summaries:
        raise ValueError("No data processed; the input file appears to be empty.")

    combined = pd.concat(summaries, ignore_index=True)
    monthly_totals = (
        combined.groupby(["year_month", "customer_no"], as_index=False)["total_spent"]
        .sum()
        .sort_values(["year_month", "total_spent"], ascending=[True, False])
    )

    monthly_totals["rank"] = (
        monthly_totals.groupby("year_month")["total_spent"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top_spenders = (
        monthly_totals[monthly_totals["rank"] <= 10]
        .copy()
        .sort_values(["year_month", "rank"])
    )

    period_index = pd.PeriodIndex(top_spenders["year_month"], name="year_month")
    top_spenders["month"] = period_index.month
    top_spenders["month_name"] = period_index.strftime("%B")
    top_spenders["year_month"] = period_index.astype(str)

    top_spenders["rank"] = top_spenders["rank"].astype("int16")
    top_spenders["month"] = top_spenders["month"].astype("int8")
    top_spenders["month_name"] = top_spenders["month_name"].astype("string")
    top_spenders["customer_no"] = top_spenders["customer_no"].astype("string")
    top_spenders["total_spent"] = top_spenders["total_spent"].astype("int64")

    column_order = ["year_month", "month", "month_name", "rank", "customer_no", "total_spent"]
    return top_spenders[column_order]


def main() -> None:
    print(f"Starting top spender extraction from {INPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df = find_top_spenders()
    result_df.to_parquet(OUTPUT_FILE, index=False, compression=PARQUET_COMPRESSION)
    print(f"Top spenders saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
