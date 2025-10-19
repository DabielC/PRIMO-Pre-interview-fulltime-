"""Synthetic order data generator aligned with the exploratory profile in `explore.ipynb`."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# --- Configuration tuned from the exploratory analysis ---
NUM_RECORDS = 10_000_000
NUM_CUSTOMERS = 100_000
BRANDS = ["ADIDAS", "NIKE", "PUMA", "REEBOK", "NEWBALANCE"]
BRANCHES = ["BANGKOK", "CHIANGMAI", "PHUKET", "KHONKAEN", "HATYAI"]
PRODUCT_TYPES = ["SHOE", "PANT", "BAG", "CAP", "SOCK"]

BRAND_WEIGHTS: Dict[str, float] = {
    "ADIDAS": 0.199386,
    "NIKE": 0.200428,
    "PUMA": 0.200078,
    "REEBOK": 0.199782,
    "NEWBALANCE": 0.200326,
}
BRANCH_WEIGHTS: Dict[str, float] = {
    "BANGKOK": 0.200184,
    "CHIANGMAI": 0.199790,
    "PHUKET": 0.200133,
    "KHONKAEN": 0.200827,
    "HATYAI": 0.199066,
}
PRODUCT_TYPE_WEIGHTS: Dict[str, float] = {
    "SHOE": 0.200030,
    "PANT": 0.200207,
    "BAG": 0.199893,
    "CAP": 0.199485,
    "SOCK": 0.200385,
}
QUANTITY_LEVELS = np.array([1, 2, 3, 4, 5], dtype=np.int8)
QUANTITY_WEIGHTS = np.array([0.200000, 0.200413, 0.200027, 0.199522, 0.200038], dtype=float)

# Order-level price spans by product type (kept within 500–10,000 THB range)
PRICE_RANGES: Dict[str, Tuple[int, int]] = {
    "SHOE": (3200, 9500),
    "PANT": (2200, 7800),
    "BAG": (1800, 8200),
    "CAP": (500, 2600),
    "SOCK": (500, 1900),
}

START_DATE = pd.Timestamp("2024-01-01 00:00:00")
END_DATE = pd.Timestamp("2024-12-31 23:59:59")
OUTPUT_FILE = Path("../large_order_data_2024/large_order_data_10M_2024.parquet")
CHUNK_SIZE = 500_000
SEED = 42
PARQUET_COMPRESSION = "snappy"


def _build_customer_pool() -> np.ndarray:
    return np.array([f"CUST{i:05d}" for i in range(1, NUM_CUSTOMERS + 1)], dtype=object)


def _build_sku_catalog(per_type: int = 60) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    catalog = {}
    catalog_types = {}
    for brand in BRANDS:
        prefix = brand[:2].upper()
        skus = []
        sku_types = []
        for product_type in PRODUCT_TYPES:
            for idx in range(1, per_type + 1):
                skus.append(f"{prefix}-{product_type}-{idx:03d}")
                sku_types.append(product_type)
        catalog[brand] = np.array(skus, dtype=object)
        catalog_types[brand] = np.array(sku_types, dtype=object)
    return catalog, catalog_types


def _generate_chunk(
    start_index: int,
    size: int,
    rng: np.random.Generator,
    customers: np.ndarray,
    sku_catalog: Dict[str, np.ndarray],
    sku_types: Dict[str, np.ndarray],
) -> pd.DataFrame:
    order_indexes = np.arange(start_index, start_index + size, dtype=np.int64)
    order_numbers = np.char.add("ORD", np.char.zfill(order_indexes.astype(str), 7))

    branch_probs = np.array([BRANCH_WEIGHTS[b] for b in BRANCHES], dtype=float)
    brand_probs = np.array([BRAND_WEIGHTS[b] for b in BRANDS], dtype=float)
    product_probs = np.array([PRODUCT_TYPE_WEIGHTS[p] for p in PRODUCT_TYPES], dtype=float)

    branches = rng.choice(BRANCHES, size=size, p=branch_probs)
    brands = rng.choice(BRANDS, size=size, p=brand_probs)
    customers_selected = rng.choice(customers, size=size)
    quantities = rng.choice(QUANTITY_LEVELS, size=size, p=QUANTITY_WEIGHTS)

    product_types = np.empty(size, dtype=object)
    skus = np.empty(size, dtype=object)
    for brand in BRANDS:
        mask = brands == brand
        if not mask.any():
            continue
        sku_pool = sku_catalog[brand]
        type_pool = sku_types[brand]
        idxs = rng.integers(0, len(sku_pool), size=mask.sum())
        skus[mask] = sku_pool[idxs]
        product_types[mask] = type_pool[idxs]

    # Add light variability to product mix so monthly trends vary
    drift_types = rng.choice(PRODUCT_TYPES, size=size, p=product_probs)
    replace_mask = rng.random(size) < 0.05
    product_types[replace_mask] = drift_types[replace_mask]
    for brand in BRANDS:
        mask = (brands == brand) & replace_mask
        if not mask.any():
            continue
        replacement_types = product_types[mask]
        sku_pool = sku_catalog[brand]
        type_pool = sku_types[brand]
        matches = {ptype: np.flatnonzero(type_pool == ptype) for ptype in PRODUCT_TYPES}
        idxs = np.empty(mask.sum(), dtype=int)
        for ptype, idx_positions in matches.items():
            type_mask = replacement_types == ptype
            if not type_mask.any():
                continue
            idxs[type_mask] = rng.choice(idx_positions, size=type_mask.sum())
        skus[mask] = sku_pool[idxs]

    amounts = np.empty(size, dtype=np.int32)
    for product_type, (low, high) in PRICE_RANGES.items():
        type_mask = product_types == product_type
        if not type_mask.any():
            continue
        amounts[type_mask] = rng.integers(low, high + 1, size=type_mask.sum())

    seconds_range = int((END_DATE - START_DATE).total_seconds()) + 1
    random_seconds = rng.integers(0, seconds_range, size=size, dtype=np.int64)
    transaction_datetimes = START_DATE + pd.to_timedelta(random_seconds, unit="s")

    return pd.DataFrame(
        {
            "order_no": order_numbers,
            "amount": amounts,
            "customer_no": customers_selected,
            "branch": branches,
            "brand": brands,
            "sku": skus,
            "quantity": quantities,
            "transaction_datetime": transaction_datetimes,
        }
    )


def _optimize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()
    optimized["amount"] = optimized["amount"].astype("int32")
    optimized["quantity"] = optimized["quantity"].astype("int8")
    optimized["customer_no"] = optimized["customer_no"].astype("string")
    optimized["branch"] = optimized["branch"].astype(
        pd.CategoricalDtype(categories=BRANCHES)
    )
    optimized["brand"] = optimized["brand"].astype(
        pd.CategoricalDtype(categories=BRANDS)
    )
    optimized["sku"] = optimized["sku"].astype("category")
    optimized["transaction_datetime"] = optimized["transaction_datetime"].astype("datetime64[ns]")
    return optimized


def generate_data() -> None:
    rng = np.random.default_rng(SEED)
    customers = _build_customer_pool()
    sku_catalog, sku_types = _build_sku_catalog()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_chunks = (NUM_RECORDS + CHUNK_SIZE - 1) // CHUNK_SIZE
    writer: pq.ParquetWriter | None = None

    with tqdm(total=total_chunks, desc="Generating data", unit="chunk") as progress:
        try:
            for start in range(0, NUM_RECORDS, CHUNK_SIZE):
                chunk_size = min(CHUNK_SIZE, NUM_RECORDS - start)
                chunk_df = _generate_chunk(start, chunk_size, rng, customers, sku_catalog, sku_types)
                chunk_df = _optimize_chunk(chunk_df)
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        OUTPUT_FILE,
                        table.schema,
                        compression=PARQUET_COMPRESSION,
                    )

                writer.write_table(table)
                total_written += chunk_size
                progress.update(1)
                progress.set_postfix({"rows": f"{total_written:,}/{NUM_RECORDS:,}"})
        finally:
            if writer is not None:
                writer.close()

    print(f"Finished generating {NUM_RECORDS:,} records at {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_data()
