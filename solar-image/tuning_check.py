"""Validate the tuning CSV in /tuning against the installed AITER's schema.

    python3 tuning_check.py <site-packages>
"""

import sys

import pandas as pd

SITE = sys.argv[1]
BF16 = "/tuning/bf16_tuned_gemms_250b_tp4.csv"


def fail(msg):
    sys.exit(f"tuning: {msg}")


# AITER looks rows up by this index, so a duplicate would make it arbitrary.
GIDX = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
LIBTYPES = {"asm", "flydsl", "hipblaslt", "skinny", "torch", "triton"}

g = pd.read_csv(BF16)
stock = pd.read_csv(f"{SITE}/aiter/configs/bf16_tuned_gemm.csv")
if list(g.columns) != list(stock.columns):
    fail("bf16_tuned_gemms_250b_tp4.csv: schema differs from aiter/configs")
if g.isna().any().any():
    fail("bf16_tuned_gemms_250b_tp4.csv: null cells")
if g.duplicated(subset=GIDX).any():
    fail("bf16_tuned_gemms_250b_tp4.csv: duplicate index rows")
targets = set(map(tuple, g[["gfx", "cu_num"]].drop_duplicates().values.tolist()))
if targets != {("gfx950", 256)}:
    fail(f"bf16_tuned_gemms_250b_tp4.csv: targets {sorted(targets)}, want gfx950/256 only")
if not set(g.libtype) <= LIBTYPES:
    fail(f"bf16_tuned_gemms_250b_tp4.csv: unknown libtype {sorted(set(g.libtype) - LIBTYPES)}")
print(
    f"gemm bf16: {len(g)} rows, {g[['N', 'K']].drop_duplicates().shape[0]} N/K pairs, "
    f"libtypes {g.libtype.value_counts().to_dict()}"
)
