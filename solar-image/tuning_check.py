"""Validate the tuning CSVs in /tuning against the installed AITER's schemas.

    python3 tuning_check.py <site-packages>
"""

import sys

import pandas as pd

SITE = sys.argv[1]
T = "/tuning"


def fail(msg):
    sys.exit(f"tuning: {msg}")


def check(name, stock_csv, idx, targets):
    """Schema, nulls, duplicate index rows and the exact (gfx, cu_num) set."""
    d = pd.read_csv(f"{T}/{name}")
    stock = pd.read_csv(f"{SITE}/aiter/configs/{stock_csv}")
    if list(d.columns) != list(stock.columns):
        fail(f"{name}: schema differs from aiter/configs/{stock_csv}")
    if d.isna().any().any():
        fail(f"{name}: null cells")
    # AITER looks rows up by this index, so a duplicate would make it arbitrary.
    if d.duplicated(subset=idx).any():
        fail(f"{name}: duplicate index rows")
    got = set(map(tuple, d[["gfx", "cu_num"]].drop_duplicates().values.tolist()))
    if got != targets:
        fail(f"{name}: targets {sorted(got)}, want {sorted(targets)}")
    return d


# bf16: read at runtime (AITER_CONFIG_GEMM_BF16).
GIDX = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
LIBTYPES = {"asm", "flydsl", "hipblaslt", "skinny", "torch", "triton"}

BF16 = "bf16_tuned_gemms_250b_tp4.csv"
g = check(BF16, "bf16_tuned_gemm.csv", GIDX, {("gfx950", 256)})
if not set(g.libtype) <= LIBTYPES:
    fail(f"{BF16}: unknown libtype {sorted(set(g.libtype) - LIBTYPES)}")
print(
    f"gemm bf16 {BF16}: {len(g)} rows, {g[['N', 'K']].drop_duplicates().shape[0]} N/K pairs, "
    f"libtypes {g.libtype.value_counts().to_dict()}"
)

# a8w8: compiled into module_gemm_a8w8.so, and build_tune_dict keeps only rows
# for the build target, so a row for another (gfx, cu_num) would be silently
# dropped rather than fail.
A8W8 = "a8w8_tuned_gemms_35b_fp8_tp1.csv"
a = check(A8W8, "a8w8_tuned_gemm.csv", ["gfx", "cu_num", "M", "N", "K"], {("gfx950", 256)})
if set(a.q_dtype_w) != {"torch.float8_e4m3fn"}:
    fail(f"{A8W8}: q_dtype_w {sorted(set(a.q_dtype_w))}")
print(f"gemm a8w8 {A8W8}: {len(a)} rows, {a[['N', 'K']].drop_duplicates().shape[0]} N/K pairs")
