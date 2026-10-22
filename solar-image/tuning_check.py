"""Validate the tuning CSVs in /tuning against the installed AITER's schemas.

    python3 tuning_check.py <site-packages>
"""

import glob
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

# fmoe: linked into aiter/configs/model_configs/, so AITER merges it with the
# stock tuned_fmoe.csv and model_configs/*tuned_fmoe*.csv at startup. That merge
# raises on a shape present in two tables (and rewrites them on disk), and the
# lookup keeps the first of equal keys, so no stock row may share a key with it.
# The tuner's schema is AITER's plus gfx/xbf16/flat/nt and minus _tag; the merge
# fills the difference, so only the stock columns are required.
FIDX = [
    "cu_num", "token", "model_dim", "inter_dim", "expert", "topk", "act_type",
    "dtype", "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1", "doweight_stage1",
]
FMOE = "a4w4_tuned_fmoe_250b_tp4.csv"
f = pd.read_csv(f"{T}/{FMOE}")
stock_cols = set(pd.read_csv(f"{SITE}/aiter/configs/tuned_fmoe.csv", nrows=0).columns)
missing = stock_cols - {"_tag"} - set(f.columns)
if missing:
    fail(f"{FMOE}: missing columns {sorted(missing)}")
if f.isna().any().any():
    fail(f"{FMOE}: null cells")
if f.duplicated(subset=["gfx"] + FIDX).any():
    fail(f"{FMOE}: duplicate index rows")
got = set(map(tuple, f[["gfx", "cu_num"]].drop_duplicates().values.tolist()))
if got != {("gfx950", 256)}:
    fail(f"{FMOE}: targets {sorted(got)}, want [('gfx950', 256)]")
if set(zip(f.q_dtype_w, f.q_type)) != {("torch.float4_e2m1fn_x2", "QuantType.per_1x32")}:
    fail(f"{FMOE}: weights are not MXFP4 (q_dtype_w, q_type {sorted(set(zip(f.q_dtype_w, f.q_type)))})")
if not (
    f.kernelName1.str.startswith("flydsl_moe1_").all()
    and f.kernelName2.str.startswith("flydsl_moe2_layout_").all()
):
    fail(f"{FMOE}: kernels other than flydsl_moe1_ / flydsl_moe2_layout_")

mine = set(map(tuple, f[FIDX].values.tolist()))
stock = [f"{SITE}/aiter/configs/tuned_fmoe.csv"] + sorted(
    p
    for p in glob.glob(f"{SITE}/aiter/configs/model_configs/*tuned_fmoe*.csv")
    if "untuned" not in p.rsplit("/", 1)[1] and not p.endswith(f"/{FMOE}")
)
for p in stock:
    clash = mine & set(map(tuple, pd.read_csv(p)[FIDX].values.tolist()))
    if clash:
        fail(f"{FMOE}: {len(clash)} shapes also in {p}, e.g. {sorted(clash)[0]}")
print(
    f"fmoe {FMOE}: {len(f)} rows, (expert, topk) "
    f"{sorted(map(tuple, f[['expert', 'topk']].drop_duplicates().values.tolist()))}, "
    f"no key shared with {len(stock)} stock tables"
)
