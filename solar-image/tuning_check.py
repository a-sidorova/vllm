"""Validate the tuning CSVs in /tuning and write the merged fmoe table.

AITER_CONFIG_FMOE names one table for the process, so the MXFP4 and FP8 MoE
tables are concatenated here, once. AITER can merge a colon-separated list
itself, but it redoes that on every start, and on a duplicate it rewrites the
source files in place.

The merge is textual (header + both bodies) so no value is reformatted, e.g. an
int column read back as float because another file has a null in it.

    python3 tuning_check.py <site-packages>
"""

import sys

import pandas as pd

SITE = sys.argv[1]
T = "/tuning"

FIDX = [
    "gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
    "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1",
    "doweight_stage1",
]  # fmt: skip

# (file, {(expert, topk)}). `expert`, `topk` and the quant columns are all
# AITER index columns, so the two models' rows cannot collide.
FMOE = (
    ("fmoe_tuned_mxfp4_decode_solar.csv", {(80, 7), (81, 8)}),
    ("fmoe_tuned_35b_fp8_tp1.csv", {(128, 4)}),
)
FMOE_MERGED = "fmoe_tuned_solar.csv"


def fail(msg):
    sys.exit(f"tuning: {msg}")


def no_dup_index(name, d, idx):
    # Per _tag partition: the flydsl_fallback rows deliberately mirror the
    # tuned index.
    tag = d["_tag"].fillna("") if "_tag" in d.columns else pd.Series("", d.index)
    for t, part in d.groupby(tag):
        if part.duplicated(subset=idx).any():
            fail(f"{name}: duplicate index rows in partition {t or 'tuned'!r}")


frames, header, bodies = [], None, []
for name, shapes in FMOE:
    path = f"{T}/{name}"
    d = pd.read_csv(path)
    if not set(FIDX) <= set(d.columns):
        fail(f"{name}: missing index columns {sorted(set(FIDX) - set(d.columns))}")
    got = set(map(tuple, d[["expert", "topk"]].drop_duplicates().values.tolist()))
    if got != shapes:
        fail(f"{name}: shapes {sorted(got)}, want {sorted(shapes)}")
    no_dup_index(name, d, FIDX)
    print(f"fmoe {name}: {len(d)} rows, shapes {sorted(got)}")

    with open(path) as f:
        lines = f.read().splitlines()
    if header is None:
        header = lines[0]
    elif lines[0] != header:
        fail(f"{name}: header differs from {FMOE[0][0]}")
    bodies += [ln for ln in lines[1:] if ln.strip()]
    frames.append(d)

out = f"{T}/{FMOE_MERGED}"
with open(out, "w") as f:
    f.write("\n".join([header] + bodies) + "\n")
m = pd.read_csv(out)
if len(m) != sum(len(d) for d in frames):
    fail(f"{FMOE_MERGED}: {len(m)} rows after merge")
no_dup_index(FMOE_MERGED, m, FIDX)
print(f"fmoe {FMOE_MERGED}: {len(m)} rows")

GIDX = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
g = pd.read_csv(f"{T}/aiter_gemm_tuning.csv")
stock = pd.read_csv(f"{SITE}/aiter/configs/bf16_tuned_gemm.csv")
if list(g.columns) != list(stock.columns):
    fail("aiter_gemm_tuning.csv: schema differs from aiter/configs")
if g.isna().any().any():
    fail("aiter_gemm_tuning.csv: null cells")
if g.duplicated(subset=GIDX).any():
    fail("aiter_gemm_tuning.csv: duplicate index rows")
if "gfx950" not in set(g.gfx) or 4096 not in set(g.K):
    fail("aiter_gemm_tuning.csv: no gfx950 K=4096 rows")
print(f"gemm bf16: {len(g)} rows, {g[['N', 'K']].drop_duplicates().shape[0]} N/K pairs")

# The a8w8 table is compiled into module_gemm_a8w8.so, and build_tune_dict
# keeps only rows for the build target, so a row for another (gfx, cu_num)
# would be silently dropped rather than fail.
AIDX = ["gfx", "cu_num", "M", "N", "K"]
a = pd.read_csv(f"{T}/a8w8_tuned_gemm_35b_fp8.csv")
stock = pd.read_csv(f"{SITE}/aiter/configs/a8w8_tuned_gemm.csv")
if list(a.columns) != list(stock.columns):
    fail("a8w8_tuned_gemm_35b_fp8.csv: schema differs from aiter/configs")
if a.isna().any().any():
    fail("a8w8_tuned_gemm_35b_fp8.csv: null cells")
if a.duplicated(subset=AIDX).any():
    fail("a8w8_tuned_gemm_35b_fp8.csv: duplicate index rows")
targets = set(map(tuple, a[["gfx", "cu_num"]].drop_duplicates().values.tolist()))
if targets != {("gfx950", 256)}:
    fail(f"a8w8_tuned_gemm_35b_fp8.csv: targets {sorted(targets)}, want gfx950/256 only")
if set(a.q_dtype_w) != {"torch.float8_e4m3fn"}:
    fail(f"a8w8_tuned_gemm_35b_fp8.csv: q_dtype_w {sorted(set(a.q_dtype_w))}")
print(f"gemm a8w8: {len(a)} rows, {a[['N', 'K']].drop_duplicates().shape[0]} N/K pairs")
