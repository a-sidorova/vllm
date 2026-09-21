#!/usr/bin/env bash
# Regenerate the two source trees the Dockerfile consumes, from git pins.
#
#   vllm-src/       engine branch, exported with `git archive`
#   aiter-overlay/  the six router files, exported from the public aiter branch
#
# Both are pinned by commit and checked against recorded md5s, so a re-run on
# another host produces byte-identical inputs or fails loudly.
#
#   ./fetch-sources.sh && docker build -t vllm-solar:0908-combined .
set -euo pipefail
cd "$(dirname "$0")"

# --- pins ------------------------------------------------------------------
VLLM_REMOTE=https://github.com/UpstageAI/upstage-vllm-amd
VLLM_REF=sshliapn/features/solar_35b_model_optimizations_baseline
# The branch tip carries this directory; the image is built from its parent,
# which is the engine source proper.
VLLM_COMMIT=8665505227ad4db9042ef55795e435a317ed8af6

AITER_REMOTE=https://github.com/sshlyapn/aiter.git
AITER_REF=sshliapn/features/fused_router_v0.25_vllm_port
AITER_COMMIT=7fb161ea1

# The overlay, and the md5 each file must have. These are the exact bytes the
# 2026-09-17 image was built from; see README "Build record".
OVERLAY_FILES=(
  "aiter/fused_moe.py:b1bd2587081695ef2e25bb42f81efb0d"
  "aiter/__init__.py:c67a8d22d27beed0d671e4bbee71e71c"
  "aiter/ops/fused_moe_router.py:8afd18e4d0e958586a7b5266320b0222"
  "aiter/jit/optCompilerConfig.json:d104a47acd966c3c78b984ba488c9fa5"
  "csrc/kernels/fused_moe_router.cu:0e9eb508f087ad4286155037b9c16876"
  "csrc/py_itfs_cu/fused_moe_router_entry.cu:85e5e798ff11ef1ba44373742098397a"
)

# --- engine ----------------------------------------------------------------
# This directory ships inside the engine repo on $VLLM_REF, and $VLLM_COMMIT is
# an ancestor of it, so a normal clone already has the source -- no network.
# The fetch is only a fallback for a shallow clone, and needs read access to
# the private remote.
echo "==> vllm-src/  $VLLM_COMMIT"
VLLM_GIT=${VLLM_GIT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
if ! git -C "$VLLM_GIT" cat-file -e "$VLLM_COMMIT^{commit}" 2>/dev/null; then
    echo "    $VLLM_COMMIT not in $VLLM_GIT (shallow clone?); fetching it"
    # Fetch the commit itself, not the ref: on a shallow clone fetching the ref
    # just re-gets the tip and leaves the ancestor missing. --unshallow is
    # needed because a shallow repo rejects a fetch that deepens it otherwise.
    depth_arg=()
    [ "$(git -C "$VLLM_GIT" rev-parse --is-shallow-repository)" = true ] &&
        depth_arg=(--unshallow)
    git -C "$VLLM_GIT" fetch -q "${depth_arg[@]}" "$VLLM_REMOTE" "$VLLM_COMMIT" 2>/dev/null ||
        git -C "$VLLM_GIT" fetch -q "${depth_arg[@]}" "$VLLM_REMOTE" "$VLLM_REF" || true
    git -C "$VLLM_GIT" cat-file -e "$VLLM_COMMIT^{commit}" 2>/dev/null || {
        echo "could not obtain $VLLM_COMMIT from $VLLM_REMOTE." >&2
        echo "Re-clone $VLLM_REMOTE at $VLLM_REF without --depth, or point" >&2
        echo "VLLM_GIT at a full checkout that has the commit." >&2
        exit 1
    }
fi
rm -rf vllm-src && mkdir vllm-src
git -C "$VLLM_GIT" archive "$VLLM_COMMIT" | tar -x -C vllm-src

# --- router overlay --------------------------------------------------------
echo "==> aiter-overlay/  $AITER_COMMIT"
AITER_TMP=$(mktemp -d)
trap 'rm -rf "$AITER_TMP"' EXIT
git -C "$AITER_TMP" init -q
git -C "$AITER_TMP" fetch -q --depth 1 "$AITER_REMOTE" "$AITER_REF"

fail=0
for entry in "${OVERLAY_FILES[@]}"; do
    f=${entry%:*}; want=${entry#*:}
    mkdir -p "aiter-overlay/$(dirname "$f")"
    git -C "$AITER_TMP" show "FETCH_HEAD:$f" > "aiter-overlay/$f"
    got=$(md5sum "aiter-overlay/$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        printf '    %-45s ok\n' "$f"
    else
        printf '    %-45s MD5 MISMATCH want=%s got=%s\n' "$f" "$want" "$got"
        fail=1
    fi
done
[ "$fail" -eq 0 ] || {
    echo "aiter branch has moved since the image was built; update the pins" >&2
    exit 1
}

# --- tuning ----------------------------------------------------------------
# These ship with the repo rather than being fetched, but they are build inputs
# like any other, so pin them too.
echo "==> tuning/"
TUNING_FILES=(
  "fmoe_tuned_mxfp4_decode_solar.csv:f0fa6ab4172f19e20d6db66e459893e5"
  "aiter_gemm_tuning.csv:ead0bff9f7f86962bc469fa9e75c1846"
)
for entry in "${TUNING_FILES[@]}"; do
    f=${entry%:*}; want=${entry#*:}
    [ -f "tuning/$f" ] || { echo "    tuning/$f MISSING" >&2; exit 1; }
    got=$(md5sum "tuning/$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        printf '    %-45s ok\n' "$f"
    else
        printf '    %-45s MD5 MISMATCH want=%s got=%s\n' "$f" "$want" "$got" >&2
        exit 1
    fi
done

echo
echo "Sources ready. Next:  docker build -t vllm-solar:0908-combined ."
