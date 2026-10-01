import torch, time
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe_router, fused_moe_router_supported
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight

dev = "cuda"
M = 64


def mxfp4(*shape):
    w = torch.randint(0, 255, (*shape[:-1], shape[-1] // 2),
                      dtype=torch.uint8, device=dev).view(dtypes.fp4x2)
    s = torch.randint(118, 126, (*shape[:-1], shape[-1] // 32),
                      dtype=torch.uint8, device=dev).view(dtypes.fp8_e8m0)
    return w, s


# Per-channel fp8 weights with fp32 scales, preshuffled as the Quark FP8 path
# loads them. Random bytes are not usable here: e4m3fn has NaN encodings.
def fp8_per_channel(*shape):
    w = torch.randn(shape, dtype=dtypes.bf16, device=dev) / 16
    wq, ws = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    wq = shuffle_weight(wq, layout=(16, 16))
    wq.is_shuffled = True
    return wq, ws


# (label, hidden K, inter N, E, topk, quant_type, weight maker)
CASES = (
    ("MXFP4 Solar-250B, 80 routed", 4096, 1280, 80, 7, QuantType.per_1x32, mxfp4),
    ("FP8   Solar-35B, 128 routed", 2048, 1024, 128, 4, QuantType.per_Token, fp8_per_channel),
)

for label, K, N, E, TOPK, qt, make_w in CASES:
    print(f"\n== {label}: M={M} K={K} N={N} E={E} topk={TOPK}")
    torch.manual_seed(0)
    hs = torch.randn(M, K, dtype=dtypes.bf16, device=dev)
    gate = torch.randn(M, E, dtype=dtypes.bf16, device=dev)
    bias = torch.randn(E, dtype=torch.float32, device=dev)
    w1, w1s = make_w(E, N * 2, K)
    w2, w2s = make_w(E, K, N)

    ok = fused_moe_router_supported(
        hs, w1, w2, TOPK,
        quant_type=qt.value,
        activation=ActivationType.Silu.value,
        num_expert_group=1, topk_group=1,
        gate_mode=GateMode.SEPARATED.value,
        dtype=dtypes.bf16,
    )
    print("fused_moe_router_supported:", ok)
    assert ok, "router declined the shape -- cannot smoke-test the kernel"

    # The first call per case also loads the expert GEMM modules it picks;
    # only the router itself is compiled into the image, so a slow FP8 first
    # call can be a CK MoE module JIT rather than the router.
    print("invoking ...")
    t0 = time.time()
    out = fused_moe_router(
        hs, gate, bias, w1, w2, TOPK,
        num_expert_group=1, topk_group=1, need_renorm=True,
        activation=ActivationType.Silu.value,
        quant_type=qt.value,
        w1_scale=w1s, w2_scale=w2s, dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
    )
    torch.cuda.synchronize()
    print(f"first call: {time.time()-t0:.1f}s  out={tuple(out.shape)} {out.dtype}")
    assert out.shape == (M, K), out.shape
    assert torch.isfinite(out).all(), "non-finite values in router output"
    print("output finite, shape correct")

print("\nOK")
