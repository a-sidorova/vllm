import torch, time
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe_router, fused_moe_router_supported
from aiter.ops.flydsl.moe_common import GateMode

# Solar-Open2 decode shape, routed-only: hidden 4096, inter 1280, E=80, topk=7.
M, K, N, E, TOPK = 64, 4096, 1280, 80, 7
dev = "cuda"
torch.manual_seed(0)

hs = torch.randn(M, K, dtype=dtypes.bf16, device=dev)
gate = torch.randn(M, E, dtype=dtypes.bf16, device=dev)
bias = torch.randn(E, dtype=torch.float32, device=dev)

def mxfp4(*shape):
    w = torch.randint(0, 255, (*shape[:-1], shape[-1] // 2),
                      dtype=torch.uint8, device=dev).view(dtypes.fp4x2)
    s = torch.randint(118, 126, (*shape[:-1], shape[-1] // 32),
                      dtype=torch.uint8, device=dev).view(dtypes.fp8_e8m0)
    return w, s

w1, w1s = mxfp4(E, N * 2, K)
w2, w2s = mxfp4(E, K, N)

ok = fused_moe_router_supported(
    hs, w1, w2, TOPK,
    quant_type=QuantType.per_1x32.value,
    activation=ActivationType.Silu.value,
    num_expert_group=1, topk_group=1,
    gate_mode=GateMode.SEPARATED.value,
    dtype=dtypes.bf16,
)
print("fused_moe_router_supported:", ok)
assert ok, "router declined the shape -- cannot smoke-test the kernel"

print("invoking (first call JIT-compiles module_fused_moe_router) ...")
t0 = time.time()
out = fused_moe_router(
    hs, gate, bias, w1, w2, TOPK,
    num_expert_group=1, topk_group=1, need_renorm=True,
    activation=ActivationType.Silu.value,
    quant_type=QuantType.per_1x32.value,
    w1_scale=w1s, w2_scale=w2s, dtype=dtypes.bf16,
    gate_mode=GateMode.SEPARATED.value,
)
torch.cuda.synchronize()
print(f"first call: {time.time()-t0:.1f}s  out={tuple(out.shape)} {out.dtype}")
assert out.shape == (M, K), out.shape
assert torch.isfinite(out).all(), "non-finite values in router output"
print("output finite, shape correct")
print("OK")
