# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prerequisites for speculative decoding on Solar-Open2.

Solar-Open2 interleaves KDA (Kimi Delta Attention) linear-attention layers with
GQA layers. This module pins what its speculative path depends on:

1. ``speculative_config`` must fail closed rather than corrupt recurrent state,
   with an explicit opt-in.
2. ``kda_state_shape`` must reserve conv-state room for draft tokens.
3. ``fused_recurrent_kda`` must forward ``num_accepted_tokens``.
4. The target must expose aux hidden states (``SupportsEagle3``), which every
   post-hoc drafter and ``extract_hidden_states`` both require.

These are import-time / signature-level / pure-function checks, so they run on
CPU without Triton.
"""

import inspect
from types import SimpleNamespace

import pytest

# Every test here is an import-time / signature-level / pure-function check that
# never initializes torch distributed, so the global cleanup fixture is pure
# overhead (see its docstring in tests/conftest.py).
pytestmark = pytest.mark.skip_global_cleanup

SOLAR_ARCHITECTURE = "SolarOpen2ForCausalLM"


# ---------------------------------------------------------------------------
# Fail-closed guard, with an explicit opt-in
# ---------------------------------------------------------------------------


def _stub_vllm_config(architecture: str, method: str | None):
    """Minimal stand-in exercising only the MODELS_CONFIG_MAP hook."""
    return SimpleNamespace(
        model_config=SimpleNamespace(architecture=architecture),
        speculative_config=(None if method is None else SimpleNamespace(method=method)),
    )


@pytest.mark.parametrize(
    "method", ["ngram", "eagle3", "dflash", "extract_hidden_states"]
)
def test_solar_open2_rejects_speculative_config(method):
    """Stay fail-closed unless explicitly enabled.

    The failure mode of a broken KDA speculative path is a silently wrong
    recurrent state, so the guard stays on by default.

    ``extract_hidden_states`` is *not* exempt. It feeds each sampled token back
    as a draft, so every decode step carries drafts and the GDN metadata builder
    takes the speculative branch exactly as a real drafter would.
    """
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    assert SOLAR_ARCHITECTURE in MODELS_CONFIG_MAP

    cls = MODELS_CONFIG_MAP[SOLAR_ARCHITECTURE]
    with pytest.raises(ValueError, match="not been validated"):
        cls.verify_and_update_config(_stub_vllm_config(SOLAR_ARCHITECTURE, method))


def test_solar_open2_allows_no_speculative_config():
    """The guard must be inert when speculative decoding is off."""
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    MODELS_CONFIG_MAP[SOLAR_ARCHITECTURE].verify_and_update_config(
        _stub_vllm_config(SOLAR_ARCHITECTURE, None)
    )


def test_solar_open2_opt_in_env_var_lifts_the_guard(monkeypatch):
    from vllm.model_executor.models import config as models_config

    monkeypatch.setattr(models_config.envs, "VLLM_KDA_SPEC_DECODE", True)

    models_config.MODELS_CONFIG_MAP[SOLAR_ARCHITECTURE].verify_and_update_config(
        _stub_vllm_config(SOLAR_ARCHITECTURE, "dflash")
    )


def test_kimi_linear_is_not_gated():
    """Kimi-Linear ships its own validated KDA speculative path."""
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        KdaHybridSpecDecodeUnsupportedConfig,
    )

    assert (
        MODELS_CONFIG_MAP.get("KimiLinearForCausalLM")
        is not KdaHybridSpecDecodeUnsupportedConfig
    )


def test_kda_spec_decode_opt_in_is_a_registered_env_var():
    """Unregistered VLLM_* reads emit a spurious unknown-env-var warning."""
    from vllm import envs

    assert "VLLM_KDA_SPEC_DECODE" in envs.environment_variables
    assert envs.VLLM_KDA_SPEC_DECODE is False, "must be off by default"


@pytest.mark.parametrize("aiter_enabled", [True, False])
@pytest.mark.parametrize("speculative", [True, False])
def test_solar_open2_fused_kda_path_is_gated_by_platform_only(
    monkeypatch, aiter_enabled, speculative
):
    """The fused KDA path implements the speculative branch itself, so it stays
    on under speculative decoding. The attention layer and
    SolarOpen2Model.load_weights both read this gate, and they have to agree
    for the merged in_proj/conv1d params to be loaded."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.models import solar_open2

    monkeypatch.setattr(
        rocm_aiter_ops, "is_enabled", classmethod(lambda cls: aiter_enabled)
    )
    vllm_config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(method="dflash", num_speculative_tokens=7)
            if speculative
            else None
        )
    )
    assert solar_open2._use_amd_fused_kda(vllm_config) is aiter_enabled


def test_kda_guard_does_not_replace_hybrid_config_verification():
    """MODELS_CONFIG_MAP and the hybrid hook both run; ours must be additive."""
    from vllm.model_executor.models.config import (
        KdaHybridSpecDecodeUnsupportedConfig,
        VerifyAndUpdateConfig,
    )

    assert issubclass(KdaHybridSpecDecodeUnsupportedConfig, VerifyAndUpdateConfig)
    assert (
        "verify_and_update_model_config"
        not in KdaHybridSpecDecodeUnsupportedConfig.__dict__
    )


# ---------------------------------------------------------------------------
# State shape reserves room for draft tokens
# ---------------------------------------------------------------------------


def _conv_state_len_index() -> int:
    """Index of the state-length axis in a conv-state shape.

    ``(dim, state_len)`` under the DS layout and ``(state_len, dim)`` under SD.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    return -1 if is_conv_state_dim_first() else 0


def test_kda_state_shape_reserves_room_for_speculative_tokens():
    """``causal_conv1d_update`` writes ``width - 1 + (seqlen - 1)`` columns on a
    speculative step, with no bounds check (``validate_data=False``), so the
    conv state must be widened by ``num_spec``.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    state_idx = _conv_state_len_index()
    dim_idx = 0 if state_idx == -1 else 1

    kwargs = dict(tp_world_size=1, num_heads=64, head_dim=128, conv_kernel_size=4)
    conv_base, recurrent_base = MambaStateShapeCalculator.kda_state_shape(
        **kwargs, num_spec=0
    )
    conv_spec, recurrent_spec = MambaStateShapeCalculator.kda_state_shape(
        **kwargs, num_spec=3
    )

    assert conv_base[state_idx] == 4 - 1, "num_spec=0 must be unchanged"
    assert conv_spec[state_idx] == conv_base[state_idx] + 3, (
        f"conv state not widened: {conv_base} -> {conv_spec}"
    )
    assert conv_spec[dim_idx] == conv_base[dim_idx], "channel axis must not move"
    assert recurrent_spec == recurrent_base, "recurrent state is fixed-size"


def test_kda_state_shape_matches_gated_delta_net_convention():
    """Both calculators must widen the conv state the same way."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    state_idx = _conv_state_len_index()

    kda_conv, _ = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=1, num_heads=8, head_dim=64, conv_kernel_size=4, num_spec=5
    )
    gdn_conv, _ = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_world_size=1,
        num_k_heads=8,
        num_v_heads=8,
        head_k_dim=64,
        head_v_dim=64,
        conv_kernel_size=4,
        num_spec=5,
    )
    assert kda_conv[state_idx] == gdn_conv[state_idx] == 4 - 1 + 5


def test_kda_layer_get_state_shape_forwards_num_spec():
    """The layer and the model must derive the same shape.

    The model-side value only sizes the reported mamba page, while the
    layer-side value drives the actual allocation -- so a mismatch
    under-allocates without tripping any assert.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.model_executor.models.solar_open2 import SolarOpen2KimiDeltaAttention

    state_idx = _conv_state_len_index()
    stub = SimpleNamespace(
        tp_size=1, num_heads=64, head_dim=128, conv_size=4, num_spec=3
    )

    conv_shape, _ = SolarOpen2KimiDeltaAttention.get_state_shape(stub)
    expected, _ = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=1, num_heads=64, head_dim=128, conv_kernel_size=4, num_spec=3
    )
    assert conv_shape[state_idx] == expected[state_idx] == 4 - 1 + 3


# ---------------------------------------------------------------------------
# Wrapper plumbing
# ---------------------------------------------------------------------------


def test_fused_recurrent_kda_forwards_num_accepted_tokens(monkeypatch):
    """Dropping ``num_accepted_tokens`` silently flips the kernel's
    ``IS_SPEC_DECODING`` heuristic, which then reads the initial state from
    draft slot 0 instead of slot ``num_accepted - 1``.
    """
    import torch

    from vllm.third_party.flash_linear_attention.ops import kda

    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return "out", "state"

    monkeypatch.setattr(kda, "fused_recurrent_kda_fwd", _spy)

    sentinel = torch.tensor([2, 1], dtype=torch.int32)
    kda.fused_recurrent_kda(
        q=torch.zeros(1, 2, 1, 1),
        k=torch.zeros(1, 2, 1, 1),
        v=torch.zeros(1, 2, 1, 1),
        g=torch.zeros(1, 2, 1, 1),
        beta=torch.zeros(1, 2, 1),
        num_accepted_tokens=sentinel,
    )

    assert seen["num_accepted_tokens"] is sentinel, (
        "wrapper dropped num_accepted_tokens on the way to the kernel"
    )


def test_fused_recurrent_kda_takes_num_accepted_tokens_explicitly():
    """An explicit parameter, so it can never be absorbed by ``**kwargs``."""
    from vllm.third_party.flash_linear_attention.ops import kda

    params = inspect.signature(kda.fused_recurrent_kda).parameters
    assert "num_accepted_tokens" in params
    assert params["num_accepted_tokens"].kind is not inspect.Parameter.VAR_KEYWORD
    assert params["num_accepted_tokens"].default is None


def test_fused_recurrent_kda_fwd_already_accepts_spec_args():
    from vllm.third_party.flash_linear_attention.ops import kda

    params = inspect.signature(kda.fused_recurrent_kda_fwd).parameters
    assert "ssm_state_indices" in params
    assert "num_accepted_tokens" in params


def _kda_fwd_kwargs(num_tokens: int) -> dict:
    """Smallest inputs that get past shape unpacking in fused_recurrent_kda_fwd."""
    import torch

    shape = (1, num_tokens, 1, 1)
    return dict(
        q=torch.zeros(shape),
        k=torch.zeros(shape),
        v=torch.zeros(shape),
        g=torch.zeros(shape),
        beta=torch.zeros(1, num_tokens, 1),
        scale=1.0,
        initial_state=None,
    )


def test_fused_recurrent_kda_fwd_requires_contiguous_state_index_rows():
    """The kernel locates a slot as ``ssm_state_indices + i_n * stride_seq +
    i_t`` -- it never multiplies ``i_t`` by the token stride, so a strided view
    would read a neighbouring request's block: in bounds, no crash, wrong
    numbers.
    """
    import torch

    from vllm.third_party.flash_linear_attention.ops import kda

    strided = torch.zeros(4, 8, dtype=torch.int32)[:, ::2]
    assert strided.stride(-1) != 1

    with pytest.raises(AssertionError, match="contiguous"):
        kda.fused_recurrent_kda_fwd(
            **_kda_fwd_kwargs(1),
            ssm_state_indices=strided,
            num_accepted_tokens=torch.ones(4, dtype=torch.int32),
        )


def test_fused_recurrent_kda_fwd_rejects_1d_indices_for_spec_decode():
    """A 1-D index tensor cannot describe a per-draft-token ladder.

    Deliberately NOT gated on ``T``: in varlen mode ``T`` is the packed token
    count for the whole batch, so a healthy non-speculative decode batch of many
    single-token sequences must get past the guard.
    """
    import torch

    from vllm.third_party.flash_linear_attention.ops import kda

    indices_1d = torch.ones(4, dtype=torch.int32)

    with pytest.raises(Exception) as excinfo:
        kda.fused_recurrent_kda_fwd(**_kda_fwd_kwargs(32), ssm_state_indices=indices_1d)
    assert "1-D ssm_state_indices" not in str(excinfo.value)

    with pytest.raises(AssertionError, match="1-D ssm_state_indices"):
        kda.fused_recurrent_kda_fwd(
            **_kda_fwd_kwargs(4),
            ssm_state_indices=indices_1d,
            num_accepted_tokens=torch.ones(4, dtype=torch.int32),
        )


# ---------------------------------------------------------------------------
# SupportsEagle3 on the target
# ---------------------------------------------------------------------------


def test_solar_open2_declares_eagle3_support():
    """Every post-hoc drafter, and extract_hidden_states, needs aux taps."""
    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.model_executor.models.solar_open2 import SolarOpen2ForCausalLM

    assert supports_eagle3(SolarOpen2ForCausalLM)


def test_solar_open2_model_inherits_eagle_mixin():
    from vllm.model_executor.models.interfaces import EagleModelMixin
    from vllm.model_executor.models.solar_open2 import SolarOpen2Model

    assert issubclass(SolarOpen2Model, EagleModelMixin)


def _run_decoder_loop(monkeypatch, *, taps, num_layers=6, start_layer=0):
    """Drive the real SolarOpen2Model.forward over stub layers.

    Layer i is the identity plus ``i + 1``, so a tapped value identifies exactly
    which layer produced it. Residual stays None so the mixin records the hidden
    state verbatim.
    """
    import torch

    from vllm.model_executor.models import solar_open2

    class _Layer:
        def __init__(self, idx):
            self.idx = idx

        def __call__(self, positions, hidden_states, residual, **kwargs):
            return hidden_states + (self.idx + 1), None

    class _PPGroup:
        is_first_rank = start_layer == 0
        is_last_rank = True

    monkeypatch.setattr(solar_open2, "get_pp_group", lambda: _PPGroup())

    model = solar_open2.SolarOpen2Model.__new__(solar_open2.SolarOpen2Model)
    object.__setattr__(
        model,
        "__dict__",
        {
            "layers": [_Layer(i) for i in range(num_layers)],
            "start_layer": start_layer,
            "end_layer": num_layers,
            "norm": lambda h, r: (h, r),
            "aux_hidden_state_layers": taps,
        },
    )

    zero = torch.zeros(1)
    if start_layer == 0:
        return solar_open2.SolarOpen2Model.forward(
            model, None, None, inputs_embeds=zero
        )

    from vllm.sequence import IntermediateTensors

    return solar_open2.SolarOpen2Model.forward(
        model,
        None,
        None,
        intermediate_tensors=IntermediateTensors(
            {"hidden_states": zero, "residual": None}
        ),
    )


def test_solar_open2_aux_taps_capture_the_requested_layer_outputs(monkeypatch):
    """Tap k must be the output of layer k, and tap 0 the embedding output."""
    hidden, aux = _run_decoder_loop(monkeypatch, taps=(0, 2, 4), num_layers=6)

    assert isinstance(aux, list)
    assert len(aux) == 3
    # Layer i adds i + 1, so after layers 0..k-1 the value is sum(1..k).
    assert [float(t) for t in aux] == [0.0, 1.0 + 2.0, 1.0 + 2.0 + 3.0 + 4.0]
    assert float(hidden) == sum(range(1, 7))


def test_solar_open2_returns_single_value_without_aux_layers(monkeypatch):
    import torch

    out = _run_decoder_loop(monkeypatch, taps=(), num_layers=6)

    assert isinstance(out, torch.Tensor), (
        "must return a bare tensor when no taps are armed"
    )


def test_solar_open2_aux_tap_indices_are_global_under_pipeline_parallel(monkeypatch):
    """Tap ids are global layer ids, so a PP shard must not renumber from 0."""
    _, aux = _run_decoder_loop(monkeypatch, taps=(4,), num_layers=6, start_layer=3)

    # Global tap 4 is the output of layers[3], the first layer this rank owns.
    assert len(aux) == 1
    assert float(aux[0]) == 4.0


def test_solar_open2_set_aux_hidden_state_layers_reaches_the_model():
    """``set_aux_hidden_state_layers`` is called on the *ForCausalLM*, but the
    decoder loop reads ``aux_hidden_state_layers`` on the *Model*.
    """
    from vllm.model_executor.models.solar_open2 import (
        SolarOpen2ForCausalLM,
        SolarOpen2Model,
    )

    model = SolarOpen2Model.__new__(SolarOpen2Model)
    object.__setattr__(model, "__dict__", {"layers": [None] * 48})
    causal_lm = SolarOpen2ForCausalLM.__new__(SolarOpen2ForCausalLM)
    object.__setattr__(causal_lm, "__dict__", {"model": model})

    SolarOpen2ForCausalLM.set_aux_hidden_state_layers(causal_lm, (2, 24, 45))
    assert model.aux_hidden_state_layers == (2, 24, 45)

    assert SolarOpen2ForCausalLM.get_eagle3_default_aux_hidden_state_layers(
        causal_lm
    ) == (2, 24, 45)


def test_solar_open2_does_not_override_default_aux_layers():
    """Tap count is a drafter-checkpoint decision, not a model-code one."""
    from vllm.model_executor.models.solar_open2 import SolarOpen2ForCausalLM

    assert (
        "get_eagle3_default_aux_hidden_state_layers"
        not in SolarOpen2ForCausalLM.__dict__
    )
