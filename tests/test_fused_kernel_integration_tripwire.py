"""Tripwires: detect when fused-attention research lands in the supported
decode path.

As of v1.0.x the fused QK kernels (`fused_qk_scores_{2,3,4}bit`,
`pre_rotate_query`) ship as research-grade primitives only.
`apply_turboquant` runs the dequant+matmul path through
`mx.fast.scaled_dot_product_attention` and `TurboQuantKVCache` exposes
no fused-mode accessors.

There have been three integration attempts so far, each preserved on a
branch and documented as a negative result:

* **v0.7.0 fused QK kernel** — measured 2.12× vs dequant+matmul in
  isolation but never tied into a real decode path. See `kernels.py`
  module docstring for the M1 Max microbench numbers.
* **v0.8.0 decomposed SDPA on `feat/fused-sdpa-qwen3`** — bit-identical
  correctness against FP16 (cos sim 1.000000) but 0.61×–0.99× decode
  speed at 1K–4K context. Lost to `mx.fast.scaled_dot_product_attention`
  even though it beat dequant+matmul in isolation. Full post-mortem:
  `docs/FUSED_SDPA_RESULTS.md`.
* **v0.9.0 full fused attention on `feat/full-fused-attention`** —
  similar speed-loss profile. See `docs/FULL_FUSED_ATTENTION_RESULTS.md`.

Read the post-mortems before reattempting. The structural reason both
v0.8 and v0.9 lost: at realistic decode shapes the path is
dispatch/latency bound, not memory-bandwidth bound, so the packed-KV
memory advantage never materializes.

When someone lands a successful integration on `main`, these tests will
fail. That failure is the cue to:

  1. Delete this tripwire file.
  2. Add a real end-to-end integration test that exercises the fused
     kernel inside a forward pass on a small model and checks logits
     against the standard SDPA path.
  3. Update README "Next Steps" with the new measured numbers.
"""

import inspect

import mlx_turboquant.cache as cache_module
import mlx_turboquant.patch as patch_module


def test_patch_module_does_not_import_fused_kernels():
    """Source-level check: patch.py must not reference fused_qk_scores
    or pre_rotate_query. If this fails, see the module docstring."""
    source = inspect.getsource(patch_module)
    assert "fused_qk_scores" not in source, (
        "patch.py now references fused_qk_scores — the v0.7.0 kernel has "
        "been wired into the decode path. Delete this tripwire and add a "
        "real end-to-end integration test (see module docstring)."
    )
    assert "pre_rotate_query" not in source, (
        "patch.py now references pre_rotate_query — the v0.7.0 kernel has "
        "been wired into the decode path. Delete this tripwire and add a "
        "real end-to-end integration test (see module docstring)."
    )


def test_cache_exposes_no_v08_fused_sdpa_hooks():
    """Source-level check: cache.py must not expose the v0.8.0
    fused-SDPA accessors. If this fails, the v0.8.0 integration is
    landing — read docs/FUSED_SDPA_RESULTS.md before celebrating.

    The v0.8.0 attempt added `_use_fused_attention` and
    `get_fused_state()` to TurboQuantKVCache so a patched
    `scaled_dot_product_attention` could pull packed state without
    materializing the dequantized middle. End-to-end it lost on speed
    despite being bit-identical on correctness."""
    source = inspect.getsource(cache_module)
    assert "_use_fused_attention" not in source, (
        "cache.py now exposes _use_fused_attention — looks like the "
        "v0.8.0 fused-SDPA integration is being landed. Read "
        "docs/FUSED_SDPA_RESULTS.md first; the prior attempt was 0.61x-"
        "0.99x slower than baseline. If the new attempt actually beats "
        "mx.fast.scaled_dot_product_attention, delete this tripwire and "
        "add a speed regression test."
    )
    assert "get_fused_state" not in source, (
        "cache.py now defines get_fused_state — see "
        "docs/FUSED_SDPA_RESULTS.md for the v0.8.0 negative result."
    )
