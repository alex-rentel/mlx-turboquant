# Changelog

All notable changes to mlx-turboquant.

## [1.0.3] — 2026-04-29

Patch release. Tooling, contributor docs, dead-code removal, input
hardening. No production runtime changes; existing behavior preserved.

### Added

- **`pytest-cov` in CI.** Coverage report runs on the 3.12 matrix entry
  after the unit suite. Reported, not gated. Current numbers (cli.py
  omitted since it's argparse glue exercised by the CLI smoke step):
  cache.py 95%, kernels.py 97%, codebook.py 94%, packing/qjl/rotation
  100%. Total **93%**.
- **`CONTRIBUTING.md`.** Setup, local-check commands, repo layout,
  tool-config map, tripwire pointers to the fused-SDPA post-mortems,
  release process.
- **CLI input validation.** Three argparse type helpers
  (`_positive_int`, `_quantize_bits`, `_csv_positive_ints`) wired into
  `--max-tokens`, `--residual-window`, `--key-bits`, `--value-bits`,
  `--contexts`. Invalid input now fails fast with a clear message
  instead of mid-run `ValueError`s. `--key-bits` / `--value-bits` were
  previously typed `int` even though `apply_turboquant` accepts 3.5;
  fixed. 7 new tests.
- **Coverage of the `metal_dequant` pure-MLX fallback path.** The
  fallback's a sticky safety net (`_metal_dequant_disabled`); previously
  untested. New test forces the flag on, runs the same workload as
  `test_compressed_key_quality_4bit`, asserts the same ≥0.99 cos sim.

### Changed

- **`ruff check` now scans `tests/` too** (was `mlx_turboquant/` only).
  Auto-fixed 18 pre-existing violations (sorted-imports, unused
  imports, useless f-prefixes), hand-fixed 2 unused locals.
- **CI lint split into a fast Linux job.** `ruff` runs on
  `ubuntu-latest` (no MLX needed) before the macos-14 test matrix —
  fail-fast on style without burning Apple-Silicon minutes. The macos
  job keeps `ruff` as a sanity check; pyright stays in macos because it
  needs to import-resolve mlx.
- **`pyrightconfig.json` consolidated into `[tool.pyright]` in
  `pyproject.toml`.** Single source of truth for tool config.
- **Dead `qjl_inner_product` removed.** Zero callers anywhere; intended
  for future fused-attention work that already concluded as a negative
  result. Recoverable from git if needed.
- **`docs/INTERNALS.md` corrected.** Was claiming `metal_quantize_4bit`
  was wired into the compress hot path. It isn't — cache.py uses pure
  MLX. Section rewritten to match reality and explain why the kernel
  stays unwired (v1.0.2 microbench: MLX wins for realistic batches).

### Test counts

- v1.0.2 shipped at 207 passing.
- v1.0.3: **214 passing** (+7 CLI validator tests).

## [1.0.2] — 2026-04-29

Patch release. Continued post-v1.0 polish — model-compatibility fix,
test depth, dead-weight removal, contributor guard rails. No
algorithmic changes; runtime is statistically tied with v1.0.1
(n=40 trials, all deltas inside one standard deviation).

### Added

- **Sliding-window-attention auto-skip in `apply_turboquant`.**
  TurboQuantKVCache cannot produce a windowed attention mask, so its
  `make_mask` raises `NotImplementedError` on `window_size`. Models
  with SWA layers (Mistral, newer Qwen, Gemma2's local layers, etc.)
  would have hit that hole. `apply_turboquant` now probes the model's
  default cache types at apply time, identifies layers whose default
  is a `RotatingKVCache`-family cache, and routes those layers to the
  model's preferred cache while still compressing the rest. Mirrors
  the existing hybrid linear-attention detection. New test
  `test_sliding_window_attention_layers_skipped`.
- **Python 3.13 in CI matrix and pyproject classifiers.** Verified
  locally on a fresh 3.13.12 venv with mlx 0.31.2: full unit suite
  green.
- **Fused-SDPA tripwire** (`test_cache_exposes_no_v08_fused_sdpa_hooks`).
  The v0.8.0 fused-SDPA branch produced bit-identical correctness
  but lost on speed (0.61x-0.99x vs the standard SDPA path); a guard
  on `cache.py` now points future contributors at
  `docs/FUSED_SDPA_RESULTS.md` before they reattempt the integration.

### Fixed

- **`precompute_codebooks` write path.** Same root cause as the
  `get_codebook` fix shipped in v1.0.1 — the function unconditionally
  wrote `.npz` files into the package source dir, causing
  `PermissionError` on pip-installed wheels. Now defaults to the user
  cache (`~/.cache/mlx_turboquant/`) with an optional `target_dir`
  for the maintainer-prep workflow.
- **`cli.py` mlx-lm load() return narrowed.** `load()` returns either
  a 2-tuple or 3-tuple depending on version; the direct unpack
  produced a persistent pyright warning. Indexed slots explicitly.
  Pyright now sits at 0 errors / 0 warnings (was 0 / 1).

### Performance

- **Removed defensive `mx.array(...)` wrap in `_drain_chunk`.** Older
  MLX versions had a lazy-graph bug that required the LHS / RHS slice
  shift to materialize through a fresh allocation; under MLX 0.31+
  the in-place semantics are correct. Drops three lines of dead
  defensive code. Verified by full chunked-compression and
  state-roundtrip suite plus a multi-chunk drain quality probe
  (1984 tokens compressed across many cycles, mean key cos sim
  0.9940).
- **`_quant_4bit_pack_kernel` shared-memory rewrite.** Closes the
  v0.6.0 CHANGELOG-deferred TODO. The original kernel recomputed
  `inp[row*D + k] * inv_norm` inside every per-byte rotation loop,
  giving O(D^2) global loads per row. New 2D-grid layout has one
  threadgroup per row; threads cooperatively pre-normalize the input
  into `threadgroup float shared_x[D]`, then read from shared memory
  for all D output rotations. Drops global loads to O(D) per row.
- **Decision NOT to wire the kernel into the production hot path.**
  Microbenchmark on M1 Max 64GB / mlx 0.31.1 (n=30, 5 warmup) shows
  the kernel beats the pure-MLX rotate+quantize_scalar+pack pipeline
  only at small batches (N <= 512); MLX wins decisively for any
  realistic LLM batch (any model with H >= 16 or T >= 128 sits in
  MLX-wins territory). Full table in `kernels.py` docstring. The
  kernel stays unwired but tested, available for future research.

### Tests

- **Parameterized round-trip coverage for `state.setter` legacy paths**
  (6/8/10-element tuples). Pre-v0.6.0, v0.6.0 pre-fractional, and
  current state shapes are now exercised against realistic state, not
  just `None` placeholders. Added 11 parameterized cases.
- **Kernel correctness tests for `metal_quantize_4bit`.** Round-trip
  cos-sim parameterized over D in {64, 96, 128, 256}, norms-vs-numpy
  reference equality, output shape contract.

### Repo housekeeping

- `tests/REAL_MODEL_RESULTS.md` moved to `docs/REAL_MODEL_RESULTS.md`
  (no internal references; tests/ is an odd home for a results log).
- `pyrightconfig.json` consolidated into `[tool.pyright]` in
  `pyproject.toml` — single source of truth for tool config.
- README requirements line updated to mention 3.10-3.13 support and
  carry a "Last validated" footer.

### Test counts

- v1.0.1 shipped at 185 passing.
- v1.0.2: **206 passing** (+21: SWA detection +1, state.setter
  parameterized round-trip +11, kernel correctness +6, codebook cache
  paths +2, fused-SDPA tripwire +1).

## [1.0.1] — 2026-04-29

Patch release. Post-v1.0.0 review polish — bug fixes and tooling. No
algorithmic changes; runtime behavior is identical for the documented
use cases that already worked.

### Fixed

- **`apply_turboquant` type annotations.** `key_bits` and `value_bits`
  were typed `int` but the README documents `3.5` as a valid fractional
  value and the underlying cache accepted it at runtime. Type-checkers
  rejected documented usage. Both parameters now typed `float`.
- **MLX dependency floor.** `pyproject.toml` had `mlx>=0.20.0` while
  the README and the `mx.fast.metal_kernel` API usage in `kernels.py`
  require `>=0.31`. Bumped to match.
- **Codebook persistence path.** `get_codebook()` previously wrote
  newly-computed `.npz` files into the package source directory. That
  path is read-only when pip-installed into site-packages, causing
  `PermissionError` for any non-precomputed `(head_dim, bits)` combo.
  Runtime writes now go to `~/.cache/mlx_turboquant/` (overridable via
  `$XDG_CACHE_HOME` or `$MLX_TURBOQUANT_CACHE`); shipped codebooks are
  still loaded from the package dir.
- **`TurboQuantKVCache.make_mask` silent contract hole.** The method
  accepted a `window_size` argument but ignored it, silently returning
  full-causal masking for sliding-window callers. Now raises
  `NotImplementedError` so the missing capability is loud.
- **`apply_turboquant` upgrade-bits warning.** Warning text now reports
  the *effective* config after the auto-upgrade for low-`nkv` models,
  not just the delta.

### Tooling

- **ruff + pyright in CI.** The package shipped `py.typed` from v1.0.0
  but had no type checker enforcing the typed contract — exactly why
  the `int`/`float` annotation bug shipped. CI now runs both on every
  push. Pyright is configured at `basic` mode with Optional-narrowing
  rules suppressed (cache.py uses runtime length-guards that pyright
  cannot see); `reportArgumentType` stays at `error` to catch the next
  int-vs-float-style bug.
- **Type-narrowing asserts in `cache.py`.** Five `assert ... is not
  None` invariant checks added at sites where pyright could not narrow
  Optional fields tied together by runtime sequencing.

### Repo housekeeping

- `BENCHMARKS_v07.md` renamed to `docs/BENCHMARKS_FUSED_QK.md` for
  clarity (it covers the v0.7.0 fused QK micro-benchmark, not
  end-to-end model results — distinct from `BENCHMARKS.md`). All
  references updated.
- 185 tests passing (was 183 in v1.0.0 — added two contract tests for
  the codebook cache redirect and the `make_mask` window_size guard).

## [1.0.0] — 2026-04-08

First stable release. No algorithmic changes — this release draws a
semver line around the library and commits to keeping the public API
stable from here on.

### Stabilized public API

Exported from `mlx_turboquant` and covered by semver:

- `apply_turboquant(model, **kwargs)` — primary entry point
- `enable_turboquant(model, bits, **kwargs)` — symmetric convenience
- `TurboQuantKVCache` — cache class
- `TurboQuantMSE`, `TurboQuantProd` — low-level quantizers

Everything else (`mlx_turboquant.kernels`, `.rotation`, `.codebook`,
`.packing`, `.qjl`) is explicitly internal and may change between minor
versions without a deprecation warning.

### Validated

- **Long-context needle-in-a-haystack at 8K / 16K / 32K.** Qwen3-8B
  and Llama-3.1-8B pass **9/9** on both FP16 baseline and K4/V2+sink128,
  with zero retrieval gap up through 32K tokens. Mistral-7B fails on
  the FP16 baseline itself at these lengths (2/9) — known model
  limitation. Raw data in `results/needle_long_context.json`.

### Added

- `__all__` in `mlx_turboquant/__init__.py` defines the public surface
- Semver + deprecation policy documented in the package docstring and README
- User-facing README rewrite: install → quick start → config selector →
  validated models → troubleshooting → limitations → versioning
- `docs/INTERNALS.md` — engineering history, kernel benchmarks, fused
  attention post-mortems, community implementations
- `.github/workflows/test.yml` — CI on macOS-14 ARM runner across
  Python 3.10 / 3.11 / 3.12
- `examples/long_context_chat.py` — 50-line demo script showing memory
  usage under K4/V2 + sink128 on Qwen3-8B

### Changed

- `fused_qk_scores_{2,3,4}bit` and `pre_rotate_query` are formally
  documented as **research-only primitives**, NOT part of the supported
  decode path. Module docstring in `kernels.py` explains the two
  integration attempts (v0.8.0 decomposed SDPA, v0.9.0 full single-dispatch
  kernel) and why both were preserved as documented negative results
  rather than shipped. The `precomputed Q·centroid table` idea is
  demoted from "most promising near-term path" to "speculative future
  work — pending 32K+ context investigation."
- Version history and engineering post-mortems moved from README to
  `docs/INTERNALS.md`.
- `__version__` synced with `pyproject.toml` (was stale at `0.5.0`).

### Fixed

(Already shipped in v0.8.1 but repeated here since v0.8.1 was never
tagged as a release.)

- `TurboQuantKVCache.nbytes` undercounted fractional-bit configs.
- Aliased in-place buffer shift in `_drain_chunk` is now safe under MLX's
  lazy graph.
- Metal dequantize fallback is sticky per cache instance and emits a
  one-time `RuntimeWarning`.
- `detect_outlier_layers` returns `[]` instead of a NaN-poisoned
  threshold when no key norms are positive.

## [0.8.1] — 2026-04-08

> Note: v0.8.0 and v0.9.0 refer to in-progress fused-attention attempts
> that live on `feat/fused-sdpa-qwen3` and `feat/full-fused-attention`
> branches as documented negative results. v0.8.1 is the first release
> on `main` after the v0.7.0 fused-QK kernel landing — a bug-fix pass.


### Fixed

- **`TurboQuantKVCache.nbytes` undercounted fractional-bit configs.** The
  low-half packed arrays (`_compressed_keys_lo`, `_compressed_values_lo`)
  used by 3.5-bit quantization were excluded from the memory total. Memory
  benchmarks for fractional configs were silently ~12% low.
  (`mlx_turboquant/cache.py`)
- **Aliased in-place buffer shift in `_drain_chunk`.** The shift
  `self.keys[:, :, :remaining, :] = self.keys[:, :, n_compress:, :]`
  reads from and writes into overlapping ranges of the same buffer.
  MLX's lazy graph does not guarantee a copy on aliased writes, so the
  result was undefined whenever `remaining > 0` (the common decode case).
  Fixed by materializing the RHS into a fresh allocation via
  `mx.async_eval` before the assignment.
  (`mlx_turboquant/cache.py`)
- **Metal dequantize fallback was non-sticky and silent.** A failure in
  the fused Metal kernel was caught with a bare `except: pass` and
  retried on every subsequent decode step, paying the failure overhead
  per token with no diagnostic. Failures are now sticky per cache instance
  and emit a `RuntimeWarning` once with the underlying exception.
  (`mlx_turboquant/cache.py`)
- **`detect_outlier_layers` returned NaN-poisoned thresholds when no key
  norms were positive.** `np.median([])` returns NaN, making
  `n > threshold * nan` always False, so the function silently classified
  every layer as non-outlier. Now returns `[]` explicitly in that
  degenerate case. (`mlx_turboquant/patch.py`)

### Validated

- Tier-1 benchmark sweep re-run on M1 Max 64GB after patches: 7 models ×
  5 configs (35 cells), zero errors, 39 min wall time. Quality, decode
  speed, TTFT, and KV memory all within run-to-run noise of the v0.7.0
  numbers — confirming no regression. Raw data in `results/post_patch/`.

## [0.7.0] — 2026-04-07

### Added

- **Fused QK scores Metal kernel** — three new Metal kernels that
  compute `Q_rot @ K^T` directly from packed codebook indices, without
  materializing dequantized K. Eliminates the per-token inverse
  rotation that dominates v0.6.0 decode overhead.
  - `mlx_turboquant.kernels.fused_qk_scores_4bit(q_rot, packed_k, norms_k, centroids, D)`
  - `mlx_turboquant.kernels.fused_qk_scores_3bit(...)`
  - `mlx_turboquant.kernels.fused_qk_scores_2bit(...)`
  - Micro-benchmark on M1 Max 64GB vs dequant+matmul baseline:
    - decode T_kv=4096 D=128: **2.12×** speedup
    - decode T_kv=1024 D=256 (Gemma3-like): **2.03×** speedup
    - decode T_kv=1024 D=128: 1.42× speedup
    - prefill is roughly tied (dispatch-bound regime)
  - Full benchmark methodology in `docs/BENCHMARKS_FUSED_QK.md`.
  - Correctness guarantee: identical to dequant+matmul to atol=1e-3 on
    random inputs across all tested shapes. 12 tests in
    `TestFusedQKScoresCorrectness{4,3,2}Bit`.

- **`pre_rotate_query(query, rotation)` utility** in `mlx_turboquant.rotation`.
  The key insight that makes the fused kernel possible: because we store
  `K = norms * (centroids[idx] @ R)`, the dot product `Q . K` simplifies
  to `norms * (Q @ R.T) . centroids[idx]`. Pre-rotating Q once per decode
  step eliminates the `D×D` inverse rotation per compressed token inside
  the attention inner loop. 5 tests in `TestPreRotateQueryMath`.

- **`docs/FUSED_ATTENTION_DESIGN.md`** — design doc covering:
  - Math derivation for the `Q @ R.T . centroids` identity
  - Kernel thread/threadgroup layout
  - Memory layout of inputs/outputs
  - Honest analysis of the integration blocker (mlx-lm's SDPA call
    accepts only dense tensors; intercepting it requires per-family
    attention patches)
  - Three considered integration approaches (A: per-family patch,
    B: custom cache method, C: utility-only — shipped in v0.7.0)
  - Numerical considerations (float32 inside kernel, rotation
    orientation, packing layout)
  - What's deferred to v0.8.0 (fusing V, SDPA integration, simd_sum
    D-reduction)

- **Phase 4 optimization: threadgroup shared centroids** in the 4-bit
  fused kernel. The 16 codebook centroids are loaded into
  `threadgroup float shared_centroids[16]` once per threadgroup, then
  every thread reads from shared memory instead of global memory during
  the D-element inner loop. Added +0.17 speedup at T_kv=4096.

- **`benchmarks/micro_fused_qk.py`** — go/no-go micro-benchmark that
  times the fused path against dequant+matmul on 7 realistic shapes.

- **`docs/BENCHMARKS_FUSED_QK.md`** — full speedup numbers, kernel architecture
  notes, and the honest "what isn't shipped in v0.7.0" section.

### Not in v0.7.0 (deferred to v0.8.0)

- **Full SDPA integration.** The fused kernels are shipped as
  first-class utilities but are NOT automatically used by
  `apply_turboquant`. `TurboQuantKVCache.update_and_fetch` still returns
  dense FP16 tensors that feed into standard
  `mx.fast.scaled_dot_product_attention`. Users who want the speedup
  today must call the kernels manually in a custom attention loop —
  see the integration pathway in `docs/BENCHMARKS_FUSED_QK.md`.
- **Fusing V.** The current work covers only the Q @ K^T side of
  attention; the softmax-weighted V accumulation still uses dequantized
  V. Fusing V requires either a two-pass kernel or online softmax.
- **`simd_sum` D-reduction.** The current kernel has one thread per
  output score with a serial D-element inner loop. A SIMD-width
  reduction would split D across 32 lanes per SIMD group.

### Changed

- No behavior changes in existing code paths. `TurboQuantKVCache`,
  `apply_turboquant`, and all v0.6.0 features are byte-identical to
  v0.6.0 when the fused kernel is not invoked. v0.6.0 → v0.7.0 is
  strictly additive.

### Tests

- 17 new tests in `tests/test_fused_attention.py`:
  - 5 `TestPreRotateQueryMath` (Phase 1)
  - 8 `TestFusedQKScoresCorrectness4Bit` (Phase 2)
  - 2 `TestFusedQKScoresCorrectness3Bit`
  - 2 `TestFusedQKScoresCorrectness2Bit`
- Full test count: **175** (158 previous + 17 new). All passing.

## [0.6.0] — 2026-04-07

### Added

- **Attention sink** (`fp16_sink_size`): Permanent FP16 region for the first
  N tokens of every sequence, never compressed regardless of compression
  cycles. Independent of the sliding `residual_window`. Use to preserve
  system prompt / tool schema tokens for long-context inference. Default
  `0` (disabled). Pattern follows helgklaizar's MLX implementation; see
  `docs/COMPETITIVE_AUDIT.md` for credit and details.
  - Quality wins on all 3 benchmarked models (Qwen3-1.7B, Qwen3-8B,
    Gemma3-1B): cosine similarity improvement of +0.0007 to +0.0032.
  - Zero measurable speed cost.
  - Validated against needle-in-haystack at 1K/2K/4K/8K — 12/12.

- **QJL residual correction** (`qjl_correction`, `qjl_n_proj`):
  Experimental opt-in 1-bit Quantized Johnson-Lindenstrauss residual
  sketch applied additively at compression time. Unlike sharpner's
  cache_v2.py which stores QJL signs but never reads them back, our
  implementation immediately consumes the sketch and bakes the
  correction into the cached dequantized vectors — zero memory
  overhead, ~5% extra compute per compression chunk.
  - Mixed benchmark results: +0.0017 cos_sim on Qwen3-8B, **−0.0052
    on Qwen3-1.7B**, near-noise on Gemma3-1B. Default OFF. Document
    explicitly mentions experimental status.
  - Synthetic Gaussian tests confirm the correction reduces MSE in
    isolation. Real-model variance stems from KV vector structure
    interacting with the random JL projection.

- **Chunked compression** (`chunk_size`): Optional fixed-size chunked
  drain path. When `chunk_size > 0`, the FP16 buffer is drained in whole
  blocks of `chunk_size` tokens whenever it exceeds `residual_window +
  chunk_size`. Default `0` selects the v0.5.0 batch behavior (single
  variable-size drain at `2 * residual_window` threshold) which
  benchmarks identically. Provided as opt-in for future Metal kernel
  work that benefits from stable input shapes.

- New benchmark harness `benchmarks/bench_v06.py` runs 5 configs across
  3 target models and writes structured JSON to `benchmarks/results_v06/`.

- New needle test harness `benchmarks/needle_haystack_v06.py` validates
  Qwen3-8B at 4 context lengths × 3 needle positions across 3 configs.

- New `docs/COMPETITIVE_AUDIT.md` documenting findings from 4 community
  TurboQuant MLX implementations (sharpner, arozanov, rachittshah,
  helgklaizar). Honest call-outs on what's real vs marketing in each
  repo, with file:line citations.

### Fixed

- **Latent state-reload bugs in `TurboQuantKVCache`** (present in v0.5.0
  but unexercised by tests):
  - `state` property now includes `_compressed_keys_lo` and
    `_compressed_values_lo` (positions 9, 10). Without these, fractional
    bit configs (e.g. `key_bits=3.5`) could not survive a state restore.
  - `meta_state` now includes `fp16_len` and `fp16_capacity`. Previously
    a state reload lost the residual buffer count, causing subsequent
    decode calls to return wrong total token counts.
  - Lazy re-dequantization after state restore is now dispatched through
    a new `_rebuild_decompressed_cache()` helper that handles fractional
    and non-fractional cases correctly. Old code called the
    non-fractional dequant helper with `None` centroids on fractional
    state reloads, which would crash.
  - State setter is backward-compatible: 6-tuple (pre-v0.6.0), 8-tuple
    (interim sink-only), and 10-tuple (current) all load correctly.

- Removed dead `_dequant_calls` counter that was incremented but never
  read.

### Changed

- `TurboQuantKVCache.__init__` and `apply_turboquant()` gain four new
  parameters: `fp16_sink_size`, `chunk_size`, `qjl_correction`,
  `qjl_n_proj`. All default to values that exactly preserve v0.5.0
  behavior — **no behavioral change unless you explicitly opt in.**

- `BENCHMARKS.md` rewritten with v0.6.0 numbers, methodology section,
  per-feature pass/fail analysis, and v0.5.0 comparison.

- `README.md` updated with v0.6.0 quality numbers, sink configuration,
  needle table, and v0.6.0 roadmap entry.

### Tests

- Added `TestAttentionSink` (8 tests) covering sink default-off,
  single-prefill fill, multi-call fill, partial overlap with residual,
  survival across compression cycles, KV head ordering, offset tracking,
  and meta_state round-trip.
- Added `TestChunkedCompression` (5 tests) for the opt-in chunked path.
- Added `TestQJLCorrection` (4 tests) including a synthetic MSE-reduction
  regression test.
- Added `TestStateReload` (3 tests) covering integer roundtrip,
  fractional roundtrip (regression for the latent bugs above), and
  legacy 6-tuple state backward compatibility.
- Total test count: **157** (up from 137 in v0.5.0). All passing.

### Deferred to v0.7.0

- Rewrite `_quant_4bit_pack_kernel` to use shared-memory pre-normalized
  vectors instead of recomputing the normalized x inside the per-byte
  inner loop. Currently O(D) redundant work per output byte; flagged in
  the competitive audit as the largest single Metal kernel win.
- Adopt `simd_sum` and SIMD-group reductions in dequant kernels per
  sharpner's pattern (`/tmp/sharpner-tq/turboquant/kernels.py:597`).
- Build a single fused attention-from-packed kernel modeled on
  sharpner's `_FUSED_ATTN_NOROT_SOURCE` to eliminate the per-step
  concat overhead that drives the residual ~11-22% decode penalty.

## [0.5.0] — 2026-04-02

- Batch compression at 2x residual window threshold
- Pre-allocated FP16 window with slice-write
- 33% → 11% decode overhead on Qwen3-8B at 2K context
- 137 tests passing

## [0.4.0] — 2026-03

- 3.5-bit fractional quantization
- Needle-in-a-haystack at 1K-8K (12/12)
- PyPI packaging

## [0.3.0] — 2026-03

- Fused Metal dequantize kernels for 2/3/4-bit
- Fused Metal quantize kernel for 4-bit
- 57% → 33% decode overhead

## [0.2.0] — 2026-02

- Real model testing across 6 model families
- Vectorized quantization
- 57% decode overhead (Python-only)
