"""Fused Metal kernels for TurboQuant quantization and dequantization.

These kernels eliminate intermediate array allocations by fusing:
- Dequantize: unpack indices + centroid lookup + inverse rotation + norm scaling
- Quantize: norm computation + normalize + rotation + boundary search + packing

Each kernel runs as a single Metal dispatch, avoiding Python overhead and
intermediate tensor materialization.

Research-only primitives
------------------------

The ``metal_dequantize`` and ``metal_quantize_4bit`` kernels are on the
supported decode path and are used by ``TurboQuantKVCache`` automatically.

The ``fused_qk_scores_{2,3,4}bit`` kernels and ``pre_rotate_query`` utility
(in ``mlx_turboquant.rotation``) ship as *research-only primitives* and
are NOT wired into ``apply_turboquant`` or the decode path. They are
preserved here as library functions for users who want to experiment
with attention-from-compressed patterns in their own code.

We made two attempts to integrate them end-to-end — a decomposed SDPA
call (v0.8.0, branch ``feat/fused-sdpa-qwen3``) and a single-dispatch
fused attention kernel (v0.9.0, branch ``feat/full-fused-attention``).
Both were provably correct but lost to ``mx.fast.scaled_dot_product_attention``
at every tested decode shape. The structural reason: at realistic decode
shapes both paths are dispatch/latency/compute bound, not memory-bandwidth
bound, so the packed-KV memory advantage never materializes. Full
post-mortems in ``docs/FUSED_SDPA_RESULTS.md`` and
``docs/FULL_FUSED_ATTENTION_RESULTS.md``.

A source-level tripwire test
(``tests/test_fused_kernel_integration_tripwire.py``) enforces that these
primitives stay out of the supported decode path. If you are wiring them
in, delete the tripwire and replace it with a real end-to-end integration
test.
"""

import mlx.core as mx

# ============================================================
# Fused Dequantize: packed indices + norms → dequantized vectors
# ============================================================
#
# For 4-bit packing (2 indices per uint8 byte):
#   packed shape: (N, D/2) uint8
#   norms shape:  (N,) float32
#   centroids:    (16,) float32
#   rotation:     (D, D) float32 — we use rotation^T for inverse
#   output:       (N, D) float32
#
# Each thread computes one element of one output vector.
# Thread grid: (N * D, 1, 1) — one thread per output element.
# Each thread:
#   1. Determines which vector (row) and which output dimension (col)
#   2. Reads all D centroid values for this vector (from packed indices)
#   3. Computes dot product with one column of the rotation matrix (inverse rotation)
#   4. Scales by the vector's norm

_DEQUANT_4BIT_HEADER = """
"""

_DEQUANT_4BIT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint D_val = D;
    uint row = elem / D_val;       // which vector
    uint col = elem % D_val;       // which output dimension

    // Read norm for this vector
    float norm_val = norms[row];

    // Compute inverse-rotated value for this output dimension:
    // out[row, col] = norm * sum_j( centroid[idx_j] * rotation[col, j] )
    // where rotation is stored row-major and we're reading column 'col' across all rows j
    // But rotation^T[col, j] = rotation[j, col], and rotation is (D, D)
    // So we need: sum over j of centroid[packed_index(row, j)] * rotation[j * D_val + col]

    float dot = 0.0f;
    uint half_D = D_val / 2;
    for (uint j = 0; j < D_val; j++) {
        // Unpack the 4-bit index for position j of vector row
        uint byte_idx = j / 2;
        uint packed_byte = packed[row * half_D + byte_idx];
        uint idx;
        if (j % 2 == 0) {
            idx = packed_byte & 0x0Fu;
        } else {
            idx = (packed_byte >> 4) & 0x0Fu;
        }
        float centroid_val = centroids[idx];

        // rotation[j, col] — rotation is (D, D) row-major
        float rot_val = rotation[j * D_val + col];

        dot += centroid_val * rot_val;
    }

    out[elem] = norm_val * dot;
"""

_dequant_4bit_kernel = mx.fast.metal_kernel(
    name="turboquant_dequant_4bit",
    input_names=["packed", "norms", "centroids", "rotation"],
    output_names=["out"],
    source=_DEQUANT_4BIT_SOURCE,
    header=_DEQUANT_4BIT_HEADER,
)

# ============================================================
# Fused Dequantize for 2-bit (4 indices per uint8 byte)
# ============================================================

_DEQUANT_2BIT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint D_val = D;
    uint row = elem / D_val;
    uint col = elem % D_val;

    float norm_val = norms[row];

    float dot = 0.0f;
    uint quarter_D = D_val / 4;
    for (uint j = 0; j < D_val; j++) {
        uint byte_idx = j / 4;
        uint shift = (j % 4) * 2;
        uint packed_byte = packed[row * quarter_D + byte_idx];
        uint idx = (packed_byte >> shift) & 0x03u;
        float centroid_val = centroids[idx];
        float rot_val = rotation[j * D_val + col];
        dot += centroid_val * rot_val;
    }

    out[elem] = norm_val * dot;
"""

_dequant_2bit_kernel = mx.fast.metal_kernel(
    name="turboquant_dequant_2bit",
    input_names=["packed", "norms", "centroids", "rotation"],
    output_names=["out"],
    source=_DEQUANT_2BIT_SOURCE,
)

# ============================================================
# Fused Dequantize for 3-bit (8 indices per 3 bytes)
# ============================================================

_DEQUANT_3BIT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint D_val = D;
    uint row = elem / D_val;
    uint col = elem % D_val;

    float norm_val = norms[row];
    uint packed_stride = D_val * 3 / 8;

    float dot = 0.0f;
    for (uint j = 0; j < D_val; j++) {
        // 3-bit packing: 8 values per 3 bytes
        uint group = j / 8;
        uint pos = j % 8;
        uint base = row * packed_stride + group * 3;
        uint b0 = packed[base];
        uint b1 = packed[base + 1];
        uint b2 = packed[base + 2];

        uint idx;
        if (pos == 0)      idx = b0 & 0x07u;
        else if (pos == 1) idx = (b0 >> 3) & 0x07u;
        else if (pos == 2) idx = ((b0 >> 6) | (b1 << 2)) & 0x07u;
        else if (pos == 3) idx = (b1 >> 1) & 0x07u;
        else if (pos == 4) idx = (b1 >> 4) & 0x07u;
        else if (pos == 5) idx = ((b1 >> 7) | (b2 << 1)) & 0x07u;
        else if (pos == 6) idx = (b2 >> 2) & 0x07u;
        else               idx = (b2 >> 5) & 0x07u;

        float centroid_val = centroids[idx];
        float rot_val = rotation[j * D_val + col];
        dot += centroid_val * rot_val;
    }

    out[elem] = norm_val * dot;
"""

_dequant_3bit_kernel = mx.fast.metal_kernel(
    name="turboquant_dequant_3bit",
    input_names=["packed", "norms", "centroids", "rotation"],
    output_names=["out"],
    source=_DEQUANT_3BIT_SOURCE,
)

# ============================================================
# Fused Quantize: input vectors → packed indices + norms
# ============================================================
# For 4-bit:
#   input:      (N, D) float32
#   rotation:   (D, D) float32
#   boundaries: (15,) float32
#   output:     packed (N, D/2) uint8, norms (N,) float32
#
# Two-pass approach:
#   Pass 1: Compute norms (one thread per vector)
#   Pass 2: Normalize, rotate, quantize, pack (one thread per output byte)

_QUANT_4BIT_NORMS_SOURCE = """
    uint row = thread_position_in_grid.x;
    uint D_val = D;

    float sum_sq = 0.0f;
    for (uint j = 0; j < D_val; j++) {
        float v = inp[row * D_val + j];
        sum_sq += v * v;
    }
    norms[row] = metal::fast::sqrt(sum_sq);
"""

_quant_4bit_norms_kernel = mx.fast.metal_kernel(
    name="turboquant_quant_4bit_norms",
    input_names=["inp"],
    output_names=["norms"],
    source=_QUANT_4BIT_NORMS_SOURCE,
)

_QUANT_4BIT_PACK_SOURCE = """
    uint byte_idx = thread_position_in_grid.x;
    uint D_val = D;
    uint half_D = D_val / 2;
    uint row = byte_idx / half_D;
    uint col_pair = byte_idx % half_D;
    uint num_boundaries = 15;  // 4-bit = 16 levels = 15 boundaries

    float norm_val = norms[row];
    float inv_norm = (norm_val > 1e-10f) ? (1.0f / norm_val) : 0.0f;

    uint8_t result = 0;
    for (uint sub = 0; sub < 2; sub++) {
        uint j = col_pair * 2 + sub;  // original dimension

        // Normalize
        float val = inp[row * D_val + j] * inv_norm;

        // Rotate: rotated[j] = dot(input_normalized, rotation[j, :])
        // But we need rotated_j = sum_k( normalized_k * rotation[j, k] )
        // Wait — rotation maps input to rotated space: y = R * x
        // y[j] = sum_k R[j,k] * x[k]
        float rotated = 0.0f;
        for (uint k = 0; k < D_val; k++) {
            float x_k = inp[row * D_val + k] * inv_norm;
            rotated += rotation[j * D_val + k] * x_k;
        }

        // Quantize: count how many boundaries are exceeded
        uint idx = 0;
        for (uint b = 0; b < num_boundaries; b++) {
            idx += (rotated > boundaries[b]) ? 1 : 0;
        }

        if (sub == 0) {
            result = (uint8_t)(idx & 0x0Fu);
        } else {
            result |= (uint8_t)((idx & 0x0Fu) << 4);
        }
    }

    packed[byte_idx] = result;
"""

_quant_4bit_pack_kernel = mx.fast.metal_kernel(
    name="turboquant_quant_4bit_pack",
    input_names=["inp", "norms", "rotation", "boundaries"],
    output_names=["packed"],
    source=_QUANT_4BIT_PACK_SOURCE,
)


# ============================================================
# Public API
# ============================================================

def metal_dequantize(packed: mx.array, norms: mx.array, centroids: mx.array,
                     rotation: mx.array, bits: int, d: int) -> mx.array:
    """Fused Metal dequantize: unpack + centroid lookup + inverse rotation + scale.

    Args:
        packed: Packed indices, shape (N, packed_dim) uint8
        norms: Vector norms, shape (N,) float32
        centroids: Codebook centroids, shape (2^bits,) float32
        rotation: Rotation matrix, shape (D, D) float32
        bits: Bit-width (2, 3, or 4)
        d: Vector dimension

    Returns:
        Dequantized vectors, shape (N, D) float32
    """
    N = norms.shape[0]

    # Flatten packed to 2D for kernel
    packed_flat = packed.reshape(N, -1)
    norms_flat = norms.reshape(N)

    # Ensure float32 for rotation and centroids
    rotation_f32 = rotation.astype(mx.float32) if rotation.dtype != mx.float32 else rotation
    centroids_f32 = centroids.astype(mx.float32) if centroids.dtype != mx.float32 else centroids
    norms_f32 = norms_flat.astype(mx.float32) if norms_flat.dtype != mx.float32 else norms_flat

    total_elems = N * d
    tg_size = min(256, total_elems)

    if bits == 4:
        kernel = _dequant_4bit_kernel
    elif bits == 2:
        kernel = _dequant_2bit_kernel
    elif bits == 3:
        kernel = _dequant_3bit_kernel
    else:
        raise ValueError(f"Metal dequantize only supports 2, 3, 4 bits, got {bits}")

    result = kernel(
        inputs=[packed_flat, norms_f32, centroids_f32, rotation_f32],
        grid=(total_elems, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(N * d,)],
        output_dtypes=[mx.float32],
        template=[("D", d)],
    )

    return result[0].reshape(N, d)


def metal_quantize_4bit(inp: mx.array, rotation: mx.array,
                        boundaries: mx.array) -> tuple[mx.array, mx.array]:
    """Fused Metal quantize: norm + normalize + rotate + quantize + pack (4-bit).

    Args:
        inp: Input vectors, shape (N, D) float32
        rotation: Rotation matrix, shape (D, D) float32
        boundaries: Decision boundaries, shape (15,) float32

    Returns:
        (packed_indices, norms) — packed shape (N, D/2) uint8, norms shape (N,) float32
    """
    N, D = inp.shape

    inp_f32 = inp.astype(mx.float32) if inp.dtype != mx.float32 else inp
    rotation_f32 = rotation.astype(mx.float32) if rotation.dtype != mx.float32 else rotation
    boundaries_f32 = boundaries.astype(mx.float32) if boundaries.dtype != mx.float32 else boundaries

    # Pass 1: compute norms
    norms = _quant_4bit_norms_kernel(
        inputs=[inp_f32],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.float32],
        template=[("D", D)],
    )[0]

    # Pass 2: normalize + rotate + quantize + pack
    half_D = D // 2
    total_bytes = N * half_D
    tg_size = min(256, total_bytes)

    packed = _quant_4bit_pack_kernel(
        inputs=[inp_f32, norms, rotation_f32, boundaries_f32],
        grid=(total_bytes, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(total_bytes,)],
        output_dtypes=[mx.uint8],
        template=[("D", D)],
    )[0]

    return packed.reshape(N, half_D), norms


# ============================================================
# Fused QK Scores: Q_rot @ K_packed^T without dequantizing K
# ============================================================
#
# Computes attention scores directly from packed codebook indices:
#
#     score[q, k] = norms_k[k] * sum_j( q_rot[q, j] * centroids[idx(k, j)] )
#
# where q_rot = Q @ R.T is the pre-rotated query (one matmul per step, cheap)
# and idx(k, j) is the j-th codebook index for token k.
#
# This eliminates the per-step dequantize+matmul that dominates decode
# overhead at v0.6.0. See docs/FUSED_ATTENTION_DESIGN.md for the math.
#
# Grid layout (Phase 2, correctness-first):
#   (T_q * T_kv, 1, 1) — one thread per output score.
#   Each thread iterates D centroid lookups serially. Phase 4 will add
#   simd_sum + threadgroup shared memory for the D reduction.
#
# Inputs (all flat; mlx.fast.metal_kernel is row-contiguous by default):
#   q_rot      : (T_q * D,)        float32
#   packed_k   : (T_kv * packed_dim,) uint8
#   norms_k    : (T_kv,)           float32
#   centroids  : (2^bits,)         float32
# Output:
#   out        : (T_q * T_kv,)     float32

_FUSED_QK_4BIT_SOURCE = """
    // 2D grid: x = kv_idx, y = q_idx. T_kv and T_q are inferred from
    // grid dimensions set on the Python side.
    uint kv_idx = thread_position_in_grid.x;
    uint q_idx = thread_position_in_grid.y;
    uint T_kv = threads_per_grid.x;

    uint D_val = D;
    uint half_D = D_val / 2;

    // Threadgroup-shared copy of the 16 centroids. Every thread reads
    // them D times inside the inner loop; sharing saves one global read
    // per centroid lookup. 16 floats = 64 bytes, trivially fits.
    threadgroup float shared_centroids[16];
    uint local_id =
        thread_position_in_threadgroup.y * threads_per_threadgroup.x
        + thread_position_in_threadgroup.x;
    uint local_size =
        threads_per_threadgroup.x * threads_per_threadgroup.y;
    for (uint i = local_id; i < 16; i += local_size) {
        shared_centroids[i] = centroids[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    uint q_base = q_idx * D_val;
    uint k_base = kv_idx * half_D;
    for (uint j = 0; j < D_val; j++) {
        uint byte_idx = j >> 1;
        uint packed_byte = packed_k[k_base + byte_idx];
        uint idx;
        if ((j & 1u) == 0u) {
            idx = packed_byte & 0x0Fu;
        } else {
            idx = (packed_byte >> 4) & 0x0Fu;
        }
        float cent = shared_centroids[idx];
        acc += q_rot[q_base + j] * cent;
    }
    out[q_idx * T_kv + kv_idx] = norms_k[kv_idx] * acc;
"""

_fused_qk_4bit_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_qk_4bit",
    input_names=["q_rot", "packed_k", "norms_k", "centroids"],
    output_names=["out"],
    source=_FUSED_QK_4BIT_SOURCE,
)

_FUSED_QK_2BIT_SOURCE = """
    uint kv_idx = thread_position_in_grid.x;
    uint q_idx = thread_position_in_grid.y;
    uint T_kv = threads_per_grid.x;

    uint D_val = D;
    uint quarter_D = D_val / 4;

    float acc = 0.0f;
    for (uint j = 0; j < D_val; j++) {
        uint byte_idx = j >> 2;
        uint shift = (j & 3u) * 2u;
        uint packed_byte = packed_k[kv_idx * quarter_D + byte_idx];
        uint idx = (packed_byte >> shift) & 0x03u;
        float cent = centroids[idx];
        acc += q_rot[q_idx * D_val + j] * cent;
    }
    out[q_idx * T_kv + kv_idx] = norms_k[kv_idx] * acc;
"""

_fused_qk_2bit_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_qk_2bit",
    input_names=["q_rot", "packed_k", "norms_k", "centroids"],
    output_names=["out"],
    source=_FUSED_QK_2BIT_SOURCE,
)

# 3-bit packing: 8 values per 3 bytes. Same layout as _DEQUANT_3BIT_SOURCE
# uses above — lifted directly.
_FUSED_QK_3BIT_SOURCE = """
    uint kv_idx = thread_position_in_grid.x;
    uint q_idx = thread_position_in_grid.y;
    uint T_kv = threads_per_grid.x;

    uint D_val = D;
    uint packed_stride = D_val * 3 / 8;

    float acc = 0.0f;
    for (uint j = 0; j < D_val; j++) {
        uint group = j / 8;
        uint pos = j % 8;
        uint base = kv_idx * packed_stride + group * 3;
        uint b0 = packed_k[base];
        uint b1 = packed_k[base + 1];
        uint b2 = packed_k[base + 2];

        uint idx;
        if (pos == 0)      idx = b0 & 0x07u;
        else if (pos == 1) idx = (b0 >> 3) & 0x07u;
        else if (pos == 2) idx = ((b0 >> 6) | (b1 << 2)) & 0x07u;
        else if (pos == 3) idx = (b1 >> 1) & 0x07u;
        else if (pos == 4) idx = (b1 >> 4) & 0x07u;
        else if (pos == 5) idx = ((b1 >> 7) | (b2 << 1)) & 0x07u;
        else if (pos == 6) idx = (b2 >> 2) & 0x07u;
        else               idx = (b2 >> 5) & 0x07u;

        float cent = centroids[idx];
        acc += q_rot[q_idx * D_val + j] * cent;
    }
    out[q_idx * T_kv + kv_idx] = norms_k[kv_idx] * acc;
"""

_fused_qk_3bit_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_qk_3bit",
    input_names=["q_rot", "packed_k", "norms_k", "centroids"],
    output_names=["out"],
    source=_FUSED_QK_3BIT_SOURCE,
)


def _dispatch_fused_qk(kernel, q_rot, packed_k, norms_k, centroids, D):
    """Shared dispatch logic for the three bit-width variants.

    Uses a 2D grid (T_kv, T_q, 1). Each thread computes one score at
    position (q_idx, kv_idx). No division in the kernel, and T_kv is
    read from Metal's ``threads_per_grid.x`` at runtime — so the
    kernel compiles once per head_dim D, not once per (D, T_kv) pair.
    """
    T_q = q_rot.shape[0]
    T_kv = packed_k.shape[0]

    if T_kv == 0 or T_q == 0:
        return mx.zeros((T_q, T_kv), dtype=mx.float32)

    q_rot_f32 = q_rot.astype(mx.float32) if q_rot.dtype != mx.float32 else q_rot
    norms_f32 = norms_k.astype(mx.float32) if norms_k.dtype != mx.float32 else norms_k
    cent_f32 = centroids.astype(mx.float32) if centroids.dtype != mx.float32 else centroids

    # 2D threadgroup: keep it 1D in the T_kv direction for SIMD-friendliness,
    # since T_kv is typically the large axis (thousands of compressed tokens)
    # and T_q is tiny (1 for decode, tens for prefill).
    tg_x = min(256, T_kv)
    tg_y = min(max(1, 256 // max(1, tg_x)), T_q)

    result = kernel(
        inputs=[q_rot_f32, packed_k, norms_f32, cent_f32],
        output_shapes=[(T_q * T_kv,)],
        output_dtypes=[mx.float32],
        grid=(T_kv, T_q, 1),
        threadgroup=(tg_x, tg_y, 1),
        template=[("D", D)],
    )
    return result[0].reshape(T_q, T_kv)


def fused_qk_scores_4bit(q_rot: mx.array, packed_k: mx.array,
                         norms_k: mx.array, centroids: mx.array,
                         D: int) -> mx.array:
    """Fused Q @ K^T for 4-bit packed K — no dequantization materialized.

    Args:
        q_rot: Pre-rotated query vectors, shape (T_q, D) float32.
               Obtain via ``pre_rotate_query(query, rotation)``.
        packed_k: Packed 4-bit K indices, shape (T_kv, D/2) uint8.
        norms_k: Per-token K norms, shape (T_kv,) float32.
        centroids: Codebook centroids, shape (16,) float32.
        D: Head dimension.

    Returns:
        Raw attention scores, shape (T_q, T_kv) float32. Pre-softmax,
        pre-scale, pre-mask. The caller is responsible for softmax,
        scaling, and causal masking — typically by combining these
        scores with the FP16 region scores before the softmax.
    """
    return _dispatch_fused_qk(_fused_qk_4bit_kernel, q_rot, packed_k,
                              norms_k, centroids, D)


def fused_qk_scores_3bit(q_rot: mx.array, packed_k: mx.array,
                         norms_k: mx.array, centroids: mx.array,
                         D: int) -> mx.array:
    """Fused Q @ K^T for 3-bit packed K."""
    return _dispatch_fused_qk(_fused_qk_3bit_kernel, q_rot, packed_k,
                              norms_k, centroids, D)


def fused_qk_scores_2bit(q_rot: mx.array, packed_k: mx.array,
                         norms_k: mx.array, centroids: mx.array,
                         D: int) -> mx.array:
    """Fused Q @ K^T for 2-bit packed K."""
    return _dispatch_fused_qk(_fused_qk_2bit_kernel, q_rot, packed_k,
                              norms_k, centroids, D)
