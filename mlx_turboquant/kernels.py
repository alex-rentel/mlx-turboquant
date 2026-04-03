"""Fused Metal kernels for TurboQuant quantization and dequantization.

These kernels eliminate intermediate array allocations by fusing:
- Dequantize: unpack indices + centroid lookup + inverse rotation + norm scaling
- Quantize: norm computation + normalize + rotation + boundary search + packing

Each kernel runs as a single Metal dispatch, avoiding Python overhead and
intermediate tensor materialization.
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
