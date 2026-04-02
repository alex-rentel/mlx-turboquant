"""Bit-packing utilities for sub-byte quantized indices.

Efficiently pack 2-bit, 3-bit, and 4-bit indices into uint8 arrays.
"""

import mlx.core as mx


def pack_2bit(indices: mx.array) -> mx.array:
    """Pack 2-bit indices: 4 values per byte.

    Args:
        indices: uint8 values in [0, 3], shape (..., d) where d is divisible by 4

    Returns:
        Packed uint8 array of shape (..., d // 4)
    """
    shape = indices.shape
    d = shape[-1]
    assert d % 4 == 0, f"Last dimension {d} must be divisible by 4 for 2-bit packing"

    flat = indices.reshape(-1, d)
    # Group into fours
    grouped = flat.reshape(-1, d // 4, 4)

    # Pack: val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)
    packed = (grouped[..., 0].astype(mx.uint8)
              | (grouped[..., 1].astype(mx.uint8) << 2)
              | (grouped[..., 2].astype(mx.uint8) << 4)
              | (grouped[..., 3].astype(mx.uint8) << 6))

    return packed.reshape(*shape[:-1], d // 4)


def unpack_2bit(packed: mx.array, d: int) -> mx.array:
    """Unpack 2-bit packed indices back to uint8.

    Args:
        packed: Packed uint8 array (..., d // 4)
        d: Original last dimension

    Returns:
        uint8 indices in [0, 3], shape (..., d)
    """
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1])
    n = flat.shape[0]

    v0 = flat & 0x03
    v1 = (flat >> 2) & 0x03
    v2 = (flat >> 4) & 0x03
    v3 = (flat >> 6) & 0x03

    # Interleave: stack and reshape
    result = mx.stack([v0, v1, v2, v3], axis=-1)  # (n, d//4, 4)
    result = result.reshape(n, d)
    return result.reshape(*shape[:-1], d)


def pack_3bit(indices: mx.array) -> mx.array:
    """Pack 3-bit indices: 8 values per 3 bytes.

    Args:
        indices: uint8 values in [0, 7], shape (..., d) where d is divisible by 8

    Returns:
        Packed uint8 array of shape (..., d * 3 // 8)
    """
    shape = indices.shape
    d = shape[-1]
    assert d % 8 == 0, f"Last dimension {d} must be divisible by 8 for 3-bit packing"

    flat = indices.reshape(-1, d).astype(mx.uint32)
    n = flat.shape[0]

    # Group into 8s
    grouped = flat.reshape(n, d // 8, 8)

    # Pack 8 x 3-bit values into 3 bytes (24 bits)
    # byte0: v0[2:0] | v1[2:0] | v2[1:0]
    # byte1: v2[2] | v3[2:0] | v4[2:0] | v5[1:0]
    # byte2: v5[2] | v6[2:0] | v7[2:0] | unused[1:0]
    byte0 = (grouped[..., 0]
             | (grouped[..., 1] << 3)
             | (grouped[..., 2] << 6))
    byte1 = ((grouped[..., 2] >> 2)
             | (grouped[..., 3] << 1)
             | (grouped[..., 4] << 4)
             | (grouped[..., 5] << 7))
    byte2 = ((grouped[..., 5] >> 1)
             | (grouped[..., 6] << 2)
             | (grouped[..., 7] << 5))

    packed = mx.stack([byte0.astype(mx.uint8),
                       byte1.astype(mx.uint8),
                       byte2.astype(mx.uint8)], axis=-1)
    return packed.reshape(*shape[:-1], d * 3 // 8)


def unpack_3bit(packed: mx.array, d: int) -> mx.array:
    """Unpack 3-bit packed indices back to uint8.

    Args:
        packed: Packed uint8 array (..., d * 3 // 8)
        d: Original last dimension

    Returns:
        uint8 indices in [0, 7], shape (..., d)
    """
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1]).astype(mx.uint32)
    n = flat.shape[0]

    # Group into 3-byte groups
    grouped = flat.reshape(n, d // 8, 3)
    b0, b1, b2 = grouped[..., 0], grouped[..., 1], grouped[..., 2]

    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x07
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x07
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07

    result = mx.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=-1)
    result = result.reshape(n, d)
    return result.reshape(*shape[:-1], d).astype(mx.uint8)


def pack_4bit(indices: mx.array) -> mx.array:
    """Pack 4-bit indices: 2 values per byte.

    Args:
        indices: uint8 values in [0, 15], shape (..., d) where d is divisible by 2

    Returns:
        Packed uint8 array of shape (..., d // 2)
    """
    shape = indices.shape
    d = shape[-1]
    assert d % 2 == 0, f"Last dimension {d} must be divisible by 2 for 4-bit packing"

    flat = indices.reshape(-1, d)
    # Group into pairs
    grouped = flat.reshape(-1, d // 2, 2)

    packed = (grouped[..., 0].astype(mx.uint8)
              | (grouped[..., 1].astype(mx.uint8) << 4))

    return packed.reshape(*shape[:-1], d // 2)


def unpack_4bit(packed: mx.array, d: int) -> mx.array:
    """Unpack 4-bit packed indices back to uint8.

    Args:
        packed: Packed uint8 array (..., d // 2)
        d: Original last dimension

    Returns:
        uint8 indices in [0, 15], shape (..., d)
    """
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1])

    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F

    result = mx.stack([lo, hi], axis=-1)
    result = result.reshape(-1, d)
    return result.reshape(*shape[:-1], d)


def pack_1bit(indices: mx.array) -> mx.array:
    """Pack 1-bit indices: 8 values per byte."""
    shape = indices.shape
    d = shape[-1]
    assert d % 8 == 0, f"Last dimension {d} must be divisible by 8 for 1-bit packing"

    flat = indices.reshape(-1, d)
    grouped = flat.reshape(-1, d // 8, 8)

    packed = (grouped[..., 0].astype(mx.uint8)
              | (grouped[..., 1].astype(mx.uint8) << 1)
              | (grouped[..., 2].astype(mx.uint8) << 2)
              | (grouped[..., 3].astype(mx.uint8) << 3)
              | (grouped[..., 4].astype(mx.uint8) << 4)
              | (grouped[..., 5].astype(mx.uint8) << 5)
              | (grouped[..., 6].astype(mx.uint8) << 6)
              | (grouped[..., 7].astype(mx.uint8) << 7))

    return packed.reshape(*shape[:-1], d // 8)


def unpack_1bit(packed: mx.array, d: int) -> mx.array:
    """Unpack 1-bit packed indices back to uint8."""
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1])

    bits = []
    for i in range(8):
        bits.append((flat >> i) & 0x01)

    result = mx.stack(bits, axis=-1)
    result = result.reshape(-1, d)
    return result.reshape(*shape[:-1], d)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack indices at the given bit-width."""
    if bits == 1:
        return pack_1bit(indices)
    elif bits == 2:
        return pack_2bit(indices)
    elif bits == 3:
        return pack_3bit(indices)
    elif bits == 4:
        return pack_4bit(indices)
    else:
        raise ValueError(f"Unsupported bit-width: {bits}. Use 1, 2, 3, or 4.")


def unpack_indices(packed: mx.array, bits: int, d: int) -> mx.array:
    """Unpack indices at the given bit-width."""
    if bits == 1:
        return unpack_1bit(packed, d)
    elif bits == 2:
        return unpack_2bit(packed, d)
    elif bits == 3:
        return unpack_3bit(packed, d)
    elif bits == 4:
        return unpack_4bit(packed, d)
    else:
        raise ValueError(f"Unsupported bit-width: {bits}. Use 1, 2, 3, or 4.")
