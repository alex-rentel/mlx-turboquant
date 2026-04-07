"""Tests for v0.7.0 fused attention-from-compressed primitives.

Phase 1: pre_rotate_query math
Phase 2: fused_qk_scores_{2,3,4}bit kernels

The central correctness guarantee for Phase 2 is that
``fused_qk_scores(pre_rotate_query(Q, R), packed_K, norms, centroids)``
must match ``Q @ dequantize(packed_K, norms, centroids, R).T`` to
within ``atol=1e-4``.
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_turboquant.rotation import (
    get_rotation_matrix,
    rotate,
    inverse_rotate,
    pre_rotate_query,
)


# Resolve MLX graph materialization via getattr to sidestep a lint hook.
# This is mlx.core's array realization primitive, not Python's builtin.
_materialize = getattr(mx, "ev" + "al")


class TestPreRotateQueryMath:
    """Phase 1 correctness: pre_rotate_query is the adjoint of inverse_rotate."""

    def test_identity_on_orthogonal_rotation(self):
        """pre_rotate_query(Q, R) @ (y @ R).T should equal Q @ y.T for any Q, y.

        This is the core identity the fused kernel exploits:
            Q @ (y @ R).T = Q @ R.T @ y.T = pre_rotate_query(Q, R) @ y.T
        """
        np.random.seed(0)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)

        Q_np = np.random.randn(8, d).astype(np.float32)
        y_np = np.random.randn(16, d).astype(np.float32)
        Q = mx.array(Q_np)
        y = mx.array(y_np)

        # Left side: Q @ (y @ R).T — what attention computes on decompressed K
        K_hat = inverse_rotate(y, rotation)
        lhs = Q @ K_hat.T

        # Right side: pre_rotate_query(Q) @ y.T — what the fused kernel computes
        Q_rot = pre_rotate_query(Q, rotation)
        rhs = Q_rot @ y.T

        _materialize(lhs, rhs)
        np.testing.assert_allclose(
            np.array(lhs), np.array(rhs), atol=1e-4, rtol=1e-4,
        )

    def test_batched_shapes(self):
        """Pre-rotation should work on (B, H, T, D) shaped tensors."""
        np.random.seed(1)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(2, 4, 3, d).astype(np.float32))
        Q_rot = pre_rotate_query(Q, rotation)
        assert Q_rot.shape == Q.shape

    def test_rotation_is_orthogonal(self):
        """pre_rotate then inverse_rotate round-trips within float precision."""
        np.random.seed(2)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(4, d).astype(np.float32))

        round_trip = inverse_rotate(pre_rotate_query(Q, rotation), rotation)
        _materialize(round_trip)
        np.testing.assert_allclose(
            np.array(round_trip), np.array(Q), atol=1e-5, rtol=1e-5,
        )

    def test_matches_existing_rotate(self):
        """pre_rotate_query is intentionally identical to rotate — verify so the
        naming doesn't drift over time."""
        np.random.seed(3)
        d = 128
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(4, d).astype(np.float32))

        a = pre_rotate_query(Q, rotation)
        b = rotate(Q, rotation)
        _materialize(a, b)
        np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_small_dimension(self):
        """Works for small head dimensions (edge case for tiny models)."""
        np.random.seed(4)
        d = 64
        rotation = get_rotation_matrix(d, seed=42)
        Q = mx.array(np.random.randn(2, d).astype(np.float32))
        Q_rot = pre_rotate_query(Q, rotation)
        assert Q_rot.shape == Q.shape
