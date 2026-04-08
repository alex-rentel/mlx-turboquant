"""mlx-turboquant: Near-optimal KV cache quantization for Apple Silicon.

Public API (stable as of v1.0.0)
--------------------------------

Top-level integration:
    apply_turboquant(model, key_bits=4, value_bits=2, ...)
        The primary entry point. Monkey-patches an mlx-lm model's
        make_cache() so every layer uses TurboQuantKVCache.

    enable_turboquant(model, bits=4, **kwargs)
        Convenience wrapper for symmetric bit allocation.

Cache type (advanced users):
    TurboQuantKVCache
        The KV cache class itself. Drop-in replacement for mlx-lm's
        KVCache. Most users should never need to construct this directly;
        use apply_turboquant() instead.

Low-level quantizers (for custom attention kernels only):
    TurboQuantMSE    — Algorithm 1 (MSE-optimized). Used by the cache.
    TurboQuantProd   — Algorithm 2 (inner-product optimized, experimental).

Everything else (``mlx_turboquant.kernels``, ``.rotation``, ``.codebook``,
``.packing``, ``.qjl``) is internal. Those modules may break between minor
versions without deprecation warnings. In particular, the fused QK score
kernels in ``mlx_turboquant.kernels`` ship as research-only primitives and
are NOT part of the supported decode path — see the kernels module
docstring for details.

Versioning policy
-----------------

Semantic versioning from v1.0.0 forward:
- MAJOR: breaking changes to the public API listed above
- MINOR: new features, new models validated, non-breaking changes
- PATCH: bug fixes, documentation, benchmark updates

Deprecations will be announced via DeprecationWarning for at least one
minor version before removal.
"""

__version__ = "1.0.0"

from .cache import TurboQuantKVCache
from .patch import apply_turboquant, enable_turboquant
from .quantizer import TurboQuantMSE, TurboQuantProd

__all__ = [
    "__version__",
    # Primary integration
    "apply_turboquant",
    "enable_turboquant",
    # Cache class (advanced)
    "TurboQuantKVCache",
    # Low-level quantizers (custom kernels)
    "TurboQuantMSE",
    "TurboQuantProd",
]
