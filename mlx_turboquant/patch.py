"""Model patching — apply TurboQuant KV cache to any mlx-lm model.

Replaces the default KV cache with TurboQuantKVCache for each layer.
Works with Llama, Qwen, Mistral, Gemma, and other mlx-lm models.
"""


import mlx.nn as nn

from .cache import TurboQuantKVCache


def _get_model_config(model: nn.Module) -> dict:
    """Extract head_dim and num_kv_heads from model config."""
    # Navigate to the inner model — handle nested wrappers like
    # Gemma3ForConditionalGeneration > language_model > model
    inner = getattr(model, "model", model)

    # Try multiple paths to find args (handles different model architectures)
    args = None
    for path in [
        inner,                                          # model.model
        model,                                          # model
        getattr(model, "language_model", None),         # Gemma3 conditional
        getattr(getattr(model, "language_model", None), "model", None),  # Gemma3 inner
    ]:
        if path is not None:
            args = getattr(path, "args", None)
            if args is not None and getattr(args, "hidden_size", None) is not None:
                break

    if args is not None:
        # Standard mlx-lm config attributes
        head_dim = getattr(args, "head_dim", None)
        num_kv_heads = getattr(args, "num_key_value_heads", None)
        num_heads = getattr(args, "num_attention_heads", None)
        hidden_size = getattr(args, "hidden_size", None)
        num_layers = getattr(args, "num_hidden_layers", None)

        # Compute head_dim if not directly available
        if head_dim is None and hidden_size is not None and num_heads is not None:
            head_dim = hidden_size // num_heads

        # Some models use num_key_value_heads, others use num_kv_heads
        if num_kv_heads is None:
            num_kv_heads = getattr(args, "num_kv_heads", num_heads)

        if num_layers is None:
            # Count layers — search multiple paths for nested architectures
            for candidate in [inner, model, getattr(model, "language_model", None),
                              getattr(getattr(model, "language_model", None), "model", None)]:
                if candidate is not None:
                    layers = getattr(candidate, "layers", None)
                    if layers is not None and len(layers) > 0:
                        num_layers = len(layers)
                        break

        return {
            "head_dim": head_dim or 128,
            "num_kv_heads": num_kv_heads or 1,
            "num_layers": num_layers or 1,
        }

    # Fallback: inspect first attention layer
    layers = getattr(inner, "layers", None)
    if layers is not None and len(layers) > 0:
        layer = layers[0]
        attn = getattr(layer, "self_attn", getattr(layer, "attention", None))
        if attn is not None:
            # Try to extract from projection weight shapes
            k_proj = getattr(attn, "k_proj", None)
            if k_proj is not None:
                weight = k_proj.weight
                # k_proj weight: (num_kv_heads * head_dim, hidden_size)
                out_features = weight.shape[0]
                num_kv_heads_guess = getattr(attn, "n_kv_heads",
                                             getattr(attn, "num_kv_heads", 1))
                head_dim = out_features // num_kv_heads_guess
                return {
                    "head_dim": head_dim,
                    "num_kv_heads": num_kv_heads_guess,
                    "num_layers": len(layers),
                }

    raise ValueError("Could not determine model configuration. "
                     "Please specify head_dim and num_kv_heads manually.")


def detect_outlier_layers(model: nn.Module, threshold: float = 3.0) -> list[int]:
    """Detect layers with extreme key norms that should stay in FP16.

    Runs a short forward pass and measures key norm statistics per layer.
    Layers where max key norm exceeds threshold * median are marked as outliers.
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import KVCache

    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", [])
    num_layers = len(layers)

    # Run a short forward pass with standard cache
    cache = [KVCache() for _ in range(num_layers)]
    dummy = mx.array([[0, 1, 2, 3, 4]])  # Token ID 0 is nearly always valid
    logits = model(dummy, cache=cache)
    mx.eval(logits)

    # Measure max key norms per layer
    max_norms = []
    for c in cache:
        if c.keys is not None:
            k = c.keys.astype(mx.float32)
            norms = mx.linalg.norm(k, axis=-1)
            max_norms.append(mx.max(norms).item())
        else:
            max_norms.append(0.0)

    max_norms = np.array(max_norms)
    positive = max_norms[max_norms > 0]
    if positive.size == 0:
        # No layer produced a positive key norm — nothing reliable to compare
        # against. Return empty rather than propagating NaN through the
        # threshold check (which would silently mark every layer as non-outlier).
        return []
    median_norm = float(np.median(positive))

    outliers = []
    for i, n in enumerate(max_norms):
        if n > threshold * median_norm:
            outliers.append(i)

    return outliers


def apply_turboquant(
    model: nn.Module,
    key_bits: float = 4,
    value_bits: float = 2,
    residual_window: int = 128,
    head_dim: int | None = None,
    num_kv_heads: int | None = None,
    num_layers: int | None = None,
    rotation_seed: int = 42,
    skip_layers: list[int] | None = None,
    auto_detect_outliers: bool = True,
    fp16_sink_size: int = 0,
    chunk_size: int = 0,
    qjl_correction: bool = False,
    qjl_n_proj: int = 32,
) -> nn.Module:
    """Apply TurboQuant KV cache compression to an mlx-lm model.

    Monkey-patches the model's make_cache() method to return
    TurboQuantKVCache instances instead of the default KVCache.

    Args:
        model: An mlx-lm model (Llama, Qwen, Mistral, Gemma, etc.)
        key_bits: Bit-width for key quantization (2, 3, 3.5, or 4).
            Fractional values quantize the high half of the rotated head
            at the integer ceiling and the low half at the floor.
        value_bits: Bit-width for value quantization (2, 3, 3.5, or 4).
        residual_window: Number of recent tokens to keep in FP16 (sliding)
        head_dim: Override head dimension (auto-detected if None)
        num_kv_heads: Override number of KV heads (auto-detected if None)
        num_layers: Override number of layers (auto-detected if None)
        rotation_seed: Seed for the rotation matrix
        skip_layers: Layer indices to keep in FP16 (no compression)
        auto_detect_outliers: Auto-detect outlier layers (default True)
        fp16_sink_size: Number of leading tokens (e.g., system prompt) to
            permanently store in FP16. These never get compressed and are
            independent of `residual_window`. 0 disables the sink (default).
        chunk_size: Compression strategy selector.
            - 0 (default): v0.5.0 batch path — single variable-size drain
              when fp16 buffer reaches 2x residual_window. Fastest in
              v0.6.0 benchmarks.
            - >0: Drain in fixed-size chunks of `chunk_size` tokens
              whenever the buffer exceeds residual_window + chunk_size.
              Architecturally cleaner; opt in if you need stable Metal
              kernel input shapes.
        qjl_correction: When True, apply a 1-bit QJL sign-sketch correction
            to the dequantized cache at compression time. Improves cosine
            similarity at the cost of ~5% extra compute per compression
            chunk. Zero memory overhead — the sketch is consumed at
            compression time, not stored. Default False.
        qjl_n_proj: Number of QJL random projections (only used when
            qjl_correction=True). Higher = more accurate correction at
            slightly more compute. Default 32.

    Returns:
        The same model (modified in-place)
    """
    config = _get_model_config(model)
    hd = head_dim or config["head_dim"]
    nkv = num_kv_heads or config["num_kv_heads"]
    nl = num_layers or config["num_layers"]

    # Auto-adjust bits for models with few KV heads (more sensitive to compression)
    import warnings
    if nkv <= 2 and key_bits < 4:
        original_key_bits = key_bits
        key_bits = 4
        warnings.warn(
            f"Model has only {nkv} KV heads — upgrading key_bits "
            f"{original_key_bits} → {key_bits} for stability. "
            f"Effective config: key_bits={key_bits}, value_bits={value_bits}.",
            UserWarning, stacklevel=2,
        )
    if nkv <= 2 and value_bits < 3:
        original_value_bits = value_bits
        value_bits = 3
        warnings.warn(
            f"Model has only {nkv} KV heads — upgrading value_bits "
            f"{original_value_bits} → {value_bits} for stability. "
            f"Effective config: key_bits={key_bits}, value_bits={value_bits}.",
            UserWarning, stacklevel=2,
        )

    # Determine which layers to skip
    layers_to_skip = set(skip_layers or [])
    if auto_detect_outliers and not skip_layers:
        try:
            outliers = detect_outlier_layers(model)
            layers_to_skip = set(outliers)
        except Exception:
            pass  # If detection fails, compress all layers

    # Detect hybrid attention layers (e.g. Qwen3.5: alternating linear_attn
    # and self_attn). Linear-attention layers have a fundamentally different
    # state shape and cannot use TurboQuantKVCache; we must hand them the
    # cache type the model itself wants. Detection is structural — we look
    # for linear_attn / linear_attention attributes on each layer.
    inner_for_layers = getattr(model, "model", model)
    raw_layers = getattr(inner_for_layers, "layers", None)
    if raw_layers is None:
        raw_layers = getattr(model, "layers", []) or []

    linear_attn_layers = set()
    for i, layer in enumerate(raw_layers):
        if hasattr(layer, "linear_attn") or hasattr(layer, "linear_attention"):
            # A layer has linear attention if it has a linear_attn module AND
            # does NOT have self_attn (or has both but linear_attn is the one
            # actually used). For Qwen3.5 the layer type alternates, so we
            # check which module is actually present and active by attribute.
            has_self = hasattr(layer, "self_attn") or hasattr(layer, "attention")
            if not has_self:
                linear_attn_layers.add(i)
            else:
                # Mixed: prefer self_attn for compression unless linear is the
                # only path. Conservative: if both exist, treat as self_attn
                # and let the model route normally.
                pass

    has_hybrid = bool(linear_attn_layers)
    if has_hybrid:
        import warnings
        warnings.warn(
            f"Detected {len(linear_attn_layers)} linear-attention layers in "
            f"model (hybrid architecture, e.g. Qwen3.5). These layers will "
            f"use the model's default cache type instead of TurboQuantKVCache.",
            UserWarning, stacklevel=2,
        )

    from mlx_lm.models.cache import KVCache, make_prompt_cache

    # Capture the model's CLASS-LEVEL make_cache (if any) before we shadow
    # it with our own instance attribute. For hybrid architectures this is
    # the model's architecture-aware cache builder; we need it to populate
    # the linear-attention slots without recursing back through our own
    # make_cache (which is what would happen if we called the high-level
    # make_prompt_cache helper, since it defers to model.make_cache when
    # present).
    model_class = type(model)
    class_level_make_cache = getattr(model_class, "make_cache", None)

    def _build_default_caches():
        """Build the model's preferred cache list, bypassing our patch."""
        if class_level_make_cache is not None:
            return class_level_make_cache(model)
        # Fallback: use a temporary detachment trick. Save and remove our
        # instance attr, call make_prompt_cache, restore.
        instance_attr = model.__dict__.pop("make_cache", None)
        try:
            return make_prompt_cache(model)
        finally:
            if instance_attr is not None:
                model.make_cache = instance_attr

    def make_cache():
        # For hybrid models, get the model's preferred cache list once per
        # call so we can copy the linear-attention slots verbatim.
        default_caches = _build_default_caches() if has_hybrid else None

        caches = []
        for i in range(nl):
            if i in linear_attn_layers:
                # Hand back the model's own cache type for this layer.
                caches.append(default_caches[i])
            elif i in layers_to_skip:
                caches.append(KVCache())  # FP16 for outlier layers
            else:
                caches.append(TurboQuantKVCache(
                    head_dim=hd,
                    num_kv_heads=nkv,
                    key_bits=key_bits,
                    value_bits=value_bits,
                    residual_window=residual_window,
                    rotation_seed=rotation_seed,
                    fp16_sink_size=fp16_sink_size,
                    chunk_size=chunk_size,
                    qjl_correction=qjl_correction,
                    qjl_n_proj=qjl_n_proj,
                ))
        return caches

    # Monkey-patch
    model.make_cache = make_cache

    # Store config for introspection
    model._turboquant_config = {
        "head_dim": hd,
        "num_kv_heads": nkv,
        "num_layers": nl,
        "key_bits": key_bits,
        "value_bits": value_bits,
        "residual_window": residual_window,
        "fp16_sink_size": fp16_sink_size,
        "chunk_size": chunk_size,
        "qjl_correction": qjl_correction,
        "qjl_n_proj": qjl_n_proj,
    }

    return model


def enable_turboquant(model: nn.Module, bits: int = 4, **kwargs) -> nn.Module:
    """Convenience wrapper: enable TurboQuant with symmetric bit allocation.

    Args:
        model: An mlx-lm model
        bits: Bit-width for both keys and values
        **kwargs: Additional arguments passed to apply_turboquant

    Returns:
        The same model (modified in-place)
    """
    return apply_turboquant(model, key_bits=bits, value_bits=bits, **kwargs)
