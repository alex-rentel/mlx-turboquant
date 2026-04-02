"""Model patching — apply TurboQuant KV cache to any mlx-lm model.

Replaces the default KV cache with TurboQuantKVCache for each layer.
Works with Llama, Qwen, Mistral, Gemma, and other mlx-lm models.
"""

from typing import Optional

import mlx.nn as nn

from .cache import TurboQuantKVCache


def _get_model_config(model: nn.Module) -> dict:
    """Extract head_dim and num_kv_heads from model config."""
    # Navigate to the inner model (e.g., model.model for LlamaForCausalLM)
    inner = getattr(model, "model", model)

    # Try to get config from model args
    args = getattr(inner, "args", None)
    if args is None:
        args = getattr(model, "args", None)

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
            # Count layers
            layers = getattr(inner, "layers", None)
            if layers is not None:
                num_layers = len(layers)

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


def apply_turboquant(
    model: nn.Module,
    key_bits: int = 4,
    value_bits: int = 2,
    residual_window: int = 128,
    head_dim: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    rotation_seed: int = 42,
) -> nn.Module:
    """Apply TurboQuant KV cache compression to an mlx-lm model.

    Monkey-patches the model's make_cache() method to return
    TurboQuantKVCache instances instead of the default KVCache.

    Args:
        model: An mlx-lm model (Llama, Qwen, Mistral, Gemma, etc.)
        key_bits: Bit-width for key quantization (2, 3, or 4)
        value_bits: Bit-width for value quantization (2, 3, or 4)
        residual_window: Number of recent tokens to keep in FP16
        head_dim: Override head dimension (auto-detected if None)
        num_kv_heads: Override number of KV heads (auto-detected if None)
        num_layers: Override number of layers (auto-detected if None)
        rotation_seed: Seed for the rotation matrix

    Returns:
        The same model (modified in-place)
    """
    config = _get_model_config(model)
    hd = head_dim or config["head_dim"]
    nkv = num_kv_heads or config["num_kv_heads"]
    nl = num_layers or config["num_layers"]

    def make_cache():
        return [
            TurboQuantKVCache(
                head_dim=hd,
                num_kv_heads=nkv,
                key_bits=key_bits,
                value_bits=value_bits,
                residual_window=residual_window,
                rotation_seed=rotation_seed,
            )
            for _ in range(nl)
        ]

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
