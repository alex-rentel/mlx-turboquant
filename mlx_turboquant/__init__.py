"""mlx-turboquant: Near-optimal KV cache quantization for Apple Silicon."""

__version__ = "0.5.0"

from .cache import TurboQuantKVCache
from .patch import apply_turboquant, enable_turboquant
from .quantizer import TurboQuantMSE, TurboQuantProd
