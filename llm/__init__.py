import importlib.metadata

__version__ = importlib.metadata.version("llm")

# register the custom torch.ops
from . import flash_attention
