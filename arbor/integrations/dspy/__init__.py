"""DSPy integration for Arbor.

Exports Arbor's DSPy provider and GRPO trainer in a structured namespace.
"""

from .arbor_provider import ArborProvider
from .grpo_optimizer import ArborGRPO

__all__ = ["ArborGRPO", "ArborProvider"]
