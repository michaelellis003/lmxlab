"""Advanced inference: sampling strategies and speculative decoding."""

from lmxlab.inference.sampling import (
    best_of_n,
    majority_vote,
)
from lmxlab.inference.speculative import speculative_decode

__all__ = [
    "best_of_n",
    "majority_vote",
    "speculative_decode",
]
