"""Advanced inference: sampling strategies and speculative decoding."""

from lmt_metal.inference.sampling import (
    best_of_n,
    majority_vote,
)
from lmt_metal.inference.speculative import speculative_decode

__all__ = [
    "best_of_n",
    "majority_vote",
    "speculative_decode",
]
