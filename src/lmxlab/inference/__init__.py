"""Advanced inference: sampling strategies and speculative decoding."""

from lmxlab.inference.beam_search import beam_search
from lmxlab.inference.reward_model import RewardModel
from lmxlab.inference.sampling import (
    best_of_n,
    majority_vote,
)
from lmxlab.inference.speculative import speculative_decode

__all__ = [
    "RewardModel",
    "beam_search",
    "best_of_n",
    "majority_vote",
    "speculative_decode",
]
