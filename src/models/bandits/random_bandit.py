import random

import numpy as np

from src.models.bandits.bandit import Bandit


class RandomBandit(Bandit):
    """Baseline Bandit, chooses a random arm at each timestep."""

    def __init__(self, boost: float | None = None):
        super().__init__(boost=boost)

    def select_arm_idx(self, arms_pool: list, context: np.ndarray | list | None = None):
        self._init_arms(arms_pool)
        return random.randrange(len(arms_pool))
