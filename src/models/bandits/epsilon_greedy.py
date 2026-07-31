import random

import numpy as np

from src.models.bandits.bandit import Bandit


class EpsilonGreedyBandit(Bandit):
    """Epsilon-Greedy Bandit, arms defined by (category, sub category, venue type) combinations."""  # noqa: E501

    def __init__(self, epsilon=0.1, boost: float | None = None):
        super().__init__(boost=boost)
        self.epsilon = epsilon

    def select_arm_idx(
        self, arms_pool: list, context: np.ndarray | list | None = None
    ) -> int:
        if not self.initialized:
            self._init_arms(arms_pool)

        if random.random() < self.epsilon:
            return random.randrange(len(arms_pool))

        if len(arms_pool) == len(self.all_arms):
            pool_rewards = self.arm_total_reward
            pool_counts = self.arm_counts
        else:
            global_indices = [self.arm_to_idx[arm] for arm in arms_pool]
            pool_rewards = self.arm_total_reward[global_indices]
            pool_counts = self.arm_counts[global_indices]

        avg_rewards = np.divide(
            pool_rewards,
            pool_counts,
            out=np.zeros_like(pool_rewards, dtype=float),
            where=pool_counts != 0,
        )

        return int(np.argmax(avg_rewards))
