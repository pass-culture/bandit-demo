import numpy as np

from src.models.bandits.epsilon_greedy import EpsilonGreedyBandit


class DecayingEpsilonGreedyBandit(EpsilonGreedyBandit):
    """Epsilon-Greedy Bandit with a decaying exploration rate over time."""

    def __init__(
        self,
        initial_epsilon=1.0,
        decay_rate=0.99,
        min_epsilon=0.01,
        boost: float | None = None,
    ):
        super().__init__(epsilon=initial_epsilon, boost=boost)
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def select_arm_idx(self, arms_pool: list, context: np.ndarray | list | None = None):
        chosen_arm = super().select_arm_idx(arms_pool, context)
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)

        return chosen_arm
