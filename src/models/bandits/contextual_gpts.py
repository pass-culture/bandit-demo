import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from src.models.bandits.bandit import Bandit


class ContextualGPTS(Bandit):
    """Gaussian Process Thompson Sampling (GP-TS) Bandit.
    For optimization purposes, this implementation uses a dynamic update interval to
    balance exploration and computational efficiency. The memory is capped to a maximum
    size to prevent O(N^3) complexity in GP fitting.
    """

    def __init__(
        self, alpha: float = 1.0, max_memory: int = 2000, boost: float | None = None
    ):
        super().__init__(boost=boost)
        self.alpha = alpha
        self.max_memory = max_memory
        self.is_fitted = False

        # Tracks current position in the bandit's lifespan
        self.total_steps = 0
        self.steps_since_fit = 0

        # Setup exact GP
        kernel = C(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)

        # Structured memory storage
        self.X_buffer = []
        self.y_buffer = []

    def _get_adaptive_k(self) -> int:
        """
        Calculates dynamic update interval.
        - Steps 0-500: Updates every 5 steps (Fast learning)
        - Steps 500-2000: Updates every 50 steps
        - Steps 2000+: Updates every 250 steps (Computation saving)
        """
        if self.total_steps < 500:
            return 5
        elif self.total_steps < 2000:
            return 50
        else:
            return 250

    def _get_joint_features(self, context: np.ndarray, arm_key: str) -> np.ndarray:
        arm_idx = self.arm_to_idx[arm_key]
        arm_one_hot = np.zeros(len(self.all_arms))
        arm_one_hot[arm_idx] = 1.0
        return np.concatenate([context, arm_one_hot])

    def select_arm_idx(
        self, arms_pool: list, context: np.ndarray | list | None = None
    ) -> int:
        if context is None or not self.is_fitted:
            return np.random.choice(len(arms_pool))

        context_vec = np.asarray(context)
        candidate_features = np.array(
            [self._get_joint_features(context_vec, arm) for arm in arms_pool]
        )

        # Version standardisée par bras (Heuristic GP-TS)
        means, stds = self.gp.predict(candidate_features, return_std=True)
        samples = means + self.alpha * stds * np.random.randn(len(arms_pool))
        return int(np.argmax(samples))

    def update_by_idx(
        self, arm_idx: int, reward: float, context: np.ndarray | list | None = None
    ):
        super().update_by_idx(arm_idx, reward, context)

        if context is not None:
            self.total_steps += 1
            self.steps_since_fit += 1

            # Map features and append to tracking arrays
            joint_feat = self._get_joint_features(
                np.asarray(context), self.all_arms[arm_idx]
            )
            self.X_buffer.append(joint_feat)
            self.y_buffer.append(reward)

            # Memory Capping: Enforce maximum matrix bounds to prevent O(N^3) explosion
            if (
                self.steps_since_fit >= self._get_adaptive_k()
                and len(self.X_buffer) > self.max_memory
            ):
                # Keep the last 80% (recency) and top 20% (high-value milestones)
                cutoff_idx = len(self.X_buffer) - int(self.max_memory * 0.8)
                recent_X = self.X_buffer[cutoff_idx:]
                recent_y = self.y_buffer[cutoff_idx:]

                # Sort remaining historical data by highest reward value
                historical_X = self.X_buffer[:cutoff_idx]
                historical_y = self.y_buffer[:cutoff_idx]

                num_core_to_keep = self.max_memory - len(recent_X)
                top_indices = np.argsort(historical_y)[-num_core_to_keep:]

                self.X_buffer = [historical_X[idx] for idx in top_indices] + recent_X
                self.y_buffer = [historical_y[idx] for idx in top_indices] + recent_y

            # Check dynamic training threshold condition
            if self.steps_since_fit >= self._get_adaptive_k():
                self.gp.fit(np.array(self.X_buffer), np.array(self.y_buffer))
                self.steps_since_fit = 0
                self.is_fitted = True
