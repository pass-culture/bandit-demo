import numpy as np

from src.models.bandits.bandit import Bandit


class ContextualLinUCB(Bandit):
    """
    Contextual LinUCB bandit, arms defined by (category, sub category, venue type) combinations.
    """  # noqa: E501

    def __init__(self, alpha: float = 0.1, boost: float | None = None):
        super().__init__(boost=boost)
        self.alpha = alpha
        self.A_inv = None  # (K, d, d)
        self.b = None  # (K, d, 1)
        self.theta = None  # (K, d, 1) - Cached weights
        self.dim = None
        self.model_logs = []

    def _init_linucb(self, n_arms: int, dim: int) -> None:
        if self.A_inv is None:
            self.dim = dim
            self.A_inv = np.zeros((n_arms, dim, dim))
            for i in range(n_arms):
                self.A_inv[i] = np.eye(dim)
            self.b = np.zeros((n_arms, dim, 1))
            self.theta = np.zeros((n_arms, dim, 1))

    def select_arm_idx(
        self, arms_pool: list, context: np.ndarray | list | None = None
    ) -> int:
        if not self.initialized:
            self._init_arms(arms_pool)

        n_global_arms = len(self.all_arms)
        dim = len(context)
        self._init_linucb(n_global_arms, dim)

        x = context.reshape(-1, 1)  # (d, 1)

        if len(arms_pool) == n_global_arms:
            theta_subset = self.theta
            A_inv_subset = self.A_inv
        else:
            global_indices = [self.arm_to_idx[arm] for arm in arms_pool]
            theta_subset = self.theta[global_indices]  # (C, d, 1)
            A_inv_subset = self.A_inv[global_indices]  # (C, d, d)
        p_mean = np.matmul(theta_subset.transpose(0, 2, 1), x).squeeze()

        tmp = np.matmul(x.T, A_inv_subset)  # (C, 1, d)
        p_var = np.matmul(tmp, x).squeeze()  # (C,)

        p_ucb = p_mean + self.alpha * np.sqrt(p_var)

        self.model_logs.append({"p_mean": p_mean, "p_var": p_var, "p_ucb": p_ucb})

        # Returns index relative to the provided arms_pool array
        return int(np.argmax(p_ucb))

    def update_by_idx(
        self, arm_idx: int, reward: float, context: np.ndarray | list | None = None
    ) -> None:
        x = context.reshape(-1, 1)
        inv = self.A_inv[arm_idx]

        # Update core counts and arm total track records in parent class
        super().update_by_idx(arm_idx, reward, context)

        # Sherman-Morrison Update for A_inv
        A_inv_x = inv @ x
        denom = 1.0 + (x.T @ A_inv_x)
        self.A_inv[arm_idx] -= (A_inv_x @ A_inv_x.T) / denom

        self.b[arm_idx] += reward * x

        # Update ONLY the theta for this arm: theta = A_inv @ b
        self.theta[arm_idx] = self.A_inv[arm_idx] @ self.b[arm_idx]
