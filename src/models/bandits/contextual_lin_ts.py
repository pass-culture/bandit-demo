import numpy as np
import scipy

from src.models.bandits.bandit import Bandit


class ContextualLinTS(Bandit):
    def __init__(
        self,
        d: int,
        v2: float = 0.3,
        lambda_reg: float = 1.0,
        boost: float | None = None,
    ):
        super().__init__(boost=boost)
        self.d = d
        self.v2 = v2
        self.lambda_reg = lambda_reg

        self.B = None  # Track actual design matrix (K, d, d)
        self.L_B = None  # Cache for Cholesky factors of B (K, d, d)
        self.f = None  # Track reward-weighted contexts (K, d, 1)
        self.theta = None  # Cached posterior means (K, d, 1)

    def _init_arms(self, arms_pool: list) -> None:
        if not self.initialized:
            super()._init_arms(arms_pool)
            n_arms = len(arms_pool)

            # Initialize B to lambda_reg * Identity matrix for numerical stability
            self.B = np.array(
                [np.identity(self.d) * self.lambda_reg for _ in range(n_arms)]
            )
            self.L_B = np.array(
                [np.identity(self.d) * np.sqrt(self.lambda_reg) for _ in range(n_arms)]
            )
            self.f = np.zeros((n_arms, self.d, 1))
            self.theta = np.zeros((n_arms, self.d, 1))

    def select_arm_idx(
        self,
        arms_pool: list,
        context: np.ndarray | list | None = None,
    ) -> int:
        if context is None:
            raise ValueError("Linear Thompson Sampling requires a context vector.")

        self._init_arms(arms_pool)

        # Safe copy & clear dimensional definition (d, 1)
        x = np.array(context, dtype=float).reshape(-1, 1)
        if x.shape[0] != self.d:
            raise ValueError(
                f"Context dimension {x.shape[0]} does not match model dimension {self.d}"  # noqa: E501
            )

        n_arms = len(arms_pool)
        predicted_rewards = np.zeros(n_arms)
        scale = np.sqrt(self.v2)

        for i, arm_key in enumerate(arms_pool):
            # FIX: Resolve the global tracking index for this specific candidate arm
            global_idx = self.arm_to_idx[arm_key]

            standard_noise = np.random.normal(0, 1, size=(self.d, 1))

            # FIX: Solve with L^T and up triangular formulation to get B^-1 covariance
            sampled_noise_transformed = scipy.linalg.solve_triangular(
                self.L_B[global_idx].T, standard_noise, lower=False
            )

            sampled_theta = self.theta[global_idx] + scale * sampled_noise_transformed

            predicted_rewards[i] = np.dot(sampled_theta.flatten(), x.flatten())

        return int(np.argmax(predicted_rewards))

    def update_by_idx(
        self,
        arm_idx: int,
        reward: float,
        context: np.ndarray | list | None = None,
    ) -> None:
        if context is None:
            raise ValueError(
                "Linear Thompson Sampling requires a context vector for updates."
            )

        x = np.array(context, dtype=float).reshape(-1, 1)
        if x.shape[0] != self.d:
            raise ValueError(
                f"Context dimension {x.shape[0]} does not match model dimension {self.d}"  # noqa: E501
            )

        # Update core counts and arm total track records in parent class
        super().update_by_idx(arm_idx, reward, context)

        # Vectorized Outer product update: B_t = B_{t-1} + x*x^T
        self.B[arm_idx] += np.outer(x, x)

        # Update cumulative reward-weighted contexts: f_t = f_{t-1} + reward * x
        self.f[arm_idx] += reward * x

        try:
            self.L_B[arm_idx] = np.linalg.cholesky(self.B[arm_idx])
        except np.linalg.LinAlgError:
            B_reg = self.B[arm_idx] + 1e-6 * np.eye(self.d)
            self.L_B[arm_idx] = np.linalg.cholesky(B_reg)

        self.theta[arm_idx] = scipy.linalg.cho_solve(
            (self.L_B[arm_idx], True), self.f[arm_idx]
        )
