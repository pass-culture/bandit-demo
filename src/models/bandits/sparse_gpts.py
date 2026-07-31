import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.models.bandits.bandit import Bandit


class SparseGPTS(Bandit):
    """Sparse Gaussian Process Thompson Sampling Bandit.

    Uses an Inducing Point framework (Subset of Regressors approximation) to
    project the continuous data stream onto a fixed set of M anchor points.

    This implementation utilizes a Reward-Weighted Pivoted Cholesky (RWPC)
    strategy to locate inducing points.
    """

    def __init__(
        self,
        context_dim: int,
        num_inducing: int = 200,
        alpha: float = 1.2,
        length_scale: float = 2.5,
        signal_variance: float = 0.05,
        noise_variance: float = 0.005,
        boost: float | None = None,
        jitter: float = 1e-5,
        warmup_multiplier: int = 4,  # Collect N = warmup_multiplier * M points before selecting initial Z  # noqa: E501
        update_interval: int = 500,  # Periodically optimize inducing points after every 'update_interval' steps  # noqa: E501
        rw_tau: float = 0.5,  # Temperature parameter for reward weighting (lower = focus more on peaks)  # noqa: E501
        rw_epsilon: float = 0.3,  # Diversity anchor parameter (ensures low-reward zones get some representation)  # noqa: E501
    ):
        super().__init__(boost=boost)

        self.context_dim = context_dim
        self.M = num_inducing
        self.alpha = alpha
        self.length_scale = length_scale
        self.sigma_f2 = signal_variance
        self.sigma_n2 = noise_variance
        self.jitter = jitter  # Added for Cholesky stability

        # RWPC Configuration parameters
        self.warmup_multiplier = warmup_multiplier
        self.N_warmup = warmup_multiplier * self.M
        self.update_interval = update_interval
        self.rw_tau = rw_tau
        self.rw_epsilon = rw_epsilon

        # Structural dimensions (will bind safely inside lazy init)
        self.num_arms = 0
        self.d = 0

        # Hold anchor inducing points
        self.inducing_points = None
        self.inducing_initialized = False
        self.num_collected = 0
        self.steps_since_reopt = 0

        # Memory buffer for candidate pools
        self._historical_features = []
        self._historical_rewards = []

        # Precision structures for Sparse Space
        self.A = np.identity(self.M) * self.sigma_n2
        self.b = np.zeros((self.M, 1))

        # Pre-computed matrix caches
        self.K_uu = None
        self.A_cho = None
        self.beta = np.zeros((self.M, 1))
        self.K_uu_cho = None
        self.needs_factorization = True

    def _lazy_initialize_dimensions(self):
        """Late-binding initialization to catch all_arms parameters populated by parent setup."""  # noqa: E501
        if self.num_arms == 0 and len(self.all_arms) > 0:
            self.num_arms = len(self.all_arms)
            self.d = self.context_dim + self.num_arms
            # Inducing points will be set dynamically via Pivoted Cholesky

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Vectorized Radial Basis Function (RBF) Kernel calculation."""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        sq_dist = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return self.sigma_f2 * np.exp(-0.5 * sq_dist / (self.length_scale**2))

    def _select_inducing_points_rwpc(self):
        """Selects M inducing points from the accumulated historical buffer
        using Reward-Weighted Pivoted Cholesky.
        """
        X_candidates = np.vstack(self._historical_features)
        y_rewards = np.array(self._historical_rewards)
        N = len(X_candidates)

        # 1. Compute Quality weights from rewards
        max_y = np.max(y_rewards)
        weights = np.exp((y_rewards - max_y) / self.rw_tau) + self.rw_epsilon

        # 2. Extract Diagonal of the weighted covariance matrix (K(x,x) = sigma_f^2)
        # We model the weighted diagonal residuals to select pivots.
        diag = (weights**2) * self.sigma_f2

        selected_indices = []

        # Store intermediate vectors to recursively construct Cholesky factors
        # L matrix keeps track of our selected pivot interactions: shape (M, N)
        L = np.zeros((self.M, N))

        for step in range(self.M):
            # Select candidate with maximum remaining weighted variance
            pivot = np.argmax(diag)
            selected_indices.append(pivot)

            # Extract the actual raw kernel row of the selected pivot
            pivot_x = X_candidates[pivot : pivot + 1]
            k_row = self._rbf_kernel(X_candidates, pivot_x).flatten()  # (N,)

            # Apply quality scaling to the kernel row
            weighted_k_row = weights[pivot] * k_row * weights

            # Compute Cholesky update terms (Schur Complement)
            # This steps through projection-space to deflate the remaining diagonal
            if step > 0:
                weighted_k_row -= np.dot(L[:step, pivot], L[:step, :])

            pivot_val = np.sqrt(np.maximum(1e-12, diag[pivot]))
            L[step, :] = weighted_k_row / pivot_val

            # Deflate remaining diagonal variances
            diag -= L[step, :] ** 2
            diag[selected_indices] = -np.inf  # Prevent re-selection of pivots

        # Extract selected anchor contexts
        self.inducing_points = X_candidates[selected_indices]
        self.inducing_initialized = True

        # Re-build precision matrices A and b from scratch using the new inducing points
        self._rebuild_precision_matrices(X_candidates, y_rewards)

    def _rebuild_precision_matrices(self, X: np.ndarray, y: np.ndarray):
        """Regenerates global precision metrics over historical data when Z updates."""
        self.A = np.identity(self.M) * self.sigma_n2
        self.b = np.zeros((self.M, 1))

        # Re-apply stream updates across all collected history in one vectorized pass
        K_ux = self._rbf_kernel(self.inducing_points, X)  # Shape: (M, N)

        # Vectorized updates: A += (K_ux @ K_ux.T) / sigma_n2
        self.A += np.dot(K_ux, K_ux.T) / self.sigma_n2
        # Vectorized updates: b += (K_ux @ y) / sigma_n2
        self.b += np.dot(K_ux, y.reshape(-1, 1)) / self.sigma_n2

        self.needs_factorization = True

    def select_arm_idx(
        self, arms_pool: list, context: np.ndarray | list | None = None
    ) -> int:
        self._lazy_initialize_dimensions()

        # Uniform fallback before warmup phase yields initialized inducing points
        if context is None or not self.inducing_initialized:
            return int(np.random.choice(len(arms_pool)))

        context_vec = np.asarray(context).flatten()

        if self.needs_factorization:
            self.K_uu = self._rbf_kernel(self.inducing_points, self.inducing_points)
            self.K_uu += np.eye(self.M) * (self.jitter * self.sigma_f2)

            try:
                self.K_uu_cho = cho_factor(self.K_uu, lower=True)
                self.A_cho = cho_factor(self.K_uu + self.A, lower=True)
                self.beta = cho_solve(self.A_cho, self.b)
                self.needs_factorization = False
            except np.linalg.LinAlgError:
                # Emergency fallback if matrix is singular
                return int(np.random.choice(len(arms_pool)))

        # Vectorized Generation of Candidate Joint Features
        num_candidates = len(arms_pool)
        candidate_features = np.zeros((num_candidates, self.d))
        candidate_features[:, : self.context_dim] = context_vec

        arm_indices = np.array([self.arm_to_idx[arm] for arm in arms_pool])
        candidate_features[
            np.arange(num_candidates), self.context_dim + arm_indices
        ] = 1.0

        K_xu = self._rbf_kernel(candidate_features, self.inducing_points)
        means = np.dot(K_xu, self.beta).flatten()

        v_prior = cho_solve(self.K_uu_cho, K_xu.T)
        v_post = cho_solve(self.A_cho, K_xu.T)

        prior_cov_sparse = np.sum(K_xu.T * v_prior, axis=0)
        dropped_uncertainty = np.sum(K_xu.T * (v_prior - v_post), axis=0)

        stds = np.sqrt(np.maximum(1e-6, prior_cov_sparse - dropped_uncertainty))

        samples = means + self.alpha * stds * np.random.randn(num_candidates)

        return int(np.argmax(samples))

    def update_by_idx(
        self, arm_idx: int, reward: float, context: np.ndarray | list | None = None
    ):
        super().update_by_idx(arm_idx, reward, context)
        self._lazy_initialize_dimensions()

        if context is not None:
            context_vec = np.asarray(context).flatten()
            arm_one_hot = np.zeros(self.num_arms)
            arm_one_hot[arm_idx] = 1.0
            x_t = np.concatenate([context_vec, arm_one_hot]).reshape(1, -1)

            # Add sample to our rolling buffers
            self._historical_features.append(x_t)
            self._historical_rewards.append(reward)
            self.num_collected += 1

            if not self.inducing_initialized:
                # Keep accumulating until we reach our target warmup pool size N
                if self.num_collected >= self.N_warmup:
                    self._select_inducing_points_rwpc()
            else:
                # Apply fast rank-1 continuous updates during runtime
                self._apply_precision_update(x_t, reward)
                self.steps_since_reopt += 1

                # Periodically optimize anchor locations to align with streaming shifts
                if self.steps_since_reopt >= self.update_interval:
                    self._select_inducing_points_rwpc()
                    self.steps_since_reopt = 0

    def _apply_precision_update(self, x_t: np.ndarray, reward: float):
        """Applies rank-1 sparse updating transformations to the system matrices."""
        K_ui = self._rbf_kernel(self.inducing_points, x_t)
        self.A += np.dot(K_ui, K_ui.T) / self.sigma_n2
        self.b += (reward * K_ui) / self.sigma_n2
        self.needs_factorization = True
