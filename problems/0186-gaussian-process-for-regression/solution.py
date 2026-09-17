import math  # ---------------------------------------- utf-8 encoding ---------------------------------

# This file contains Gaussian Process implementation.
import numpy as np
import math
import scipy
from scipy.special import gamma, kv


def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5, sigma=1.0):
    r = np.linalg.norm(x - x_prime)
    if r == 0.0:
        return 1.0
    z = np.sqrt(2 * nu) * r / length_scale
    return sigma ** 2 * (
        2 ** (1 - nu)
        / gamma(nu)
        * z ** nu
        * kv(nu, z)
    )


def rbf_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0):
    squared_distance = np.sum((x - x_prime) ** 2)
    return sigma ** 2 * np.exp(-squared_distance / (2 * length_scale ** 2))


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    squared_distance = np.sum((x - x_prime) ** 2)
    return sigma ** 2 * np.exp(-2.0 / length_scale ** 2 * np.sin(np.pi * squared_distance / period) ** 2)


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    return sigma_b ** 2 + sigma_v ** 2 * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0
):
    squared_distance = np.sum((x - x_prime) ** 2)
    return sigma ** 2 * (1 + squared_distance / (2 * alpha * length_scale ** 2)) ** (-alpha)


# --- BASE CLASS -------------------------------------------------------------


class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        if noise < 0.0:
            raise ValueError("noise must be non-negative.")
        
        self.kernel = kernel
        self.noise = noise
        self.kernel_params = {} if kernel_params is None else kernel_params.copy()

    def _select_kernel(self, x1, x2):
        '''Selects and computes the kernel value for two single data points.'''
        kernels = {
            "rbf": rbf_kernel,
            "matern": matern_kernel,
            "periodic": periodic_kernel,
            "linear": linear_kernel,
            "rational_quadratic": rational_quadratic_kernel,
        }
        if self.kernel not in kernels:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        return kernels[self.kernel](x1, x2, **self.kernel_params)

    def _compute_covariance(self, X1, X2):
        '''
        Computes the covariance matrix between two sets of points.
        This method fixes the vectorization bug from the original code.
        '''
        return np.array([
            [self._select_kernel(x1, x2) for x2 in X2]
            for x1 in X1
        ])


# --- REGRESSION MODEL -------------------------------------------------------
class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        num_samples, _ = X.shape
        C = self._compute_covariance(X, X) + self.noise * np.eye(num_samples)
        self.L_ = np.linalg.cholesky(C)
        z = scipy.linalg.solve_triangular(self.L_, y, lower=True)
        self.alpha_ = scipy.linalg.solve_triangular(self.L_.T, z, lower=False)
        return self

    def predict(self, X_test, return_std=False):
        K_test_train = self._compute_covariance(X_test, self.X_train_)
        posterior_mean = K_test_train @ self.alpha_
        if not return_std:
            return posterior_mean

        K_test_test = self._compute_covariance(X_test, X_test)
        K_train_test = K_test_train.T
        V = scipy.linalg.solve_triangular(self.L_, K_train_test, lower=True)
        posterior_covariance = K_test_test - V.T @ V
        return posterior_mean, np.sqrt(np.maximum(np.diag(posterior_covariance), 0.0))

    def log_marginal_likelihood(self):
        num_samples = self.y_train_.shape[0]
        data_fit = -0.5 * self.y_train_ @ self.alpha_
        log_det = -np.sum(np.log(np.diag(self.L_)))
        normalization = -0.5 * num_samples * np.log(2.0 * np.pi)
        return float(data_fit + log_det + normalization)

    def optimize_hyperparameters(
        self,
        bounds=((1e-3, 1e3), (1e-3, 1e3)),
        maxiter=100,
    ):
        if self.kernel != "rbf":
            raise NotImplementedError(
                "Hyperparameter optimization currently supports only RBF."
            )

        if not hasattr(self, "X_train_") or not hasattr(self, "y_train_"):
            raise RuntimeError("Call fit() before optimizing hyperparameters.")

        X = self.X_train_
        y = self.y_train_
        n = len(y)

        # Precompute pairwise squared Euclidean distances.
        squared_distances = scipy.spatial.distance.cdist(
            X, X, metric="sqeuclidean"
        )

        identity = np.eye(n)

        # Optimize in log-parameter space to ensure positivity.
        initial_params = np.array([
            self.kernel_params.get("sigma", 1.0),
            self.kernel_params.get("length_scale", 1.0),
        ], dtype=float)

        if np.any(~np.isfinite(initial_params)) or np.any(initial_params <= 0):
            raise ValueError("Initial hyperparameters must be finite and positive.")

        bounds = np.asarray(bounds, dtype=float)

        if (
            bounds.shape != (2, 2)
            or np.any(~np.isfinite(bounds))
            or np.any(bounds <= 0)
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("bounds must contain two valid positive intervals.")

        log_bounds = [
            tuple(np.log(interval))
            for interval in bounds
        ]

        def objective(log_params):
            sigma, length_scale = np.exp(log_params)

            # RBF kernel matrix.
            K = sigma ** 2 * np.exp(
                -squared_distances / (2 * length_scale ** 2)
            )

            # Covariance matrix of noisy observations.
            C = K + self.noise * identity

            # Cholesky decomposition.
            L = np.linalg.cholesky(C)

            # Solve C alpha = y.
            alpha = scipy.linalg.cho_solve(
                (L, True), y
            )

            # Negative log marginal likelihood.
            return float(
                0.5 * (y @ alpha)
                + np.sum(np.log(np.diag(L)))
                + 0.5 * n * np.log(2.0 * np.pi)
            )

        result = scipy.optimize.minimize(
            fun=objective,
            x0=np.log(initial_params),
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"maxiter": maxiter},
        )

        if not result.success:
            raise RuntimeError(
                f"Hyperparameter optimization failed: {result.message}"
            )

        # Convert optimized log-parameters back to original scale.
        optimal_sigma, optimal_length_scale = np.exp(result.x)

        # Update model hyperparameters.
        self.kernel_params["sigma"] = float(optimal_sigma)
        self.kernel_params["length_scale"] = float(optimal_length_scale)

        # Refit the model with the optimized hyperparameters.
        self.fit(X, y)

        return result