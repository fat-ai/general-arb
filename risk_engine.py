import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

def shrinkage(returns: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Computes the Ledoit-Wolf optimal shrinkage for the covariance matrix.
    
    This balances the high variance of the sample covariance (noisy) with the 
    high bias of the constant correlation estimator (rigid).
    
    Ref: Ledoit & Wolf, "Honey, I shrunk the sample covariance matrix" (2004)
    """
    t, n = returns.shape
    mean_returns = np.mean(returns, axis=0, keepdims=True)
    centered_returns = returns - mean_returns
    sample_cov = centered_returns.T @ centered_returns / t

    # --- 1. Compute the Target: Constant Correlation Matrix ---
    var = np.diag(sample_cov).reshape(-1, 1)
    sqrt_var = np.sqrt(var)
    unit_cor_var = sqrt_var @ sqrt_var.T
    
    # Average correlation
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var
    np.fill_diagonal(prior, var.flatten())

    # --- 2. Compute Shrinkage Intensity (Kappa) ---
    # Pi-hat (Asymptotic variance of sample covariance)
    y = centered_returns ** 2
    phi_mat = (y.T @ y) / t - sample_cov ** 2
    phi = phi_mat.sum()

    # Rho-hat (Asymptotic covariance of sample covariance and prior)
    theta_mat = ((centered_returns ** 3).T @ centered_returns) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0)
    rho = (
        np.diag(phi_mat).sum() + 
        average_cor * (1 / sqrt_var @ sqrt_var.T * theta_mat).sum()
    )

    # Gamma-hat (Distance between sample cov and prior)
    gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

    # Calculate optimal shrinkage constant
    kappa = (phi - rho) / gamma
    shrink = max(0, min(1, kappa / t))

    # --- 3. Shrink ---
    sigma = shrink * prior + (1 - shrink) * sample_cov
    
    return sigma, average_cor, shrink

class KellyOptimizer:
    """
    Robust Multi-Asset Kelly Optimizer.
    Uses Quadratic Programming (SLSQP) + Ledoit-Wolf Shrinkage.
    """
    
    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.0):
        self.returns = returns_df
        self.r_f = risk_free_rate
        
        # 1. Expected Returns (Simple Mean)
        # In production, blend this with shorter-term signals if available.
        self.mu = self.returns.mean()
        
        # 2. Risk Model (Shrunk Covariance)
        # This is the key stability fix. It pulls extreme correlations 
        # back toward the average, preventing the optimizer from betting 
        # huge on "fake" arbitrage opportunities.
        shrunk_values, _, _ = shrinkage(self.returns.values)
        self.cov = pd.DataFrame(
            shrunk_values, 
            index=self.returns.columns, 
            columns=self.returns.columns
        )

    def get_optimal_weights(self, fraction=0.5, max_leverage=1.0, long_only=True) -> pd.Series:
        """
        Calculates optimal portfolio weights.
        """
        num_assets = len(self.mu)
        w0 = np.ones(num_assets) / num_assets # Initial guess
        
        # Objective: Maximize Growth Rate (g = r - 0.5*var)
        def negative_growth_rate(w):
            port_ret = np.dot(w, self.mu)
            port_var = np.dot(w.T, np.dot(self.cov, w))
            return -(port_ret - 0.5 * port_var)
        
        # Constraint: Leverage Limit
        constraints = [{'type': 'ineq', 'fun': lambda w: max_leverage - np.sum(np.abs(w))}]
        
        # Bounds: Long Only or Long/Short
        bounds = [(0.0, max_leverage) for _ in range(num_assets)] if long_only else \
                 [(-max_leverage, max_leverage) for _ in range(num_assets)]

        result = minimize(
            negative_growth_rate, 
            w0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            tol=1e-6
        )
        
        if not result.success:
            return pd.Series(0, index=self.mu.index)

        # Scale by fractional Kelly for safety
        return pd.Series(result.x * fraction, index=self.mu.index)
