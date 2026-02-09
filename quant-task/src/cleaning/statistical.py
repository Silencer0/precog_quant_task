import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def impute_mle(df, col="Close"):
    """
    Maximum Likelihood Estimation for parameter estimation and imputation (Causal).
    Assumes data follows a normal distribution with causal expanding mean/std.
    """
    series = df[col].copy()
    # Causal parameter estimation using expanding window
    exp_mean = series.expanding().mean()
    exp_std = series.expanding().std().fillna(0.0)
    
    # Fill each NaN with the most recent causal mean
    series = series.fillna(exp_mean.ffill())
    # If starting values are NaN, use the first non-NaN mean found
    series = series.fillna(exp_mean.bfill())
    
    return pd.DataFrame({col: series}, index=df.index)


def bayesian_smoothing(series, prior_mu=None, prior_sigma=None, noise_sigma=0.1):
    """
    Simple Bayesian smoothing (Causal).
    Updates local estimates using a causal expanding prior if none provided.
    """
    s_orig = series.ffill().bfill()
    if prior_mu is None:
        prior_mu = s_orig.expanding().mean()
    else:
        prior_mu = pd.Series(prior_mu, index=series.index)
        
    if prior_sigma is None:
        prior_sigma = s_orig.expanding().std().replace(0.0, 1.0).fillna(1.0)
    else:
        prior_sigma = pd.Series(prior_sigma, index=series.index)
        
    s = s_orig.values
    p_mu = prior_mu.values
    p_sig = prior_sigma.values
    
    w_prior = 1.0 / (p_sig**2)
    w_noise = 1.0 / (noise_sigma**2)
    
    posterior_mu = (p_mu * w_prior + s * w_noise) / (w_prior + w_noise)
    return pd.Series(posterior_mu, index=series.index)


def markov_smoothing(series, n_states=5):
    """
    Discrete Markov Model smoothing (Causal).
    Estimates transitions using an expanding window to maintain causality.
    """
    s_orig = series.ffill().bfill()
    s = s_orig.values
    
    n = len(s)
    smoothed = np.zeros(n)
    trans_counts = np.zeros((n_states, n_states))
    
    # We need a way to map values to states causally. 
    # We use the expanding range to define bins at each step.
    for i in range(n):
        # Update bins causally
        curr_min = s_orig.iloc[:i+1].min()
        curr_max = s_orig.iloc[:i+1].max()
        if curr_max == curr_min:
            curr_max += 1e-6
            
        bins = np.linspace(curr_min, curr_max, n_states + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        states = np.digitize(s[:i+1], bins) - 1
        states[states == n_states] = n_states - 1
        
        if i > 0:
            # Update transitions with the observed pair (t-1, t)
            trans_counts[states[i-1], states[i]] += 1
            
        curr_state = states[i]
        row_sum = trans_counts[curr_state].sum()
        if row_sum > 0:
            trans_prob = trans_counts[curr_state] / row_sum
            smoothed[i] = np.dot(trans_prob, bin_centers)
        else:
            smoothed[i] = bin_centers[curr_state]
            
    return pd.Series(smoothed, index=series.index)


def hmm_smoothing(series, n_states=3):
    """
    Simplified Hidden Markov Model (HMM) smoothing (Causal).
    Identifies regimes using only past and current data.
    """
    s_orig = series.ffill().bfill()
    s = s_orig.values
    exp_std = s_orig.expanding().std().fillna(1.0)
    exp_min = s_orig.expanding().min()
    exp_max = s_orig.expanding().max()
    
    A = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(A, 0.9)
    
    prob = np.full(n_states, 1.0 / n_states)
    smoothed_vals = []
    
    for i in range(len(s)):
        val = s[i]
        # Parameters derived only from data seen up to i
        s_min = exp_min.iloc[i]
        s_max = exp_max.iloc[i]
        if s_max == s_min:
            s_max += 1e-6
        means = np.linspace(s_min, s_max, n_states)
        std = exp_std.iloc[i] / n_states
        
        # Emission probability
        e = norm.pdf(val, means, std + 1e-6)
        prob = (prob @ A) * e
        p_sum = prob.sum()
        if p_sum > 0:
            prob /= p_sum
        else:
            prob = np.full(n_states, 1.0 / n_states)
            
        smoothed_vals.append(means[np.argmax(prob)])
        
    return pd.Series(smoothed_vals, index=series.index)


def em_imputation(df, col="Close", n_iter=10):
    """
    Expectation-Maximization (EM) for imputation (Causal).
    Iteratively estimates missing values using causal trailing windows.
    """
    series = df[col].copy()
    s = series.values.copy()
    mask = np.isnan(s)
    
    if not mask.any():
        return df
        
    # Initial seed with ffill
    s_series = pd.Series(s).ffill().bfill()
    
    for _ in range(n_iter):
        # Use trailing rolling mean as the "expected" value for hidden states (NaNs)
        s_series = s_series.rolling(window=10, min_periods=1, center=False).mean()
        # Restore original non-NaN values
        s_series.iloc[~mask] = s[~mask]
        s_series = s_series.ffill().bfill()
        
    series.update(s_series)
    return pd.DataFrame({col: series}, index=df.index)
