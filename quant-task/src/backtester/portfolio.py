import numpy as np
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def equal_weight(candidates: list) -> dict:
    if not candidates:
        return {}
    n = len(candidates)
    return {a: 1.0 / n for a in candidates}


def optimize_mpt(
    returns_matrix: pd.DataFrame, candidates: list, current_date, lookback_days=126
) -> dict:
    if not candidates:
        return {}

    start_date = current_date - pd.Timedelta(days=lookback_days)
    history = (
        returns_matrix.loc[start_date:current_date, list(candidates)]
        .dropna(how="all")
        .fillna(0)
    )

    if len(history) < 20:
        return equal_weight(candidates)

    mean_returns = history.mean() * 252
    cov_matrix = history.cov() * 252

    def objective(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - 0.02) / (port_vol + 1e-9)

    n = len(candidates)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0, 0.4) for _ in range(n)]
    initial_guess = [1.0 / n] * n

    result = minimize(
        objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if result.success:
        return dict(zip(candidates, result.x))
    return equal_weight(candidates)


def optimize_nmpt_hrp(
    returns_matrix: pd.DataFrame, candidates: list, current_date, lookback_days=126
) -> dict:
    if not candidates or len(candidates) < 2:
        return equal_weight(candidates)

    start_date = current_date - pd.Timedelta(days=lookback_days)
    history = (
        returns_matrix.loc[start_date:current_date, list(candidates)]
        .dropna(how="all")
        .fillna(0)
    )

    if len(history) < 20:
        return equal_weight(candidates)

    # HRP implementation
    def get_ivp(cov):
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def get_cluster_var(cov, c_items):
        cov_ = cov.loc[c_items, c_items]
        w_ = get_ivp(cov_)
        c_var = np.dot(np.dot(w_, cov_), w_)
        return c_var

    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
        return sort_ix.tolist()

    def get_rec_bisection(cov, sort_ix):
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                v0 = get_cluster_var(cov, c_items0)
                v1 = get_cluster_var(cov, c_items1)
                alpha = 1 - v0 / (v0 + v1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    corr = history.corr()
    cov = history.cov()
    dist = np.sqrt((1 - corr).clip(lower=0) / 2.0)
    link = linkage(squareform(dist), "single")
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    hrp_weights = get_rec_bisection(cov, sort_ix)

    return hrp_weights.to_dict()
