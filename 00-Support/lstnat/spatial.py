"""Adaptive KNN weights and esda Lee statistics, derived from Relationship notebooks."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags


def adaptive_weights(coords_km, k=8, kernel="gaussian"):
    coords = np.asarray(coords_km, float)
    n = len(coords)
    if n < 3 or not np.isfinite(coords).all():
        raise ValueError("Spatial analysis requires at least three finite locations")
    k = min(k, n - 1)
    if k < 1:
        raise ValueError("k must be positive")
    distances, neighbors = cKDTree(coords).query(coords, k=k + 1)
    rows, cols, data, bandwidths = [], [], [], []
    for i in range(n):
        mask = neighbors[i] != i
        js, ds = neighbors[i][mask][:k], distances[i][mask][:k]
        bw = max(float(ds.max()), 1e-9)
        bandwidths.append(bw)
        if kernel == "gaussian":
            weights = np.exp(-0.5 * (ds / bw)**2)
        elif kernel == "bisquare":
            weights = np.maximum(1 - (ds / bw)**2, 0)**2
        elif kernel == "inverse_distance":
            weights = 1 / np.maximum(ds, 1e-9)
        else:
            raise ValueError(kernel)
        rows.extend([i] * len(js)); cols.extend(js.tolist()); data.extend(weights.tolist())
    directed = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Missing reverse links count as zero in (W + W.T) / 2.
    symmetric = (directed + directed.T) * 0.5
    symmetric.setdiag(0); symmetric.eliminate_zeros()
    row_sums = np.asarray(symmetric.sum(axis=1)).ravel()
    if np.any(row_sums == 0):
        raise ValueError("Kernel produced an isolated station")
    w = (diags(1 / row_sums) @ symmetric).tocsr()
    return w, {"k": k, "kernel": kernel, "bandwidth_mean_km": float(np.mean(bandwidths))}


def zscore(values):
    a = np.asarray(values, float)
    sd = a.std(ddof=0)
    if sd < 1e-12 or not np.isfinite(a).all():
        raise ValueError("Constant or non-finite spatial variable")
    return (a - a.mean()) / sd


def lee(nat, lst, w, permutations=999, seed=42, local_permutations=None):
    from esda.lee import Spatial_Pearson, Spatial_Pearson_Local
    x, y = np.asarray(nat, float).reshape(-1, 1), np.asarray(lst, float).reshape(-1, 1)
    # esda Lee uses numpy's global RNG; restore caller state after deterministic inference.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        global_model = Spatial_Pearson(w, permutations=permutations).fit(x, y)
        np.random.seed(seed)
        local_n = permutations if local_permutations is None else local_permutations
        local_model = Spatial_Pearson_Local(w, permutations=local_n).fit(x, y)
    finally:
        np.random.set_state(state)
    value = float(global_model.association_[0, 1])
    local = np.asarray(local_model.associations_).ravel()
    p_global = float(global_model.significance_[0, 1]) if permutations else np.nan
    p_local = np.asarray(local_model.significance_) if local_n else np.full(len(x), np.nan)
    # The mean local statistic must equal the global statistic for row-standardized W.
    if not np.isclose(local.mean(), value, atol=1e-8):
        raise ArithmeticError("Global/local Lee decomposition failed")
    return value, p_global, local, p_local


def clusters(nat, lst, w, p_values, alpha=0.05):
    # Both variables are spatially smoothed, as in the source Lee implementation.
    a, b = w @ zscore(lst), w @ zscore(nat)
    labels = np.full(len(a), "Not significant", dtype=object)
    sig = np.isfinite(p_values) & (np.asarray(p_values) < alpha)
    labels[sig] = "Undefined"
    for sa, sb, label in [(1, 1, "High-High"), (-1, -1, "Low-Low"),
                          (1, -1, "High-Low"), (-1, 1, "Low-High")]:
        labels[sig & (a * sa > 0) & (b * sb > 0)] = label
    return labels


def as_pysal(w):
    from libpysal.weights import W
    neighbors, weights = {}, {}
    for i in range(w.shape[0]):
        row = w.getrow(i)
        neighbors[i], weights[i] = row.indices.tolist(), row.data.tolist()
    return W(neighbors, weights, id_order=list(range(w.shape[0])), silence_warnings=True)


def alternative_statistics(nat, lst, w, permutations=999, seed=42):
    """Moran BV and source-defined cross-variable Geary statistics."""
    from esda.moran import Moran_BV, Moran_Local_BV
    from esda.geary_local_mv import Geary_Local_MV
    x, y = zscore(nat), zscore(lst)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        m = Moran_BV(x, y, as_pysal(w), permutations=permutations)
        ml = Moran_Local_BV(x, y, as_pysal(w), permutations=permutations, seed=seed, n_jobs=1)
        np.random.seed(seed)
        gl = Geary_Local_MV(connectivity=as_pysal(w), permutations=permutations).fit(np.vstack([x, y]))
        row, col = w.nonzero()
        weights = np.asarray(w[row, col]).ravel()
        def cross_geary(a, b):
            return float(np.sum(weights * (a[row] - b[col])**2) / weights.sum())
        observed = cross_geary(x, y)
        rng = np.random.default_rng(seed)
        sims = []
        for _ in range(permutations):
            ids = rng.permutation(len(x))
            sims.append(cross_geary(x[ids], y[ids]))
        larger = int(np.sum(np.asarray(sims) >= observed))
        gp = (min(larger, permutations - larger) + 1) / (permutations + 1) if permutations else np.nan
    finally:
        np.random.set_state(state)
    return {
        "moran": {"value": float(m.I), "p": float(m.p_sim) if permutations else np.nan,
                  "local": ml.Is, "local_p": ml.p_sim if permutations else np.full(len(x), np.nan)},
        "geary": {"value": observed, "p": gp, "local": gl.localG,
                  "local_p": gl.p_sim if permutations else np.full(len(x), np.nan)},
    }
