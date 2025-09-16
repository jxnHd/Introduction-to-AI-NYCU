import numpy as np

def _pairwise_sq_dists(X, C):
    X_norm = np.sum(X*X, axis=1, keepdims=True)
    C_norm = np.sum(C*C, axis=1, keepdims=True).T
    return X_norm + C_norm - 2.0 * X @ C.T

def _kmeans_pp_init(X, k, rng):
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    D2 = np.sum((X - centers[0])**2, axis=1)
    for i in range(1, k):
        probs = D2 / np.sum(D2)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        d2_new = np.sum((X - centers[i])**2, axis=1)
        D2 = np.minimum(D2, d2_new)
    return centers

def _update_centers(X, labels, k):
    d = X.shape[1]
    C = np.zeros((k, d), dtype=X.dtype)
    for j in range(k):
        mask = (labels == j)
        if np.any(mask):
            C[j] = X[mask].mean(axis=0)
        else:
            C[j] = np.nan
    return C

def _fix_empty_clusters(X, C, labels, rng):
    nan_rows = np.isnan(C).any(axis=1)
    if not np.any(nan_rows):
        return C
    dists = _pairwise_sq_dists(X, C[~nan_rows]) if (~nan_rows).any() else np.full((X.shape[0], 1), np.inf)
    nearest = np.min(dists, axis=1)
    order = np.argsort(-nearest)
    idx_iter = iter(order)
    for j in np.where(nan_rows)[0]:
        idx = next(idx_iter, None)
        if idx is None:
            idx = rng.integers(0, X.shape[0])
        C[j] = X[idx]
    return C

def kmeans_numpy(X, k=12, n_init=6, max_iters=100, tol=1e-4, random_state=12):
    rng_master = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = None

    for trial in range(n_init):
        rng = np.random.default_rng(rng_master.integers(1<<31))
        C = _kmeans_pp_init(X, k, rng)
        prev_C = C.copy()

        for it in range(max_iters):
            d2 = _pairwise_sq_dists(X, C)
            labels = np.argmin(d2, axis=1)

            C = _update_centers(X, labels, k)
            C = _fix_empty_clusters(X, C, labels, rng)

            shift = np.linalg.norm(C - prev_C)
            if shift <= tol:
                break
            prev_C = C.copy()

        d2 = _pairwise_sq_dists(X, C)
        inertia = np.sum(d2[np.arange(X.shape[0]), labels])

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels.astype(np.int32)

def clustering(X):
    return kmeans_numpy(X, k=10, n_init=10, max_iters=100, tol=1e-4, random_state= 12)

if __name__ == "__main__":
    X = np.load("./features.npy")
    y = clustering(X)
    np.save("111511281.npy", y)
    # print("Clustering completed successfully!")
