"""Measurements from the paper, applied to company observations."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def whitener(X, k=None, eps=1e-6):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = k or X.shape[1]
    W = Vt[:k].T / (S[:k] / np.sqrt(len(X)) + eps)
    return lambda Y: (Y - mu) @ W


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def probe(Xtr, ytr, Xte, yte, C=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())


def knn_precision(X, label, group, thread, queries, k=10, other_group_only=True, chunk=2048):
    """Precision@k of same-`label` neighbours for each query, excluding its own thread; optionally only among
    candidates from a different `group` (system): that is what system-invariance means."""
    Xn = l2n(X)
    hits = []
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk]
        sim = Xn[q] @ Xn.T
        sim[thread[q][:, None] == thread[None, :]] = -np.inf
        if other_group_only:
            sim[group[q][:, None] == group[None, :]] = -np.inf
        nn_ = np.argpartition(-sim, k, axis=1)[:, :k]
        hits.append((label[nn_] == label[q][:, None]).mean(1))
    return float(np.concatenate(hits).mean())


def explained(X, groups, k=32, min_count=5):
    """Fraction of a whitened top-k embedding explained by each grouping (between-group covariance, with the
    finite-sample within-group bias removed), reported two ways:
      r2  = tr(B) / k                (variance fraction)
      rho = tr(B^{1/2}) / k          (the paper's scene-explained fraction, Thm. 3)"""
    Z = whitener(X, k)(X)
    out = {}
    for name, g in groups.items():
        ok = g >= 0
        Zg, gg = Z[ok], g[ok]
        B = np.zeros((k, k))
        n = len(Zg)
        for u in np.unique(gg):
            m = gg == u
            nu = m.sum()
            if nu < min_count:
                continue
            mu = Zg[m].mean(0)
            Wu = np.cov(Zg[m].T) if nu > 1 else np.zeros((k, k))
            B += (nu / n) * (np.outer(mu, mu) - Wu / nu)
        ev = np.clip(np.linalg.eigvalsh((B + B.T) / 2), 0, None)
        out[name] = dict(r2=float(ev.sum() / k), rho=float(np.sqrt(ev).sum() / k))
    return out


def procrustes(A, B):
    """R in O(d) minimising ||A - B R||_F (both rows paired)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def mutual_knn(A, B, k=10):
    def nn(X):
        Xn = l2n(X)
        s = Xn @ Xn.T
        np.fill_diagonal(s, -np.inf)
        return np.argpartition(-s, k, axis=1)[:, :k]
    na, nb = nn(A), nn(B)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(na, nb)]))


def paired_retrieval(A, B, la, lb, ks=(1, 10)):
    """For each row of A, rank rows of B by cosine; hit@k if a top-k row of B shares A's label."""
    s = l2n(A) @ l2n(B).T
    order = np.argsort(-s, axis=1)
    return {f"R@{k}": float((lb[order[:, :k]] == la[:, None]).any(1).mean()) for k in ks}
