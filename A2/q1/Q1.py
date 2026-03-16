"""
COL761 Assignment 2 - Q1: k-means clustering in high dimensions
Usage:
    python3 Q1.py <dataset_num>          # loads from API (IIT network only)
    python3 Q1.py <path_to_dataset>.npy  # loads from .npy file

Optimal k is selected by consensus of three methods:
  1. Kneedle    – perpendicular distance from the WCSS curve diagonal
  2. Gap Stat   – compares WCSS to a uniform-random null reference
  3. Silhouette – average intra/inter-cluster separation (peaks at best k)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════

def load_from_api(dataset_num: int) -> np.ndarray:
    import urllib.request, json
    student_id = "cs5170418"   # <-- replace with your kerberos id
    url = (
        f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id={student_id}&dataset_num={dataset_num}"
    )
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read().decode("utf-8"))
    return np.array(data["X"], dtype=np.float64)


def load_from_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float64)


# ═══════════════════════════════════════════════
# k-means core
# ═══════════════════════════════════════════════

def _init_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = [X[rng.integers(0, n)]]
    for _ in range(k - 1):
        d2 = np.array([min(np.sum((x - c)**2) for c in centers) for x in X])
        centers.append(X[rng.choice(n, p=d2 / d2.sum())])
    return np.array(centers)


def _kmeans_once(X: np.ndarray, k: int, rng: np.random.Generator,
                 max_iter: int = 300) -> tuple[float, np.ndarray]:
    if k == 1:
        mu = X.mean(axis=0)
        return float(np.sum((X - mu)**2)), np.zeros(len(X), dtype=int)

    centers = _init_plusplus(X, k, rng)
    labels  = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        sq_d       = np.sum((X[:, None] - centers[None])**2, axis=2)
        new_labels = np.argmin(sq_d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(axis=0)

    wcss = float(sum(
        np.sum((X[labels == j] - centers[j])**2)
        for j in range(k) if (labels == j).any()
    ))
    return wcss, labels


def best_kmeans(X: np.ndarray, k: int,
                n_restarts: int = 10, seed: int = 42) -> tuple[float, np.ndarray]:
    """Returns (best_wcss, best_labels) over n_restarts runs."""
    rng = np.random.default_rng(seed)
    best_wcss, best_labels = np.inf, None
    for _ in range(n_restarts):
        wcss, labels = _kmeans_once(X, k, rng)
        if wcss < best_wcss:
            best_wcss, best_labels = wcss, labels
    return best_wcss, best_labels


# ═══════════════════════════════════════════════
# Method 1 – Kneedle (perpendicular distance)
# Handles: sharp elbows well
# Weakness: can be fooled when curve is smooth
# ═══════════════════════════════════════════════

def kneedle(objectives: list[float]) -> int:
    objs = np.array(objectives, dtype=float)
    n    = len(objs)
    if n < 3:
        return 1
    yr = objs[0] - objs[-1]
    if yr == 0:
        return 1
    x    = np.linspace(0.0, 1.0, n)
    y    = (objs - objs[-1]) / yr        # y[0]=1, y[-1]=0
    # perpendicular distance from diagonal (0,1)→(1,0)
    perp = np.abs(-x - (y - 1.0)) / np.sqrt(2)
    return int(np.argmax(perp)) + 1      # 1-indexed


# ═══════════════════════════════════════════════
# Method 2 – Gap Statistic  (Tibshirani et al. 2001)
# Handles: smooth curves with no visible elbow
# Weakness: slow (B reference datasets), noisy for small B
# ═══════════════════════════════════════════════

def gap_statistic(X: np.ndarray, objectives: list[float],
                  B: int = 10, seed: int = 0) -> int:
    rng  = np.random.default_rng(seed)
    n, d = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    K    = len(objectives)

    log_wcss_ref = np.zeros((K, B))
    for b in range(B):
        Xref = rng.uniform(size=(n, d)) * (maxs - mins) + mins
        for ki, k in enumerate(range(1, K + 1)):
            wcss_b, _ = best_kmeans(Xref, k, n_restarts=3, seed=seed + b)
            log_wcss_ref[ki, b] = np.log(wcss_b + 1e-10)

    log_wcss_data = np.log(np.array(objectives) + 1e-10)
    gap = log_wcss_ref.mean(axis=1) - log_wcss_data
    sk  = log_wcss_ref.std(axis=1) * np.sqrt(1 + 1.0 / B)

    for ki in range(K - 1):
        if gap[ki] >= gap[ki + 1] - sk[ki + 1]:
            return ki + 1
    return K


# ═══════════════════════════════════════════════
# Method 3 – Silhouette Score
# Handles: both sharp and gradual curves
# Weakness: O(n²) per k, slow on large datasets
# ═══════════════════════════════════════════════

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) <= 1:
        return -1.0
    n = len(X)
    a = np.zeros(n)
    b = np.full(n, np.inf)

    for c in unique:
        mask_c = labels == c
        Xc     = X[mask_c]
        idx_c  = np.where(mask_c)[0]

        if Xc.shape[0] > 1:
            dists    = np.sqrt(np.sum((Xc[:, None] - Xc[None])**2, axis=2))
            a[idx_c] = dists.sum(axis=1) / (Xc.shape[0] - 1)

        for c2 in unique:
            if c2 == c:
                continue
            Xc2      = X[labels == c2]
            d2       = np.sqrt(np.sum((Xc[:, None] - Xc2[None])**2, axis=2)).mean(axis=1)
            b[idx_c] = np.minimum(b[idx_c], d2)

    s = (b - a) / np.maximum(a, b)
    return float(s.mean())


# ═══════════════════════════════════════════════
# Consensus vote  (adaptive weighting)
#
# Key insight: Silhouette and Gap Stat both tend to default to k=2
# on smooth featureless curves (no true clusters). We detect this
# case via the "silhouette strength" — if the best silhouette score
# is low (<0.5) AND silhouette scores are monotonically falling,
# the data has weak/no cluster structure and we weight Kneedle more
# heavily since it is geometry-based and not fooled by the baseline.
# ═══════════════════════════════════════════════

def consensus(votes: dict[str, int],
              k_values: list[int],
              sil_scores: dict[int, float]) -> int:
    """
    Adaptive weighted consensus.

    Weak-structure is detected when ALL three hold:
      1. max silhouette < 0.5  (poor absolute separation)
      2. silhouette peak is at k=2  (trivial first-split winner)
      3. relative drop from peak to k=3 > 15% of total sil range
         (steep fall = baseline effect, not a real cluster signal)
    In that case Kneedle is upweighted 8x vs Gap/Silhouette.
    """
    if not sil_scores:
        scores = {k: sum(1.0 / (abs(k - v) + 1) for v in votes.values())
                  for k in k_values}
        return max(scores, key=scores.get)

    ks_sorted = sorted(sil_scores.keys())
    sil_vals  = [sil_scores[k] for k in ks_sorted]
    max_sil   = max(sil_vals)
    min_sil   = min(sil_vals)
    sil_range = max_sil - min_sil if max_sil != min_sil else 1e-9

    peak_idx      = int(np.argmax(sil_vals))
    peak_k        = ks_sorted[peak_idx]
    drop_after    = (sil_vals[peak_idx] - sil_vals[peak_idx + 1]
                     if peak_idx + 1 < len(sil_vals) else 0.0)
    relative_drop = drop_after / sil_range

    weak_structure = (max_sil < 0.5) and (peak_k == 2) and (relative_drop > 0.15)

    print(f"[INFO] Sil diagnostics: max={max_sil:.3f}, peak_k={peak_k}, "
          f"rel_drop={relative_drop:.3f} -> weak_structure={weak_structure}",
          file=sys.stderr)

    if weak_structure:
        # Kneedle dominates; Gap Stat and Silhouette almost ignored
        weights = {"Kneedle": 8.0, "Gap Stat": 0.5, "Silhouette": 0.5}
        print("[INFO] Weak structure detected -> upweighting Kneedle (8x)",
              file=sys.stderr)
    else:
        weights = {"Kneedle": 1.0, "Gap Stat": 1.0, "Silhouette": 1.0}

    scores = {k: 0.0 for k in k_values}
    for method, vk in votes.items():
        w = weights.get(method, 1.0)
        for k in k_values:
            scores[k] += w / (abs(k - vk) + 1)

    return max(scores, key=scores.get)


# ═══════════════════════════════════════════════
# Plot  (2 panels)
# ═══════════════════════════════════════════════

def make_plot(k_values, objectives, sil_scores, optimal_k, votes,
              out_path="plot.png"):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: WCSS elbow ──
    ax1.plot(k_values, objectives, marker="o", linewidth=2,
             color="#2563EB", markersize=6, label="WCSS")
    ax1.axvline(x=optimal_k, color="#DC2626", linestyle="--",
                linewidth=2, label=f"Consensus k = {optimal_k}")
    ax1.scatter([optimal_k], [objectives[optimal_k - 1]],
                color="#DC2626", s=120, zorder=5)

    method_colors = {"Kneedle": "#F59E0B", "Gap Stat": "#10B981", "Silhouette": "#8B5CF6"}
    for method, vk in votes.items():
        ax1.axvline(x=vk, color=method_colors[method], linestyle=":",
                    linewidth=1.3, alpha=0.85, label=f"{method} → k={vk}")

    ax1.set_xlabel("Number of clusters k", fontsize=12)
    ax1.set_ylabel("WCSS", fontsize=12)
    ax1.set_title("WCSS vs. k  (Elbow)", fontsize=13)
    ax1.set_xticks(k_values)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # ── Panel 2: Silhouette ──
    ks  = sorted(sil_scores.keys())
    ss  = [sil_scores[k] for k in ks]
    bar_colors = ["#DC2626" if k == optimal_k else "#93C5FD" for k in ks]
    ax2.bar(ks, ss, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax2.axvline(x=optimal_k, color="#DC2626", linestyle="--",
                linewidth=2, label=f"Consensus k = {optimal_k}")
    ax2.set_xlabel("Number of clusters k", fontsize=12)
    ax2.set_ylabel("Mean Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score vs. k", fontsize=13)
    ax2.set_xticks(ks)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")

    fig.suptitle(f"k-means Model Selection  |  Optimal k = {optimal_k}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to {out_path}", file=sys.stderr)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Q1.py <dataset_num>  OR  python3 Q1.py <path>.npy",
              file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    if arg.endswith(".npy"):
        X = load_from_npy(arg)
        print(f"[INFO] Loaded .npy  shape={X.shape}", file=sys.stderr)
    else:
        X = load_from_api(int(arg))
        print(f"[INFO] Loaded API dataset {arg}  shape={X.shape}", file=sys.stderr)
    # X = load_from_api(2)
    K_MAX    = 15
    k_values = list(range(1, K_MAX + 1))

    # ── Step 1: k-means for all k ──
    print("[INFO] Running k-means...", file=sys.stderr)
    objectives: list[float]          = []
    all_labels: dict[int, np.ndarray] = {}
    for k in k_values:
        wcss, labels = best_kmeans(X, k, n_restarts=10, seed=42)
        objectives.append(wcss)
        all_labels[k] = labels
        print(f"  k={k:2d}  WCSS={wcss:.4f}", file=sys.stderr)

    # ── Step 2: silhouette (subsample if large) ──
    MAX_SIL = 2000
    Xs, lbl_sub = X, all_labels
    if X.shape[0] > MAX_SIL:
        idx_sub = np.random.default_rng(99).choice(X.shape[0], MAX_SIL, replace=False)
        Xs      = X[idx_sub]
        lbl_sub = {k: all_labels[k][idx_sub] for k in k_values}

    print("[INFO] Computing silhouette scores...", file=sys.stderr)
    sil_scores: dict[int, float] = {}
    for k in k_values:
        if k == 1:
            continue
        sil_scores[k] = silhouette_score(Xs, lbl_sub[k])
        print(f"  silhouette k={k:2d}  score={sil_scores[k]:.4f}", file=sys.stderr)

    # ── Step 3: gap statistic ──
    print("[INFO] Computing gap statistic...", file=sys.stderr)
    k_gap = gap_statistic(X, objectives, B=10, seed=0)

    # ── Step 4: individual votes ──
    k_kneedle = kneedle(objectives)
    k_sil     = max(sil_scores, key=sil_scores.get)
    votes     = {"Kneedle": k_kneedle, "Gap Stat": k_gap, "Silhouette": k_sil}
    print(f"[INFO] Votes  Kneedle={k_kneedle}  GapStat={k_gap}  Silhouette={k_sil}",
          file=sys.stderr)

    # ── Step 5: consensus ──
    optimal_k = consensus(votes, k_values, sil_scores)
    print(f"[INFO] Consensus optimal k = {optimal_k}", file=sys.stderr)

    # ── Step 6: plot ──
    make_plot(k_values, objectives, sil_scores, optimal_k, votes, "plot.png")

    sys.stderr.flush()
    print(optimal_k)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
