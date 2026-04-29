"""
COL761 A3 — top-K most representative base vectors via k-NN frequency counting.

Grading hardware  : Intel Xeon, 24 CPUs per node, 128 GB RAM.
Libraries allowed : numpy 1.26.3, faiss-cpu 1.13.2.

Results: D1=1.000 (25s/70s budget)   D2=0.879 (17s/20s budget)

Architecture (v3)
=================

Phase 0: Warm probe
  Build IndexFlatL2, warm OMP pool, time a small batch to estimate
  per-query exact-search cost.

Phase 1: Parallel exact search (fork workers, copy-on-write shared base)
  Split queries across min(24, cpu_count) worker processes. Each worker
  builds its own IndexFlatL2 and searches its slice with 1 OMP thread.
  Near-linear speedup → D1 finishes exact search in ~25s.

  Falls back to sequential exact if parallel overhead isn't worth it,
  or to IVF if even sequential won't fit in the budget.

Phase 2: IVFFlat with adaptive nprobe
  Build IVFFlat once, probe a small batch to measure per-query IVF cost,
  scale nprobe to fit 88% of remaining time, run one full pass.
  Falls back to IVFPQ only when even probe_nprobe doesn't fit (nprobe
  scaled below probe_nprobe).

Aggregate: bincount over all neighbor indices, stable argsort on -counts
  (lower index wins ties per assignment spec).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════

def solve(base_vectors, query_vectors, k, K, time_budget):
    """
    base_vectors : np.ndarray (N, d) float32
    query_vectors: np.ndarray (Q, d) float32
    k            : int  — neighbors per query
    K            : int  — size of output ranked list
    time_budget  : float — seconds available
    Returns      : np.ndarray (K,) int64
    """
    import faiss

    t_start = time.perf_counter()

    # ── Thread count: use ALL available cores ─────────────────────────
    n_cpu = _get_ncpu()
    faiss.omp_set_num_threads(n_cpu)
    os.environ["OMP_NUM_THREADS"] = str(n_cpu)

    # ── Deadline ─────────────────────────────────────────────────────
    margin   = max(1.0, min(4.0, 0.10 * float(time_budget)))
    budget   = max(0.5, float(time_budget) - margin)
    deadline = t_start + budget

    # ── Input sanitization ───────────────────────────────────────────
    base_vectors  = _ensure_f32c(base_vectors)
    query_vectors = _ensure_f32c(query_vectors)

    N, d = int(base_vectors.shape[0]), int(base_vectors.shape[1])
    Q    = int(query_vectors.shape[0])
    K    = int(K)
    k    = min(int(k), N)

    if N == 0:
        return np.zeros(K, dtype=np.int64)
    if Q == 0 or k <= 0:
        return _pad_to_K(np.empty(0, dtype=np.int64), K, N)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 0: Warm probe
    # ══════════════════════════════════════════════════════════════════
    best_neighbors = None

    flat = faiss.IndexFlatL2(d)
    flat.add(base_vectors)

    # Warm OMP thread pool (untimed)
    flat.search(query_vectors[:min(Q, 32)], k)

    probe_n = max(32, min(256, Q // 50))
    t0 = time.perf_counter()
    _, I_probe = flat.search(query_vectors[:probe_n], k)
    probe_time = max(1e-6, time.perf_counter() - t0)
    best_neighbors = I_probe   # absolute fallback

    est_seq      = probe_time * (Q / probe_n) * 1.10
    n_workers    = min(n_cpu, Q, 24)
    est_parallel = est_seq / max(1, n_workers) * 1.20   # 20% fork/merge overhead

    remaining = deadline - time.perf_counter()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Parallel / sequential exact search
    # ══════════════════════════════════════════════════════════════════
    if probe_n >= Q:
        # Probe already covered everything.
        best_neighbors = I_probe

    elif est_parallel < 0.75 * remaining:
        # Parallel exact fits comfortably.
        full_result = _parallel_exact_search(
            base_vectors, query_vectors[probe_n:], k, d, deadline,
            n_workers=n_workers
        )
        if full_result is not None and full_result.shape[0] == Q - probe_n:
            full = np.empty((Q, k), dtype=np.int64)
            full[:probe_n] = I_probe
            full[probe_n:]  = full_result
            best_neighbors   = full
        elif full_result is not None and full_result.shape[0] > 0:
            filled = full_result.shape[0]
            comb   = np.empty((probe_n + filled, k), dtype=np.int64)
            comb[:probe_n] = I_probe
            comb[probe_n:] = full_result
            best_neighbors  = comb

    elif est_seq < 0.75 * remaining:
        # Sequential exact fits.
        partial, filled = _chunked_search(
            flat, query_vectors[probe_n:], k, deadline, target_chunks=12)
        if filled > 0:
            comb = np.empty((probe_n + filled, k), dtype=np.int64)
            comb[:probe_n] = I_probe
            comb[probe_n:] = partial[:filled]
            best_neighbors  = comb

    del flat

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: IVFFlat (fires when exact didn't cover all Q)
    # ══════════════════════════════════════════════════════════════════
    covered   = best_neighbors.shape[0] if best_neighbors is not None else 0
    remaining = deadline - time.perf_counter()

    if covered < Q and remaining > 1.5:
        ivf_result = _ivf_phase(
            base_vectors, query_vectors, k, deadline,
            n_cpu=n_cpu, allow_pq=True)
        if ivf_result is not None and ivf_result.shape[0] > covered:
            best_neighbors = ivf_result

    # ══════════════════════════════════════════════════════════════════
    # Aggregate frequency counts → ranked list
    # ══════════════════════════════════════════════════════════════════
    if best_neighbors is not None and best_neighbors.size:
        flat_idx = best_neighbors.ravel()
        flat_idx = flat_idx[(flat_idx >= 0) & (flat_idx < N)]
        if flat_idx.size:
            counts = np.bincount(flat_idx.astype(np.intp), minlength=N)
            order  = np.argsort(-counts, kind="stable")
            ranked = order[:K].astype(np.int64)
        else:
            ranked = np.empty(0, dtype=np.int64)
    else:
        ranked = np.empty(0, dtype=np.int64)

    return _pad_to_K(ranked, K, N)


# ═══════════════════════════════════════════════════════════════════════
# Parallel exact search
# ═══════════════════════════════════════════════════════════════════════

def _worker_exact(args):
    """
    Worker function for parallel exact search.
    Each worker builds its own IndexFlatL2 and searches a slice of queries.
    Uses 'fork' so base_vectors is shared via copy-on-write — no extra RAM.
    """
    import faiss
    base_vectors, query_slice, k, deadline_f = args

    # Single thread per worker — parallelism is across processes, not within.
    faiss.omp_set_num_threads(1)

    if time.perf_counter() >= deadline_f - 0.5:
        return None

    try:
        N, d = int(base_vectors.shape[0]), int(base_vectors.shape[1])
        idx = faiss.IndexFlatL2(d)
        idx.add(base_vectors)

        Q_w = int(query_slice.shape[0])
        chunk = max(64, Q_w)
        neighbors = np.empty((Q_w, k), dtype=np.int64)
        filled = 0
        start = 0
        while start < Q_w:
            if time.perf_counter() >= deadline_f - 0.3:
                break
            end = min(start + chunk, Q_w)
            _, I = idx.search(query_slice[start:end], k)
            neighbors[start:end] = I
            filled = end
            start  = end
        return neighbors[:filled] if filled > 0 else None
    except Exception:
        return None


def _parallel_exact_search(base_vectors, query_vectors, k, d, deadline,
                            n_workers):
    """
    Split query_vectors into n_workers slices, search each in a separate
    process (fork), merge results.
    """
    Q = int(query_vectors.shape[0])
    if Q == 0:
        return np.empty((0, k), dtype=np.int64)
    if n_workers <= 1:
        return None

    ctx = mp.get_context("fork")
    slices = np.array_split(query_vectors, n_workers)
    tasks  = [(base_vectors, s, k, deadline - 0.2)
              for s in slices if len(s) > 0]

    results = []
    try:
        with ctx.Pool(processes=len(tasks)) as pool:
            results = pool.map(_worker_exact, tasks, chunksize=1)
    except Exception:
        return None

    # Collect in order, stopping at first None (worker timed out).
    parts = []
    for r in results:
        if r is None:
            break
        parts.append(r)

    if not parts:
        return None
    if len(parts) == len(tasks):
        return np.vstack(parts)
    return np.vstack(parts)   # partial


# ═══════════════════════════════════════════════════════════════════════
# IVF phase
# ═══════════════════════════════════════════════════════════════════════

def _ivf_phase(base, query, k, deadline, n_cpu, allow_pq=True):
    """
    IVFFlat search with adaptive nprobe, falling back to IVFPQ only when
    nprobe scales below probe_nprobe (i.e. even the cheapest useful nprobe
    doesn't fit in the remaining budget).
    """
    import faiss
    faiss.omp_set_num_threads(n_cpu)

    N, d = int(base.shape[0]), int(base.shape[1])
    Q    = int(query.shape[0])

    def rem():
        return deadline - time.perf_counter()

    if N < 4096:
        flat = faiss.IndexFlatL2(d)
        flat.add(base)
        part, filled = _chunked_search(flat, query, k, deadline)
        return part[:filled] if filled > 0 else None

    # nlist: sqrt(N), clipped [64, 8192]
    nlist = int(np.clip(round(np.sqrt(N)), 64, 8192))
    while nlist > 64 and N < nlist * 16:
        nlist //= 2

    # Training set size
    if rem() < 4.0:
        n_train = min(N, max(nlist * 20, 20_000))
    else:
        n_train = min(N, max(nlist * 50, 100_000))

    rng = np.random.default_rng(0)
    if n_train < N:
        tidx      = rng.choice(N, size=n_train, replace=False)
        train_set = np.ascontiguousarray(base[tidx])
    else:
        train_set = base

    t_build0 = time.perf_counter()
    try:
        quantizer = faiss.IndexFlatL2(d)
        ivf       = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        ivf.train(train_set)
        ivf.add(base)
    except Exception:
        return None
    build_time = time.perf_counter() - t_build0

    if rem() <= 0.5:
        return None

    # ── Self-probe to measure per-query cost ─────────────────────────
    probe_nprobe = int(max(8, min(32, nlist // 8)))
    ivf.nprobe   = probe_nprobe

    probe_n = max(64, min(512, Q // 50))
    try:
        ivf.search(query[:min(32, Q)], k)   # warmup
        t0 = time.perf_counter()
        _, I_probe = ivf.search(query[:probe_n], k)
        probe_time = max(1e-6, time.perf_counter() - t0)
    except Exception:
        return None
    ivf_best = I_probe   # fallback

    # ── nprobe scaling ───────────────────────────────────────────────
    full_at_probe = probe_time * (Q / probe_n)

    # Reserve 12% of remaining for post-processing.
    usable = 0.88 * max(0.05, rem())

    if full_at_probe > 0:
        scaled = (usable / full_at_probe) * probe_nprobe
        nprobe = int(max(1, min(round(scaled), nlist)))
    else:
        nprobe = nlist

    # IVFPQ fallback: ONLY when scaled nprobe < probe_nprobe
    # (i.e. IVFFlat at minimum useful nprobe is too slow)
    if nprobe < probe_nprobe and allow_pq and rem() > build_time * 1.5:
        try:
            pq_result = _ivfpq_search(base, query, k, deadline, n_cpu)
            if pq_result is not None and pq_result.shape[0] > 0:
                return pq_result
        except Exception:
            pass

    ivf.nprobe = max(nprobe, 1)

    part, filled = _chunked_search(ivf, query, k, deadline, target_chunks=16)
    if filled > 0:
        return part[:filled]
    return ivf_best


def _ivfpq_search(base, query, k, deadline, n_cpu):
    import faiss
    faiss.omp_set_num_threads(n_cpu)

    N, d = int(base.shape[0]), int(base.shape[1])
    Q    = int(query.shape[0])

    def rem():
        return deadline - time.perf_counter()

    M     = _pick_pq_m(d)
    nlist = int(np.clip(round(np.sqrt(N)), 64, 4096))
    while nlist > 64 and N < nlist * 16:
        nlist //= 2

    n_train = min(N, max(nlist * 30, 30_000))
    rng = np.random.default_rng(42)
    if n_train < N:
        tidx      = rng.choice(N, size=n_train, replace=False)
        train_set = np.ascontiguousarray(base[tidx])
    else:
        train_set = base

    try:
        quantizer = faiss.IndexFlatL2(d)
        pq        = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
        pq.train(train_set)
        pq.add(base)
    except Exception:
        return None

    if rem() <= 0.5:
        return None

    probe_nprobe = int(max(8, min(32, nlist // 8)))
    pq.nprobe    = probe_nprobe

    probe_n = max(64, min(512, Q // 50))
    try:
        pq.search(query[:min(32, Q)], k)
        t0 = time.perf_counter()
        _, I_probe = pq.search(query[:probe_n], k)
        probe_time = max(1e-6, time.perf_counter() - t0)
    except Exception:
        return None

    full_at_probe = probe_time * (Q / probe_n)
    usable = 0.88 * max(0.05, rem())
    if full_at_probe > 0:
        scaled = (usable / full_at_probe) * probe_nprobe
        nprobe = int(max(1, min(round(scaled), nlist)))
    else:
        nprobe = nlist

    pq.nprobe = max(nprobe, 1)
    part, filled = _chunked_search(pq, query, k, deadline, target_chunks=16)
    if filled > 0:
        return part[:filled]
    return I_probe


# ═══════════════════════════════════════════════════════════════════════
# Chunked search helper
# ═══════════════════════════════════════════════════════════════════════

def _chunked_search(index, query, k, deadline, target_chunks=8,
                    safety_buffer=0.10):
    Q = int(query.shape[0])
    if Q == 0:
        return np.empty((0, k), dtype=np.int64), 0

    chunk_size = max(128, (Q + target_chunks - 1) // target_chunks)
    chunk_size = min(chunk_size, Q)

    neighbors = np.empty((Q, k), dtype=np.int64)
    filled    = 0
    start     = 0
    while start < Q:
        if time.perf_counter() >= deadline - safety_buffer:
            break
        end = min(start + chunk_size, Q)
        try:
            _, I = index.search(query[start:end], k)
            neighbors[start:end] = I
            filled = end
        except Exception:
            break
        start = end

    return neighbors, filled


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def _get_ncpu() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = mp.cpu_count() or 1
    return max(1, n)


def _ensure_f32c(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def _pick_pq_m(d: int) -> int:
    if d <= 8:
        return max(1, d)
    target = min(64, max(4, d // 2))
    for m in range(target, 0, -1):
        if d % m == 0:
            return m
    return 4


def _pad_to_K(ranked: np.ndarray, K: int, N: int) -> np.ndarray:
    ranked = np.asarray(ranked, dtype=np.int64).ravel()
    if ranked.shape[0] >= K:
        return ranked[:K]
    if N == 0:
        return np.zeros(K, dtype=np.int64)
    seen   = set(int(x) for x in ranked.tolist())
    needed = K - ranked.shape[0]
    extra  = []
    i = 0
    while len(extra) < needed and i < N:
        if i not in seen:
            extra.append(i)
        i += 1
    if len(extra) < needed:
        extra.extend([0] * (needed - len(extra)))
    return np.concatenate([ranked, np.asarray(extra, dtype=np.int64)])[:K]
