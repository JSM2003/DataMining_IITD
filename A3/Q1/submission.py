"""
COL761 A3 — v12b: streamlined D2, actual-timing-based Pass B, no flat probe overhead
"""

from __future__ import annotations
import os, sys, time, threading
import numpy as np

_DEBUG_FILE = "/tmp/col761_debug.txt"
_t_global   = [time.perf_counter()]

def _log(*args):
    elapsed = time.perf_counter() - _t_global[0]
    line = f"[+{elapsed:6.2f}s] " + " ".join(str(a) for a in args)
    try:
        with open(_DEBUG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def solve(base_vectors, query_vectors, k, K, time_budget):
    _t_global[0] = time.perf_counter()
    try:
        open(_DEBUG_FILE, "w").close()
    except Exception:
        pass

    import faiss
    t_start = time.perf_counter()
    tb      = float(time_budget)

    n_cpu  = _get_ncpu()
    ram_gb = _get_ram_gb()
    _log(f"SYSTEM: n_cpu={n_cpu}, ram_gb={ram_gb:.1f}, budget={tb}s")

    margin   = max(0.5, min(2.0, 0.07 * tb))
    deadline = t_start + tb - margin
    _log(f"margin={margin:.2f}s, effective={tb-margin:.2f}s")

    base_vectors  = _ensure_f32c(base_vectors)
    query_vectors = _ensure_f32c(query_vectors)

    N, d = int(base_vectors.shape[0]), int(base_vectors.shape[1])
    Q    = int(query_vectors.shape[0])
    K    = int(K)
    k    = min(int(k), N)
    _log(f"DATASET: N={N}, d={d}, Q={Q}, k={k}, K={K}")

    def rem():
        return deadline - time.perf_counter()

    if N == 0:
        return np.zeros(K, dtype=np.int64)
    if Q == 0 or k <= 0:
        return _pad_to_K(np.empty(0, dtype=np.int64), K, N)

    path = "D1" if tb > 25.0 else "D2"
    _log(f"PATH: {path}")

    if tb > 25.0:
        neighbors = _d1_path(base_vectors, query_vectors, k,
                             deadline, n_cpu, faiss)
    else:
        neighbors = _d2_path(base_vectors, query_vectors, k,
                             deadline, n_cpu, faiss)

    _log(f"AGGREGATE: neighbors={'None' if neighbors is None else str(neighbors.shape)}")
    if neighbors is not None and neighbors.size > 0:
        n_rows   = neighbors.shape[0]
        flat_idx = neighbors.ravel()
        flat_idx = flat_idx[(flat_idx >= 0) & (flat_idx < N)]
        _log(f"  coverage={n_rows}/{Q}, valid_entries={flat_idx.size}/{n_rows*k}")
        if flat_idx.size:
            counts = np.bincount(flat_idx.astype(np.intp), minlength=N)
            order  = np.argsort(-counts, kind="stable")
            ranked = order[:K].astype(np.int64)
        else:
            ranked = np.empty(0, dtype=np.int64)
    else:
        _log("  WARNING: no neighbors!")
        ranked = np.empty(0, dtype=np.int64)

    result = _pad_to_K(ranked, K, N)
    _log(f"DONE: elapsed={time.perf_counter()-t_start:.3f}s, result={len(result)}")
    return result


# ── D1: shared flat index + threaded query search ─────────────────────

def _d1_path(base, query, k, deadline, n_cpu, faiss):
    faiss.omp_set_num_threads(1)
    _log(f"[D1] building flat index N={base.shape[0]}")
    t0  = time.perf_counter()
    idx = faiss.IndexFlatL2(int(base.shape[1]))
    idx.add(base)
    _log(f"[D1] built in {time.perf_counter()-t0:.3f}s")

    Q         = int(query.shape[0])
    n_workers = max(1, min(n_cpu, 32, Q))
    splits    = np.array_split(np.arange(Q), n_workers)
    results   = [None] * n_workers
    lock      = threading.Lock()

    _log(f"[D1] {n_workers} threads")

    def worker(tid, rng):
        try:
            if time.perf_counter() >= deadline - 1.0:
                return
            _, I = idx.search(query[rng], k)
            with lock:
                results[tid] = (rng, I)
        except Exception as e:
            _log(f"[D1] thread {tid} err: {e}")

    threads = []
    for tid, rng in enumerate(splits):
        if len(rng) == 0:
            continue
        t = threading.Thread(target=worker, args=(tid, rng), daemon=True)
        threads.append((t, tid, rng))
        t.start()

    for t, tid, rng in threads:
        t.join(timeout=max(0.1, deadline - time.perf_counter() - 0.5))

    out = np.full((Q, k), -1, dtype=np.int64)
    filled_count = 0
    for tid, rng in enumerate(splits):
        if results[tid] is not None:
            out[rng] = results[tid][1]
            filled_count += len(rng)
    _log(f"[D1] threads filled {filled_count}/{Q}")

    missing = np.where(out[:, 0] == -1)[0]
    if len(missing) > 0:
        _log(f"[D1] OMP fallback for {len(missing)} rows")
        faiss.omp_set_num_threads(n_cpu)
        for start in range(0, len(missing), 512):
            if time.perf_counter() >= deadline - 0.3:
                break
            sl = missing[start:start + 512]
            try:
                _, I = idx.search(query[sl], k)
                out[sl] = I
            except Exception:
                break

    if out[0, 0] == -1:
        _log("[D1] critical fallback")
        faiss.omp_set_num_threads(n_cpu)
        part, filled = _chunked_search(idx, query, k, deadline, label="D1-fb")
        return part[:filled] if filled > 0 else None

    last = Q
    for i in range(Q - 1, -1, -1):
        if out[i, 0] != -1:
            last = i + 1
            break
    _log(f"[D1] final filled={last}/{Q}")
    return out[:last]


# ── D2: pure IVFFlat, timing-driven, actual-timing Pass B ─────────────

def _d2_path(base, query, k, deadline, n_cpu, faiss):
    faiss.omp_set_num_threads(n_cpu)
    N, d = int(base.shape[0]), int(base.shape[1])
    Q    = int(query.shape[0])

    def rem():
        return deadline - time.perf_counter()

    _log(f"[D2] rem={rem():.2f}s")

    if N < 8192:
        _log("[D2] tiny N → flat")
        idx = faiss.IndexFlatL2(d)
        idx.add(base)
        p, f = _chunked_search(idx, query, k, deadline, label="D2-tiny")
        return p[:f] if f > 0 else None

    # ── Step 1: IVF parameter selection (CPU-speed adaptive) ──────────
    # Measure raw OMP throughput with a tiny inline probe (no index build).
    # Use dot-product on a small slice: tells us OMP scaling factor.
    _log(f"[D2] Step1: OMP speed probe")
    probe_sz = min(10_000, N)
    t_sp = time.perf_counter()
    # Cheap OMP-parallel operation: norm of a slice
    _ = np.linalg.norm(base[:probe_sz], axis=1)
    omp_probe_t = max(time.perf_counter() - t_sp, 1e-9)
    # Baseline (1 thread) would take longer; ratio approximates parallelism
    # We use a simpler heuristic: just use n_cpu directly
    _log(f"[D2] omp_probe={omp_probe_t:.4f}s on {probe_sz} vecs")

    # nlist: sqrt(N) scaled by CPU count
    # More CPUs → can afford more centroids → better recall
    cpu_scale   = min(2.0, max(0.5, n_cpu / 12.0))
    nlist_raw   = int(np.sqrt(N) * cpu_scale * 0.6)
    nlist       = int(np.clip(nlist_raw, 32, 4096))
    while nlist > 32 and N < nlist * 16:
        nlist //= 2

    # n_train: adaptive to time budget (leave ≥65% of budget for search)
    max_build_t  = rem() * 0.30                     # at most 30% of rem for build
    n_train_time = int(max_build_t * n_cpu * 3000)  # ~3000 train pts/s/thread
    n_train      = min(N, max(nlist * 30, 20_000), n_train_time)
    n_train      = max(n_train, min(nlist * 10, N))

    _log(f"[D2] cpu_scale={cpu_scale:.2f}, nlist={nlist}, n_train={n_train}, max_build_t={max_build_t:.2f}s")

    # ── Step 2: build IVFFlat ─────────────────────────────────────────
    _log(f"[D2] Step2: build IVFFlat")
    rng = np.random.default_rng(0)
    if n_train < N:
        tidx  = rng.choice(N, size=n_train, replace=False)
        train = np.ascontiguousarray(base[tidx])
    else:
        train = base

    t_build = time.perf_counter()
    try:
        quantizer = faiss.IndexFlatL2(d)
        ivf       = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        ivf.train(train)
        ivf.add(base)
    except Exception as e:
        _log(f"[D2] build FAILED: {e}")
        return None

    build_t = time.perf_counter() - t_build
    _log(f"[D2] built in {build_t:.3f}s, rem={rem():.2f}s")

    if rem() <= 0.5:
        _log("[D2] no time after build!")
        return None

    # ── Step 3: calibration probe ─────────────────────────────────────
    # Use a small nprobe to measure per-query cost; scale up from there.
    _log(f"[D2] Step3: calibration")
    cal_nprobe = int(max(4, min(32, nlist // 8)))
    ivf.nprobe = cal_nprobe
    cal_n      = max(64, min(512, Q // 20))

    try:
        ivf.search(query[:min(32, Q)], k)              # warmup
        t_cal = time.perf_counter()
        _, I_cal = ivf.search(query[:cal_n], k)
        cal_time = max(1e-9, time.perf_counter() - t_cal)
    except Exception as e:
        _log(f"[D2] cal FAILED: {e}")
        return None

    # time for all Q at cal_nprobe:
    full_at_cal     = cal_time * (Q / cal_n)
    # cost per unit nprobe (linear scaling):
    time_per_nprobe = full_at_cal / cal_nprobe

    _log(f"[D2] cal_nprobe={cal_nprobe}, cal_n={cal_n}, cal_time={cal_time:.4f}s")
    _log(f"[D2] full_at_cal={full_at_cal:.3f}s, time_per_nprobe={time_per_nprobe:.4f}s")

    # ── Step 4: Pass A ────────────────────────────────────────────────
    # Use 78% of remaining time budget for Pass A.
    # Clamp nprobe_A ≥ cal_nprobe (never regress, no IVFPQ escape).
    _log(f"[D2] Step4: Pass A, rem={rem():.2f}s")
    usable_A = 0.78 * rem()
    nprobe_A = int(usable_A / max(time_per_nprobe, 1e-9))
    nprobe_A = int(max(cal_nprobe, min(nprobe_A, nlist)))
    est_A    = nprobe_A * time_per_nprobe
    _log(f"[D2] nprobe_A={nprobe_A} (est={est_A:.2f}s, nlist={nlist})")

    ivf.nprobe = nprobe_A
    t_a = time.perf_counter()
    chunks_A = max(8, min(40, Q // 200))
    part_a, filled_a = _chunked_search(ivf, query, k, deadline,
                                        target_chunks=chunks_A,
                                        safety_buffer=0.25, label="D2-A")
    elapsed_a = time.perf_counter() - t_a
    _log(f"[D2] Pass A: filled={filled_a}/{Q} ({100*filled_a/max(Q,1):.1f}%) in {elapsed_a:.3f}s, rem={rem():.2f}s")

    if filled_a <= 0:
        _log("[D2] Pass A empty! cal fallback")
        return I_cal

    best = part_a[:filled_a]

    # ── Step 5: Pass B using ACTUAL Pass A timing ─────────────────────
    if filled_a == Q and rem() >= 1.0:
        _log(f"[D2] Step5: Pass B, rem={rem():.2f}s")

        # Use actual elapsed_a to compute real time_per_nprobe at nprobe_A
        # This is more accurate than the calibration estimate for large nprobe.
        actual_tpn = elapsed_a / max(nprobe_A, 1)
        _log(f"[D2] cal_tpn={time_per_nprobe:.4f}s, actual_tpn={actual_tpn:.4f}s")

        usable_B = 0.90 * rem()
        # Use actual timing for the estimate
        nprobe_B = int(usable_B / max(actual_tpn, 1e-9))
        nprobe_B = int(max(nprobe_A + 1, min(nprobe_B, nlist)))
        est_B    = nprobe_B * actual_tpn
        _log(f"[D2] nprobe_B={nprobe_B} (est={est_B:.2f}s, nlist={nlist})")

        if nprobe_B > nprobe_A:
            ivf.nprobe = nprobe_B
            chunks_B = max(8, min(40, Q // 200))
            t_b = time.perf_counter()
            part_b, filled_b = _chunked_search(ivf, query, k, deadline,
                                                target_chunks=chunks_B,
                                                safety_buffer=0.25, label="D2-B")
            elapsed_b = time.perf_counter() - t_b
            _log(f"[D2] Pass B: filled={filled_b}/{Q} in {elapsed_b:.3f}s, rem={rem():.2f}s")
            if filled_b == Q:
                _log("[D2] Pass B complete → using")
                return part_b
            else:
                _log(f"[D2] Pass B partial → keeping A")
        else:
            _log(f"[D2] nprobe_B={nprobe_B} not better than nprobe_A={nprobe_A} or nlist={nlist} reached")
    else:
        _log(f"[D2] Skip B: filled_a={filled_a}, rem={rem():.2f}s")

    return best


# ── Chunked search ────────────────────────────────────────────────────

def _chunked_search(index, query, k, deadline, target_chunks=16,
                    safety_buffer=0.15, label=""):
    Q = int(query.shape[0])
    if Q == 0:
        return np.empty((0, k), dtype=np.int64), 0
    chunk_size = max(64, (Q + target_chunks - 1) // target_chunks)
    chunk_size = min(chunk_size, Q)
    neighbors  = np.empty((Q, k), dtype=np.int64)
    filled = start = chunk_idx = 0
    while start < Q:
        if time.perf_counter() >= deadline - safety_buffer:
            _log(f"  [{label}] deadline at chunk {chunk_idx}, filled={filled}/{Q}")
            break
        end = min(start + chunk_size, Q)
        try:
            _, I = index.search(query[start:end], k)
            neighbors[start:end] = I
            filled = end
        except Exception as e:
            _log(f"  [{label}] chunk {chunk_idx} err: {e}")
            break
        start = end
        chunk_idx += 1
    return neighbors, filled


# ── System detection ──────────────────────────────────────────────────

def _get_ncpu() -> int:
    # 1. sched_getaffinity: respects SLURM/cgroup limits (most accurate on HPC)
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n
    except (AttributeError, OSError):
        pass
    # 2. /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            n = f.read().count("processor\t:")
        if n > 0:
            return n
    except Exception:
        pass
    # 3. multiprocessing
    try:
        import multiprocessing as mp
        return max(1, mp.cpu_count() or 1)
    except Exception:
        return 1


def _get_ram_gb() -> float:
    # MemAvailable: current free RAM (best for capacity planning)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    # MemTotal fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 32.0


# ── Array utilities ───────────────────────────────────────────────────

def _ensure_f32c(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def _pad_to_K(ranked: np.ndarray, K: int, N: int) -> np.ndarray:
    ranked = np.asarray(ranked, dtype=np.int64).ravel()
    if ranked.shape[0] >= K:
        return ranked[:K]
    if N == 0:
        return np.zeros(K, dtype=np.int64)
    seen   = set(int(x) for x in ranked.tolist())
    needed = K - ranked.shape[0]
    extra, i = [], 0
    while len(extra) < needed and i < N:
        if i not in seen:
            extra.append(i)
        i += 1
    if len(extra) < needed:
        extra.extend([0] * (needed - len(extra)))
    return np.concatenate([ranked, np.asarray(extra, dtype=np.int64)])[:K]
