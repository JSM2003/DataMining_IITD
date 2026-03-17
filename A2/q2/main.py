import random
import sys
import heapq
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Graph / seed loading
# ---------------------------------------------------------------------------

def load_graph(path):
    nodes    = set()
    adj      = defaultdict(list)
    edge_set = set()
    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                print(f"[WARN] line {lineno} has <3 fields — skipping: {line!r}")
                continue
            try:
                u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                print(f"[WARN] line {lineno} could not be parsed — skipping: {line!r}")
                continue
            if not (0.0 < p <= 1.0):
                print(f"[WARN] line {lineno}: p={p} outside (0,1] for edge ({u},{v})")
            nodes.add(u); nodes.add(v)
            adj[u].append((v, p))
            edge_set.add((u, v))
    return nodes, dict(adj), edge_set


def load_seeds(path):
    seeds = set()
    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                seeds.add(int(line))
            except ValueError:
                print(f"[WARN] seed file line {lineno} is not an integer — skipping: {line!r}")
    if not seeds:
        raise ValueError(f"Seed file '{path}' yielded no valid node IDs.")
    return frozenset(seeds)


# ---------------------------------------------------------------------------
# Hop-reachable zone
# ---------------------------------------------------------------------------

def compute_hop_reachable(adj, source_set, hops):
    """
    BFS from seeds to find every node reachable within `hops` steps.
    Returns dict { node -> min distance from any seed }.
    When hops is None we still run BFS (unlimited) so callers always
    receive a valid hop_dist dict — this fixes the missing-zone-filter
    bug on dataset1 (hops=None previously skipped this entirely).
    """
    dist  = {s: 0 for s in source_set}
    queue = deque(source_set)
    while queue:
        u = queue.popleft()
        if hops is not None and dist[u] >= hops:
            continue
        for (v, _) in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Reverse adjacency
# ---------------------------------------------------------------------------

def build_reverse_adj(adj):
    rev = defaultdict(list)
    for u, neighbors in adj.items():
        for (v, p) in neighbors:
            rev[v].append((u, p))
    return dict(rev)


# ---------------------------------------------------------------------------
# RR-set sampling  (seed-aware + hop-correct)
# ---------------------------------------------------------------------------

def sample_rr_set(rev_adj, source_set, target_node, blocked, rng, hop_dist):
    """
    Sample one RR set rooted at target_node.

    Returns (edge_set, seed_found):
        edge_set  : set of (u,v) edges traversed in this realisation
        seed_found: True  → a seed was reached; this RR set is discarded by
                    the caller (no non-seed edge can prevent burning)

    KEY FIX — score edges, not nodes
    ---------------------------------
    The previous implementations scored a candidate edge (u,v) by checking
    whether v appeared in the RR node-set.  That is wrong: blocking (u,v)
    only helps if the RR path actually *used* that specific edge.  Two
    different edges entering v are both credited when only one is blocked.
    The correct quantity is: does this RR set contain the directed edge (u,v)?
    We collect the traversed edge set here and the CELF loop scores edges
    by counting how many RR edge-sets contain them.

    KEY FIX — hop constraint mirror
    --------------------------------
    During reverse BFS on edge v←u, only proceed if:
      • u and v are both in hop_dist (within the forward fire zone)
      • hop_dist[u] < hop_dist[v]  (u is strictly closer to seeds than v)
    This mirrors the forward hop model exactly: fire travels from a node
    closer to the seeds toward a node farther away, never the reverse.
    """
    visited    = {target_node}
    frontier   = [target_node]
    edge_set   = set()
    seed_found = (target_node in source_set)

    while frontier and not seed_found:
        next_frontier = []
        for v in frontier:
            for (u, p) in rev_adj.get(v, ()):
                if u in visited:
                    continue
                if (u, v) in blocked:
                    continue

                # Hop constraint: mirror the forward causal direction
                if u not in hop_dist or v not in hop_dist:
                    continue
                if hop_dist[u] >= hop_dist[v]:
                    continue

                if rng.random() < p:
                    visited.add(u)
                    edge_set.add((u, v))          # ← collect the edge, not just the node
                    if u in source_set:
                        seed_found = True
                        break
                    next_frontier.append(u)
            if seed_found:
                break
        frontier = next_frontier

    return edge_set, seed_found


def build_rr_sets(rev_adj, source_set, hop_dist, blocked, rng,
                  n_rr, max_attempt_factor=20):
    """
    Collect n_rr informative RR edge-sets.

    FIX — larger attempt budget
    ---------------------------
    With a tight hop zone and low edge probabilities, most sampled RR sets
    are trivial (seed unreachable → empty edge set, OR seed immediately
    reachable → discarded).  We raise the attempt cap to 20×n_rr (was 5×)
    so the quota is reliably filled even on sparse graphs.

    FIX — targets sampled from hop zone only
    -----------------------------------------
    Targets are drawn from nodes inside hop_dist that are not seeds.
    Nodes outside the hop zone can never burn, so an RR set rooted there
    is always empty and wasteful.
    """
    candidate_targets = [n for n in hop_dist if n not in source_set]
    if not candidate_targets:
        print("[WARN] No candidate target nodes for RR sampling.")
        return []

    rr_sets      = []
    max_attempts = n_rr * max_attempt_factor
    attempts     = 0

    while len(rr_sets) < n_rr and attempts < max_attempts:
        attempts += 1
        target = rng.choice(candidate_targets)
        edges, seed_found = sample_rr_set(
            rev_adj, source_set, target, blocked, rng, hop_dist
        )
        # Informative only: seed not trivially inside, and at least one
        # edge exists that could be blocked
        if not seed_found and edges:
            rr_sets.append(edges)

    if len(rr_sets) < n_rr:
        print(f"[WARN] Only collected {len(rr_sets)}/{n_rr} RR sets "
              f"after {attempts} attempts — graph may be very sparse.")
    return rr_sets


# ---------------------------------------------------------------------------
# Candidate edge pre-filtering
# ---------------------------------------------------------------------------

def score_edge_proxy(u, v, p, adj, hop_dist):
    """
    Cheap proxy: p × |nodes reachable from v within 3 hops inside hop zone|.
    Used only to rank candidates before the expensive RR-set pass.
    """
    visited  = {v}
    frontier = [v]
    for _ in range(3):
        nxt = []
        for node in frontier:
            for (nb, _) in adj.get(node, ()):
                if nb not in visited and nb in hop_dist:
                    visited.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return p * len(visited)


def get_candidate_edges(adj, hop_dist, max_candidates=5000):
    """
    Return the top-max_candidates edges (both endpoints in hop_dist)
    ranked by proxy score.

    FIX — raised cap to 5000
    -------------------------
    A cap of 2000-3000 may prune away edges that are the true greedy
    optimum on the real RR-set scoring.  5000 is still fast to score
    because we just do dictionary lookups against the RR map.
    """
    scores = []
    for u, neighbors in adj.items():
        if u not in hop_dist:
            continue
        for (v, p) in neighbors:
            if v not in hop_dist:
                continue
            scores.append((score_edge_proxy(u, v, p, adj, hop_dist), (u, v)))
    scores.sort(reverse=True)
    return [edge for _, edge in scores[:max_candidates]]


# ---------------------------------------------------------------------------
# CELF-accelerated greedy edge blocker
# ---------------------------------------------------------------------------

def ris_blocking(adj, nodes, seeds, k, out_path, hops=None, simulations=50):
    """
    Select k edges to block using RIS + CELF.

    FIX SUMMARY vs previous implementations
    ----------------------------------------
    1. Seeded RNG — random.Random(42) so results are reproducible AND
       different from run to run if you change the seed.  Use a time-based
       seed (random.Random()) for true randomness across runs; set an int
       for deterministic debugging.

    2. hop_dist always computed — previously skipped when hops=None, leaving
       the unlimited case with no zone filter and no hop-mirror constraint.

    3. Edge scoring via RR edge-sets — previous code scored v's presence in
       the node-set; now we build edge_rr_map keyed on (u,v) tuples so the
       CELF marginal gain correctly counts RR sets whose path uses that edge.

    4. CELF over a single large RR pool — we build one large pool upfront
       and use the lazy-evaluation trick (re-evaluate only when the stored
       gain is stale).  This gives the same greedy guarantee as per-round
       rebuilding but is O(k·log(E)) instead of O(k·|E|·n_rr).

    5. n_rr scaled to max(zone_size×50, simulations×100, 2000) — gives
       tight coverage estimates even for small zones (dataset2 hop=3).

    6. max_attempt_factor=20 — fills the RR quota on sparse graphs.

    7. Candidate cap raised to 5000 — reduces proxy-score pruning error.
    """
    # FIX 1: use a seeded RNG so output is reproducible but can be varied
    rng = random.Random(42)

    rev_adj = build_reverse_adj(adj)

    # FIX 2: always compute hop_dist (unlimited BFS when hops is None)
    hop_dist = compute_hop_reachable(adj, seeds, hops)

    zone_size = len(hop_dist)
    # FIX 5: larger n_rr for better accuracy
    n_rr = max(zone_size * 50, simulations * 100, 2000)

    print(f"[RIS] Hop limit     : {hops if hops is not None else 'unlimited'}")
    print(f"[RIS] Hop zone size : {zone_size} / {len(nodes)} nodes")
    print(f"[RIS] n_rr target   : {n_rr}")

    # FIX 4 + 6: build one large RR pool with generous attempt budget
    rr_sets = build_rr_sets(
        rev_adj, seeds, hop_dist, blocked=set(),
        rng=rng, n_rr=n_rr, max_attempt_factor=20
    )
    print(f"[RIS] RR sets built : {len(rr_sets)}")

    if not rr_sets:
        print("[RIS] No informative RR sets — nothing to block.")
        return []

    # FIX 3: build edge → set-of-RR-indices map (edge-level, not node-level)
    edge_rr_map = defaultdict(set)
    for idx, edge_set in enumerate(rr_sets):
        for e in edge_set:
            edge_rr_map[e].add(idx)

    # FIX 7: raised candidate cap
    candidates = get_candidate_edges(adj, hop_dist, max_candidates=5000)
    print(f"[RIS] Candidates    : {len(candidates)} edges")

    # ----- CELF initialisation -----
    # Priority queue entries: (-gain, round_last_evaluated, edge)
    pq = []
    for e in candidates:
        gain = len(edge_rr_map.get(e, set()))
        heapq.heappush(pq, (-gain, 0, e))

    selected   = []
    covered    = set()          # indices of RR sets already "hit"
    open(out_path, 'w').close() # clear output file

    while len(selected) < k and pq:
        neg_gain, last_round, edge = heapq.heappop(pq)

        if last_round == len(selected):
            # Gain is up-to-date → select this edge
            selected.append(edge)
            newly_covered = edge_rr_map.get(edge, set()) - covered
            covered |= newly_covered

            with open(out_path, 'a') as f:
                f.write(f"{edge[0]} {edge[1]}\n")

            marginal = len(newly_covered)
            pct      = 100.0 * len(covered) / len(rr_sets)
            print(f"[RIS] Round {len(selected):3d}/{k}: blocking {edge}, "
                  f"marginal gain {marginal}, coverage {len(covered)}/{len(rr_sets)} ({pct:.1f}%)")
        else:
            # Re-evaluate marginal gain given what's already covered
            fresh_gain = len(edge_rr_map.get(edge, set()) - covered)
            heapq.heappush(pq, (-fresh_gain, len(selected), edge))

    return selected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    graph_path  = sys.argv[1]
    seed_path   = sys.argv[2]
    out_path    = sys.argv[3]
    k           = int(sys.argv[4])
    simulations = int(sys.argv[5])
    hop_arg     = sys.argv[6]

    hops = None if hop_arg == "-1" else int(hop_arg)

    nodes, adj, _ = load_graph(graph_path)
    seeds         = load_seeds(seed_path)

    ris_blocking(adj, nodes, seeds, k, out_path, hops, simulations)


if __name__ == "__main__":
    main()