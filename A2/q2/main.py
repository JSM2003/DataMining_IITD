import random
import heapq
import sys
from collections import defaultdict

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
                print(f"[WARN] graph line {lineno} has fewer than 3 fields — skipping: {line!r}")
                continue
            try:
                u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                print(f"[WARN] graph line {lineno} could not be parsed — skipping: {line!r}")
                continue
            if not (0.0 < p <= 1.0):
                print(f"[WARN] graph line {lineno}: p={p} outside (0,1] for edge ({u},{v})")
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


def simulate_once(adj, source_set, blocked, rng, hops=None):
    """
    One Independent-Cascade realisation on the graph with `blocked` edges removed.

    Parameters
    ----------
    adj        : adjacency list
    source_set : initially burning nodes A0
    blocked    : set of (u, v) edges that are blocked
    rng        : seeded random instance
    hops       : if provided, fire spread is limited to this many hops from
                 the seed set. Nodes beyond `hops` steps away from any seed
                 cannot be ignited. None means unlimited spread.

    Returns |A_∞| — total number of burned nodes at termination.
    """
    burned = set(source_set)
    # Each frontier entry is (node, current_hop_depth) so we can enforce
    # the hop limit. Seeds start at depth 0.
    frontier = [(node, 0) for node in source_set]

    while frontier:
        next_frontier = []
        for u, depth in frontier:
            # If hops is set and we've reached the limit, this node cannot
            # ignite any further neighbours — skip without stopping the BFS,
            # since other nodes at shallower depths may still propagate.
            if hops is not None and depth >= hops:
                continue
            for (v, p) in adj.get(u, ()):
                if (u, v) not in blocked and v not in burned:
                    if rng.random() < p:
                        burned.add(v)
                        next_frontier.append((v, depth + 1))
        frontier = next_frontier

    return len(burned)



def greedy_route_blocking(adj, seeds, k, blocks_out_path, hops=None, simulations=10):

    blocked = set()
    base_spread = 0
    
    for _ in range(simulations):
        base_spread += simulate_once(adj, seeds, blocked, random, hops)
    
    base_spread /= simulations

    pq = []
    gains = {}

    for u in adj:
        for (v,p) in adj[u]:

            test_block = blocked | {(u, v)}
            spread = 0

            for _ in range(simulations):
                spread += simulate_once(adj, seeds, test_block, random, hops)
            
            spread /= simulations
            gain = base_spread - spread
            gains[(u, v)] = gain

            heapq.heappush(pq, (-gain, (u, v), 0))

    selected = []
    iteration = 0

    while len(selected) < k:

        gain, edge, last_iter = heapq.heappop(pq)
        gain = -gain

        if last_iter == iteration:
            blocked.add(edge)
            selected.append(edge)

            with open(blocks_out_path, "a+") as file:
                line = " ".join(list(map(str,edge)))
                file.write(line+'\n')
            
            iteration += 1

        else:

            test_block = blocked | {edge}

            spread = 0
            for _ in range(simulations):
                spread += simulate_once(adj, seeds, test_block, random, hops)

            spread /= simulations
            new_gain = base_spread - spread
            heapq.heappush(pq, (-new_gain, edge, iteration))

    #return selected

def main():
    graph_path = sys.argv[1]
    seed_path = sys.argv[2]
    blocks_out_path = sys.argv[3]
    k_val = int(sys.argv[4])
    n_random_inst_val = int(sys.argv[5])
    hop_arg = sys.argv[6]

    # Normalise: -1 sentinel → None (unlimited)
    hops     = None if hop_arg == "-1" else int(hop_arg)

    nodes, adj, edge_set = load_graph(graph_path)
    source_set           = load_seeds(seed_path)

    #print(adj)
    #exit(-1)

    greedy_route_blocking(adj,source_set,k_val,blocks_out_path,hops,n_random_inst_val)
    
if __name__ == "__main__":
    main()