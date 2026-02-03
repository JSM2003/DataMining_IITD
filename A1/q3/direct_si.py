import networkx as nx
from networkx.algorithms import isomorphism as iso
from multiprocessing import Pool, cpu_count
import sys

'''
The code to perform subgraph isomorphism tests was done by us, and it was given to ChatGPT (LLM)
to return a code that parallelizes the process by utilizing multiple cpu cores at once

Prompt: <attached file direct_si.py> modify the code to parallelize the subgraph isomorphism
checks

Text Response: Below is a parallelized version of the same program that uses Python 
multiprocessing to speed up subgraph isomorphism checks while preserving correctness 
and output format.

Design choices are conservative and grading-safe.
'''

def load_graphs(path):
    graphs = []
    G = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()

            elif line.startswith("v"):
                _, i, lab = line.split()
                G.add_node(int(i), label=lab)

            elif line.startswith("e"):
                _, u, v, lab = line.split()
                G.add_edge(int(u), int(v), label=lab)

    if G is not None:
        graphs.append(G)

    return graphs


def is_subgraph_worker(args):
    
    q, gi, g = args

    node_match = iso.categorical_node_match("label", None)
    edge_match = iso.categorical_edge_match("label", None)

    matcher = iso.GraphMatcher(g,q,node_match=node_match,edge_match=edge_match)

    if matcher.subgraph_is_isomorphic():
        return gi
    
    else:
        return None


def run_subgraph_search_parallel(db_graphs, query_graphs):
    nproc = cpu_count()

    results = {}

    for qi, q in enumerate(query_graphs, start=1):
        tasks = [(q, gi + 1, g) for gi, g in enumerate(db_graphs)]

        with Pool(processes=nproc) as pool:
            hits = pool.map(is_subgraph_worker, tasks)

        matches = [gi for gi in hits if gi is not None]
        results[qi] = matches

    return results


if __name__ == "__main__":

    db_path = sys.argv[1]
    q_path = sys.argv[2]
    out_path = sys.argv[3]

    db_graphs = load_graphs(db_path)
    query_graphs = load_graphs(q_path)

    results = run_subgraph_search_parallel(db_graphs, query_graphs)

    with open(out_path, "w") as f:
        for qid in sorted(results):
            f.write(f"q # {qid}\n")
            f.write("c # " + " ".join(map(str, results[qid])) + "\n")
