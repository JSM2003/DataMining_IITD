import sys
import argparse


def convert_to_gspan(input_path, output_path):
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    out = []
    i = 0
    graph_count = 0

    # Node label encoding map
    node_label_map = {}
    next_node_label_id = 0

    def encode_node(label):
        nonlocal next_node_label_id
        if label not in node_label_map:
            node_label_map[label] = next_node_label_id
            next_node_label_id += 1
        return node_label_map[label]

    while i < len(lines):

        # Expect a graphID line like "#42614805"
        if not lines[i].startswith("#"):
            raise ValueError(f"Expected graph ID at line {i}, got: {lines[i]}")
        i += 1  # skip transaction ID completely

        # Number of nodes
        num_nodes = int(lines[i])
        i += 1

        # Node labels → encode to integers
        node_labels = []
        for _ in range(num_nodes):
            encoded = encode_node(lines[i])
            node_labels.append(encoded)
            i += 1

        # Number of edges
        num_edges = int(lines[i])
        i += 1

        # Edge list; labels kept as-is (integers)
        edges = []
        for _ in range(num_edges):
            parts = lines[i].split()  # no need for replace(",", " ")
            src = int(parts[0])
            dst = int(parts[1])
            label = int(parts[2])   # edge label stays integer
            edges.append((src, dst, label))
            i += 1

        # ------- Build gSpan Output --------
        out.append(f"t # {graph_count}")

        # Write vertices
        for vid, lab in enumerate(node_labels):
            out.append(f"v {vid} {lab}")

        # Write edges
        for (src, dst, lab) in edges:
            out.append(f"e {src} {dst} {lab}")

        graph_count += 1

    # Write gSpan formatted output
    with open(output_path, "w") as f:
        f.write("\n".join(out))

    print(f"✔ GLibs format conversion complete: {graph_count} graphs → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input file path")
    parser.add_argument("output_path", help="Output file path")
    args = parser.parse_args()

    convert_to_gspan(args.input_path, args.output_path)

# convert_to_gspan("/home/santhana/Downloads/sample_graph.txt_graph", "output_gspan.txt")
