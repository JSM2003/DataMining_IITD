import argparse

def convert_to_fsg(input_path, output_path):
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    out = []
    i = 0
    graph_count = 0

    while i < len(lines):

        # Expect Graph ID line (like "#42614805")
        if not lines[i].startswith("#"):
            raise ValueError(f"Expected graph ID at line {i}, got: {lines[i]}")

        graph_id = lines[i][1:]  # keep original number/string
        i += 1

        # Number of nodes
        num_nodes = int(lines[i]); i += 1

        # Node labels (KEEP AS-IS, NO ENCODING)
        node_labels = []
        for vid in range(num_nodes):
            label = lines[i]
            node_labels.append((vid, label))
            i += 1

        # Number of edges
        num_edges = int(lines[i]); i += 1

        edges = []
        for _ in range(num_edges):
            parts = lines[i].split()
            src = int(parts[0])
            dst = int(parts[1])
            label = parts[2]  # keep original label (string)
            if src > dst:
                src, dst = dst, src  # ensure ascending order
            edges.append((src, dst, label))
            i += 1

        # Sort vertex lines by ID
        node_labels.sort(key=lambda x: x[0])

        # Sort edges by (src, dst)
        edges.sort(key=lambda x: (x[0], x[1]))

        # ---------- Build FSG format ----------
        out.append(f"t # {graph_id}")

        for vid, lab in node_labels:
            out.append(f"v {vid} {lab}")

        for (src, dst, lab) in edges:
            out.append(f"u {src} {dst} {lab}")

        graph_count += 1

    # Write output file
    with open(output_path, "w") as f:
        f.write("\n".join(out))

    print(f"✔ FSG format conversion complete: {graph_count} graphs → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input file path")
    parser.add_argument("output_path", help="Output file path")
    args = parser.parse_args()

    convert_to_fsg(args.input_path, args.output_path)

# convert_to_fsg("/home/santhana/Downloads/sample_graph.txt_graph", "output_fsg.txt")