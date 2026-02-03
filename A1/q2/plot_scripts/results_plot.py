import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_results(csv_path):
    """Reads the CSV file using file.readline() and generates the threshold vs runtime plot."""

    # Data structure:
    # { algorithm_name : [(threshold, exec_time), ...] }
    data = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")

        # Identify column indices
        algo_idx = header.index("algorithm")
        thresh_idx = header.index("threshold")
        time_idx = header.index("exec_time")

        # Read file line by line
        while True:
            line = file.readline()
            if not line:
                break

            parts = line.strip().split(",")
            algorithm = parts[algo_idx]
            threshold = float(parts[thresh_idx])
            exec_time = float(parts[time_idx])

            data[algorithm].append((threshold, exec_time))

    # Sort each algorithm's data by threshold
    for algo in data:
        data[algo].sort(key=lambda x: x[0])

    plt.figure(figsize=(12, 6))

    max_exec_time = 0

    # Plot each algorithm
    for algo, values in data.items():
        thresholds = [v[0] for v in values]
        exec_times = [v[1] for v in values]
        max_exec_time = max(max_exec_time, max(exec_times))

        plt.plot(
            thresholds,
            exec_times,
            marker='o',
            linewidth=2,
            label=algo
        )

    plt.xlabel("Support Threshold (%)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Comparison: gSpan vs FSG vs Gaston")

    # Y-axis tick interval = 1000
    plt.yticks(range(0, int(max_exec_time) + 1000, 1000))

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_image = "threshold_vs_runtime.png"
    plt.savefig(output_image, dpi=300)
    print(f"\nPlot saved as: {output_image}\n")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path_to_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    plot_results(csv_path)
