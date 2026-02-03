import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_path):
    """Reads the CSV file and generates the threshold vs runtime plot."""

    # Read CSV
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="threshold")

    plt.figure(figsize=(12, 6))

    # Plot each algorithm line
    for algo in df['algorithm'].unique():
        subset = df[df['algorithm'] == algo]
        plt.plot(
            subset['threshold'],
            subset['exec_time'],
            marker='o',
            linewidth=2,
            label=algo
        )

    plt.xlabel("Support Threshold (%)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Comparison: gSpan vs FSG vs Gaston")

    # Y-axis tick interval = 1000
    max_y = df['exec_time'].max()
    plt.yticks(range(0, max_y + 1000, 1000))

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save image next to the input CSV file
    output_image = "threshold_vs_runtime.png"
    plt.savefig(output_image, dpi=300)
    print(f"\n Plot saved as: {output_image}\n")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path_to_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    plot_results(csv_path)
