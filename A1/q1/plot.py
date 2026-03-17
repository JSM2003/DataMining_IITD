import sys
import matplotlib.pyplot as plt

outdir = sys.argv[1]
supports = [5, 10, 25, 50, 90]

ap_times = []
fp_times = []

def read_time(path):
    with open(path) as f:
        val = f.read().strip()
        if val == "TIMEOUT":
            return None
        return float(val)

for s in supports:
    ap_times.append(read_time(f"{outdir}/ap{s}/time.txt"))
    fp_times.append(read_time(f"{outdir}/fp{s}/time.txt"))

# Plot Apriori (skip TIMEOUT points)
ap_x = [s for s, t in zip(supports, ap_times) if t is not None]
ap_y = [t for t in ap_times if t is not None]

plt.figure()
if ap_x:
    plt.plot(ap_x, ap_y, marker='o', label="Apriori")

plt.plot(supports, fp_times, marker='o', label="FP-Growth")

plt.xlabel("Support Threshold (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Apriori vs FP-Growth Runtime Comparison")
plt.legend()
plt.grid(True)

plt.savefig(f"{outdir}/plot.png")
