import random
import sys

def generate_transactions(items, num_transactions, seed=42):
    random.seed(seed)
    num_items = len(items)
    transactions = []

    for _ in range(num_transactions):
        t = set()
        r = random.random()

        # -------- Dense transactions (Apriori slowdown) --------
        if r < 0.65:
            max_item_prob = 0.92
            k = random.randint(30, 38)

            for item in items:
                if random.random() < max_item_prob:
                    t.add(item)

            # enforce density safely
            needed = k - len(t)
            if needed > 0:
                remaining = [i for i in items if i not in t]
                needed = min(needed, len(remaining))
                t |= set(random.sample(remaining, needed))

        # -------- Medium transactions --------
        elif r < 0.85:
            k = random.randint(15, 22)
            t |= set(random.sample(items, k))

        # -------- Sparse transactions (kills 90%) --------
        else:
            k = random.randint(1, 4)
            t |= set(random.sample(items, k))

        transactions.append(tuple(sorted(t)))

    return transactions


if __name__ == "__main__":
    """
    Usage:
    python generate_dataset.py <num_items> <num_transactions>
    Example:
    python generate_dataset.py 45 15000
    """

    num_items = int(sys.argv[1])
    num_transactions = int(sys.argv[2])

    if num_items > 50:
        raise ValueError("Universal itemset must be ≤ 50")

    items = list(range(1, num_items + 1))
    transactions = generate_transactions(items, num_transactions)

    with open("dataset.dat", "w") as f:
        for t in transactions:
            f.write(" ".join(map(str, t)) + "\n")

    print(f"Generated {len(transactions)} transactions with {num_items} items.")
