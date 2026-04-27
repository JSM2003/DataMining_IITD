"""
train_B.py  –  COL761 Assignment 3
Binary node classification on Dataset B (ROC-AUC metric)

Architecture: GraphSAGE with neighbor sampling for scalability on ~2.9M nodes.
Uses class-weighted BCE loss to handle potential class imbalance.

Usage:
    python train_B.py \
        --data_dir /absolute/path/to/datasets \
        --model_dir /path/to/models \
        --kerberos YOUR_KERBEROS \
        [--epochs 30] [--hidden 256] [--layers 3] [--lr 1e-3] [--batch_size 2048]
"""

import argparse
import importlib
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from load_dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGE_B(nn.Module):
    """
    Multi-layer GraphSAGE for binary node classification.
    Outputs a single logit per node (compatible with predict_B which handles sigmoid).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input → first hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Last hidden → output
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classifier head: outputs [N, 2] so predict_B can use softmax[:, 1]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)  # [N, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced binary classification."""
    counts = torch.bincount(labels, minlength=2).float()
    total  = counts.sum()
    weights = total / (2.0 * counts + 1e-8)
    print(f"  Class counts : {counts.tolist()}")
    print(f"  Class weights: {weights.tolist()}")
    return weights


def make_balanced_train_idx(train_idx: torch.Tensor, train_labels: torch.Tensor) -> torch.Tensor:
    """
    Oversample the minority class indices so the returned index array
    has equal representation of both classes. This is passed directly
    to NeighborLoader as input_nodes — no custom sampler needed,
    so it works with all PyG versions.
    """
    idx_0 = train_idx[train_labels == 0]
    idx_1 = train_idx[train_labels == 1]

    # make minority match majority size by repeating with some randomness
    n_majority = max(len(idx_0), len(idx_1))
    if len(idx_0) < len(idx_1):
        repeat = (n_majority // len(idx_0)) + 1
        idx_0  = idx_0.repeat(repeat)[:n_majority]
    else:
        repeat = (n_majority // len(idx_1)) + 1
        idx_1  = idx_1.repeat(repeat)[:n_majority]

    balanced = torch.cat([idx_0, idx_1])
    # shuffle
    perm = torch.randperm(len(balanced))
    print(f"  Balanced train idx: {len(balanced):,} ({n_majority:,} per class)")
    return balanced[perm]


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Compute ROC-AUC on the provided NeighborLoader (val set)."""
    model.eval()
    all_scores = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)       # [N_sub, 2]
        probs  = torch.softmax(logits, dim=1)[:, 1]    # [N_sub]

        # Use full_y to get correct labels for seed nodes only
        n           = batch.batch_size
        seed_labels = batch.full_y[:n]                 # [n]  from scattered full_y
        mask        = seed_labels >= 0                 # only labeled seeds

        all_scores.append(probs[:n][mask].cpu())
        all_labels.append(seed_labels[mask].cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    return roc_auc_score(labels, scores)


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification — down-weights easy majority-class
    examples so the model focuses on hard minority-class examples.
    gamma=2 is standard; alpha balances class weights.
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # [2] class weights
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading dataset B ...")
    t0 = time.time()
    ds   = load_dataset("B", args.data_dir)
    data = ds[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Nodes     : {data.num_nodes:,}")
    print(f"  Edges     : {data.num_edges:,}")
    print(f"  Features  : {data.x.shape[1]}")
    print(f"  Labeled   : {data.labeled_nodes.shape[0]:,}")
    print(f"  Train     : {data.train_mask.sum().item():,}")
    print(f"  Val       : {data.val_mask.sum().item():,}")

    # ── Build index arrays for NeighborLoader ──────────────────────────────
    labeled     = data.labeled_nodes          # [L]
    train_idx   = labeled[data.train_mask]    # absolute node indices
    val_idx     = labeled[data.val_mask]

    # Labels are indexed over labeled_nodes only; scatter to full node space
    full_y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    full_y[labeled] = data.y.long()
    data.full_y = full_y

    # ── Class weights + balanced sampling ────────────────────────────────
    train_labels  = data.y[data.train_mask].long()
    class_weights = compute_class_weights(train_labels).to(device)
    balanced_idx  = make_balanced_train_idx(train_idx, train_labels)

    # ── Data loaders ───────────────────────────────────────────────────────
    fan_out = [15, 10, 5][:args.layers]  # neighbors per layer
    train_loader = NeighborLoader(
        data,
        num_neighbors=fan_out,
        input_nodes=balanced_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,           # 0 avoids nohup/background process hangs
        pin_memory=True,         # faster CPU→GPU transfer to compensate
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=fan_out,
        input_nodes=val_idx,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = GraphSAGE_B(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # ── Register module once before training loop (mirrors train_A.py pattern) ──
    spec = importlib.util.spec_from_file_location("train_B", __file__)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["train_B"] = mod
    spec.loader.exec_module(mod)

    # ── Training loop ──────────────────────────────────────────────────────
    best_auc        = 0.0
    patience_count  = 0
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_B.pt")

    print(f"\n{'Epoch':>6}  {'Loss':>10}  {'Val AUC':>10}  {'Time':>8}  {'Patience':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t_ep = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index)   # [batch_size_full, 2]
            n      = batch.batch_size
            # Only compute loss on seed nodes that are labeled
            seed_logits = logits[:n]                    # [n, 2]
            seed_labels = batch.full_y[:n]              # [n]

            # Filter out unlabeled (-1) nodes (shouldn't happen in train_idx but safe)
            mask = seed_labels >= 0
            if mask.sum() == 0:
                continue

            loss = criterion(seed_logits[mask], seed_labels[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_auc = evaluate(model, val_loader, device)
        elapsed = time.time() - t_ep

        if val_auc > best_auc:
            best_auc       = val_auc
            patience_count = 0
            # Rebind class to the registered module so pickle stores it as
            # "train_B_final.GraphSAGE_B" — same pattern as train_A_final
            # in train_A.py, which predict.py can import successfully.
            model.__class__ = mod.GraphSAGE_B
            torch.save(model, model_path)
            print(f"{epoch:>6}  {avg_loss:>10.4f}  {val_auc:>10.4f}  {elapsed:>7.1f}s  ↑ best saved")
        else:
            patience_count += 1
            print(f"{epoch:>6}  {avg_loss:>10.4f}  {val_auc:>10.4f}  {elapsed:>7.1f}s  {patience_count:>4}/{args.patience}")
            if patience_count >= args.patience:
                print(f"\n  Early stopping triggered at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    total_time = time.time() - t0
    print(f"\nTraining complete. Best Val AUC : {best_auc:.4f}")
    print(f"Total training time: {total_time/60:.1f} mins ({total_time:.0f}s)")
    print(f"Model saved to     : {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE for Dataset B")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--kerberos",   required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--hidden",     type=int,   default=256)
    parser.add_argument("--layers",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--wd",         type=float, default=1e-5)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping: stop after this many epochs with no AUC improvement")
    args = parser.parse_args()

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    train(args)


if __name__ == "__main__":
    main()
