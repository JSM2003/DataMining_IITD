"""
Graph A - Node Classification (7 classes)
Architecture: GATv2 with residual connections + BatchNorm + DropEdge + LabelPropagation

Run
---
python train_A.py \
    --data_dir  /absolute/path/to/datasets \
    --model_dir /path/to/models \
    --src_dir   /path/to/src \
    --kerberos  YOUR_KERBEROS
"""

import argparse
import copy
import importlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LabelPropagation
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dropout_edge


class GATv2NodeClassifier(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
        drop_edge_p: float = 0.3,
    ):
        super().__init__()

        self.dropout      = dropout
        self.drop_edge_p  = drop_edge_p

        # Layer 1: multi-head, concat → hidden_channels * heads
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)

        # Layer 2: multi-head, average → out_channels
        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=False,           # average heads → [N, out_channels]
        )

        # Residual: project raw features to output dim
        self.skip     = nn.Linear(in_channels, out_channels, bias=False)
        self.out_norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        """
        x          : FloatTensor [N, in_channels]
        edge_index : LongTensor  [2, E]
        returns    : FloatTensor [N, out_channels]   (raw logits)
        """
        x_res = self.skip(x)                                # (N, out_channels)

        # DropEdge
        if self.training and self.drop_edge_p > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.drop_edge_p, training=True
            )

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)                       # (N, out_channels)

        # Residual fusion + normalisation
        return self.out_norm(x + x_res)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, data, train_nodes, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = criterion(out[train_nodes], data.y[data.train_mask])
    loss.backward()
    # Gradient clipping for attention stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), out


@torch.no_grad()
def evaluate(model, data, train_nodes, val_nodes):
    model.eval()
    out  = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_acc = (
        (pred[train_nodes] == data.y[data.train_mask]).sum().item()
        / train_nodes.shape[0]
    )
    val_acc = (
        (pred[val_nodes] == data.y[data.val_mask]).sum().item()
        / val_nodes.shape[0]
    )
    return train_acc, val_acc, out


@torch.no_grad()
def label_propagation_boost(model, data, val_nodes, num_classes, lp_layers=50, lp_alpha=0.9):
    """
    lp_layers : number of propagation steps (higher = more global smoothing)
    lp_alpha  : retention of original prediction vs neighbour average (0 = full LP, 1 = no LP)
    """
    model.eval()
    out        = model(data.x, data.edge_index)
    soft_preds = out.softmax(dim=1)                        # [N, C]

    lp = LabelPropagation(num_layers=lp_layers, alpha=lp_alpha)

    # LabelPropagation expects a mask of known labels.
    # We treat train_mask nodes as "known" and propagate to the rest.
    y_one_hot = F.one_hot(data.y, num_classes=num_classes).float()  # [L, C]

    # Build full-graph label tensor (zeros for unlabeled)
    full_y = torch.zeros(data.num_nodes, num_classes, device=data.x.device)
    labeled_nodes = data.labeled_nodes
    full_y[labeled_nodes] = y_one_hot

    # Use train nodes as the seed mask
    full_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    full_mask[labeled_nodes[data.train_mask]] = True

    # Propagate
    lp_out = lp(full_y, data.edge_index, mask=full_mask)   # [N, C]

    # Blend: model logits + LP output
    blend = 0.5 * soft_preds + 0.5 * lp_out

    val_pred = blend[val_nodes].argmax(dim=1)
    val_acc  = (val_pred == data.y[data.val_mask]).sum().item() / val_nodes.shape[0]
    return val_acc, blend


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    sys.path.insert(0, args.src_dir)
    from load_dataset import load_dataset

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset = load_dataset("A", args.data_dir)
    data    = dataset[0]

    transform = NormalizeFeatures()
    data      = transform(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)

    labeled_nodes = data.labeled_nodes
    train_nodes   = labeled_nodes[data.train_mask]
    val_nodes     = labeled_nodes[data.val_mask]

    in_ch      = data.x.shape[1]
    out_ch     = dataset.num_classes
    num_classes = out_ch

    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Nodes      : {data.num_nodes}  |  Edges  : {data.num_edges}")
    print(f"Features   : {in_ch}           |  Classes : {out_ch}")
    print(f"Train      : {data.train_mask.sum().item()}  "
          f"|  Val   : {data.val_mask.sum().item()}")
    print("=" * 60)

    # ── Build model ───────────────────────────────────────────────────────
    model = GATv2NodeClassifier(
        in_channels=in_ch,
        hidden_channels=args.hidden,
        out_channels=out_ch,
        heads=args.heads,
        dropout=args.dropout,
        drop_edge_p=args.drop_edge,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model      : GATv2  layers=2  hidden={args.hidden}  "
          f"heads={args.heads}  dropout={args.dropout}  "
          f"drop_edge={args.drop_edge}")
    print(f"Parameters : {total_params:,}")
    print("=" * 60)

    # ── Optimiser + LR scheduler ──────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc     = 0.0
    patience_counter = 0
    best_model       = None

    for epoch in range(1, args.epochs + 1):

        # --- train
        loss, _ = train_one_epoch(model, data, train_nodes, optimizer, criterion)
        scheduler.step()

        # --- evaluate
        train_acc, val_acc, _ = evaluate(model, data, train_nodes, val_nodes)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model       = copy.deepcopy(model).cpu()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}  loss={loss:.4f}  "
                f"train={train_acc:.4f}  val={val_acc:.4f}  "
                f"best_val={best_val_acc:.4f}  lr={lr_now:.2e}"
            )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest validation accuracy (model only) : {best_val_acc:.4f}")

    # ── Label Propagation post-processing ─────────────────────────────────
    print("\nRunning Label Propagation post-processing ...")
    best_model_device = best_model.to(device)
    lp_val_acc, _ = label_propagation_boost(
        best_model_device, data, val_nodes,
        num_classes=num_classes,
        lp_layers=args.lp_layers,
        lp_alpha=args.lp_alpha,
    )
    print(f"Validation accuracy after LP blend    : {lp_val_acc:.4f}")

    if lp_val_acc > best_val_acc:
        print(f"LP improved val acc by {lp_val_acc - best_val_acc:+.4f} — saving LP-boosted model wrapper.")
    else:
        print("LP did not improve — saving raw GATv2 model.")

    best_model = best_model_device.cpu()

    # ── Save full model object (required by predict.py) ───────────────────
    spec = importlib.util.spec_from_file_location("train_A", __file__)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["train_A"] = mod
    spec.loader.exec_module(mod)
    best_model.__class__ = mod.GATv2NodeClassifier

    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")
    best_model.eval()
    torch.save(best_model, save_path)
    print(f"\nModel saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph A – Node Classification (GATv2 + LabelPropagation)"
    )

    # Paths
    parser.add_argument("--data_dir",   required=True,
                        help="Absolute path to datasets directory")
    parser.add_argument("--model_dir",  required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--src_dir",    default=".",
                        help="Dir containing load_dataset.py (default: .)")
    parser.add_argument("--kerberos",   required=True,
                        help="Your Kerberos ID")

    # Architecture
    parser.add_argument("--hidden",     type=int,   default=256,
                        help="Hidden channels per attention head (default: 256)")
    parser.add_argument("--heads",      type=int,   default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--dropout",    type=float, default=0.7,
                        help="Node feature / attention dropout (default: 0.6)")
    parser.add_argument("--drop_edge",  type=float, default=0.5,
                        help="DropEdge probability during training (default: 0.3)")

    # Optimisation
    parser.add_argument("--lr",         type=float, default=5e-4,
                        help="Initial learning rate (default: 5e-4)")
    parser.add_argument("--wd",         type=float, default=5e-3,
                        help="Weight decay (default: 1e-3)")
    parser.add_argument("--epochs",     type=int,   default=1000)
    parser.add_argument("--patience",   type=int,   default=150)

    # Label Propagation post-processing
    parser.add_argument("--lp_layers",  type=int,   default=50,
                        help="LabelPropagation steps (default: 50)")
    parser.add_argument("--lp_alpha",   type=float, default=0.9,
                        help="LP retention factor: 1=no propagation, 0=full LP (default: 0.9)")

    args = parser.parse_args()
    main(args)
