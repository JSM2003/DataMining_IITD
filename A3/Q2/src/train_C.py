"""
train_C.py  –  COL761 Assignment 3  –  Dataset C
Task        : Link prediction
Metric      : Hits@50
Architecture: GAT encoder + MLP decoder with hard-negative mining

Strategy:
  1. Graph Attention Network (GATv2) encoder — multi-head attention learns which
     neighbours matter most, with residual connections and LayerNorm per layer.
  2. Score(u,v) = MLP(h_u * h_v || h_u - h_v || h_u + h_v) -- captures richer interactions
  3. Train with Binary Cross-Entropy on pos + hard-negative pairs
  4. Hits@50: fraction of pos edges where pos score > all but <50 of the 500 hard-neg scores
  5. Feature normalisation + cosine-annealing with warm restarts
  6. Hard negative mining from train_neg (supports both flat [M,2] and [M,K,2] shapes)
"""

import argparse
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, to_undirected

sys.path.append(os.path.dirname(__file__))
from load_dataset import load_dataset, COL761LinkDataset


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _true_num_nodes(ds: COL761LinkDataset) -> int:
    """
    Return the true number of nodes as (max node id + 1) rather than the
    count of unique node ids, which avoids index-out-of-bounds errors when
    some node ids are never an endpoint in any split.
    """
    max_id = 0
    for attr in ("train_pos", "valid_pos", "test_pos",
                 "train_neg", "valid_neg", "test_neg"):
        if hasattr(ds, attr):
            t = getattr(ds, attr)
            val = t.max().item()
            if val > max_id:
                max_id = val
    return int(max_id) + 1


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    Multi-layer GATv2 encoder.

    GATv2Conv fixes the expressiveness limitation of the original GAT by
    computing attention as  a · LeakyReLU(W[h_i || h_j])  (dynamic attention)
    rather than the static  a · LeakyReLU(Wh_i || Wh_j)  formulation.

    Design choices:
    - Each layer uses `heads` independent attention heads; their outputs are
      concatenated (concat=True) inside the conv, so the actual output width
      per layer is `head_dim * heads`. The final layer averages the heads
      (concat=False) to produce exactly `hidden` dimensions.
    - Residual connection: a 1×1 projection aligns the input to the output
      width before adding, so residuals work correctly at every layer.
    - LayerNorm after every attention step — more stable than BatchNorm for
      graphs with heavy-tailed degree distributions.
    - `edge_dim=None`; no edge features are available for dataset C.
    """

    def __init__(self, in_channels: int, hidden: int, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert hidden % heads == 0, \
            f"hidden ({hidden}) must be divisible by heads ({heads})"
        self.dropout   = dropout
        self.num_layers = num_layers
        head_dim       = hidden // heads   # per-head dimension

        # Project raw features into `hidden` dimensions once.
        self.input_proj = nn.Linear(in_channels, hidden)

        self.convs = nn.ModuleList()
        self.lns   = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_heads   = 1      if is_last else heads
            concat_heads = False if is_last else True
            out_channels = hidden if is_last else head_dim   # per head

            self.convs.append(
                GATv2Conv(
                    in_channels=hidden,
                    out_channels=out_channels,
                    heads=out_heads,
                    concat=concat_heads,
                    dropout=dropout,
                    add_self_loops=False,   # already added globally
                )
            )
            # Output width after this layer
            layer_out = out_channels * out_heads if concat_heads else out_channels * 1
            self.lns.append(nn.LayerNorm(layer_out))
            # Residual projection: input is always `hidden`; output may differ
            # for intermediate layers (hidden - head_dim * heads = hidden, ok)
            self.res_projs.append(
                nn.Identity() if layer_out == hidden
                else nn.Linear(hidden, layer_out, bias=False)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.input_proj(x))   # ELU pairs well with GAT attention
        for conv, ln, res_proj in zip(self.convs, self.lns, self.res_projs):
            residual = res_proj(x)
            x = conv(x, edge_index)
            x = ln(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return x   # [N, hidden]


class LinkPredictor(nn.Module):
    """
    MLP over concatenated element-wise product, difference, and sum of the
    endpoint embeddings — captures both symmetric and asymmetric signals.
    """

    def __init__(self, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([h_u * h_v, h_u - h_v, h_u + h_v], dim=-1)
        return self.mlp(feat).squeeze(-1)  # [E]


class LinkPredModel(nn.Module):
    def __init__(self, in_channels: int, hidden: int,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder   = GATEncoder(in_channels, hidden, num_layers, heads, dropout)
        self.predictor = LinkPredictor(hidden, dropout)

    def encode(self, x: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        x          : node features [N, F]
        edge_index : graph edges   [2, E]
        edge_pairs : edges to score [M, 2]
        Returns scores [M]
        """
        h = self.encode(x, edge_index)
        u = h[edge_pairs[:, 0]]
        v = h[edge_pairs[:, 1]]
        return self.predictor(u, v)


# -----------------------------------------------------------------------------
# Pickle / torch.save compatibility fix
# -----------------------------------------------------------------------------
# When train_C.py is run directly it becomes __main__.  pickle serialises each
# class as  "<cls.__module__>.<cls.__qualname__>".  If __module__ is "__main__"
# then torch.load() called from predict.py (where __main__ is predict.py) fails
# with "Can't get attribute 'LinkPredModel' on <module '__main__'>".
#
# Fix: (1) overwrite __module__ on every model class to "train_C" so pickle
# writes "train_C.GATEncoder" etc. in the .pt file, and (2) insert the current
# module object into sys.modules["train_C"] so that pickle's unpickling step
# (which does  sys.modules["train_C"].LinkPredModel) resolves correctly even
# when predict.py never explicitly imports train_C.
if __name__ == "__main__":
    sys.modules["train_C"] = sys.modules["__main__"]
    for _cls in (GATEncoder, LinkPredictor, LinkPredModel):
        _cls.__module__ = "train_C"

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
              k: int = 50) -> float:
    """
    Fraction of positives whose score beats all but fewer than k negatives.

    pos_scores : [P]
    neg_scores : [P, K]
    """
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Load dataset ----------------------------------------------------------
    ds: COL761LinkDataset = load_dataset("C", args.data_dir)
    print(ds)

    # Use max-node-id + 1 as num_nodes to avoid index-out-of-bounds
    num_nodes = _true_num_nodes(ds)
    print(f"True num_nodes (max id + 1): {num_nodes}")

    # Node features - L2-normalise for stable dot-product similarity
    x = ds.x
    x = F.normalize(x, p=2, dim=1)
    x = x.to(device)

    # Build undirected training graph with self-loops
    edge_index = to_undirected(ds.train_pos.t(), num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = edge_index.to(device)

    train_pos = ds.train_pos.to(device)   # [M, 2]
    train_neg = ds.train_neg              # may be [M,2] or [M,K,2]

    # Support both flat and 3-D negative tensors
    if train_neg.dim() == 3:
        # [M, K, 2] - keep structured for per-positive sampling
        train_neg_3d   = train_neg.to(device)
        train_neg_flat = None
    else:
        train_neg_3d   = None
        train_neg_flat = train_neg.to(device)     # [M, 2]

    valid_pos = ds.valid_pos.to(device)   # [V, 2]
    valid_neg = ds.valid_neg.to(device)   # [V, 500, 2]

    print(f"train_pos : {train_pos.shape}")
    if train_neg_3d is not None:
        print(f"train_neg : {train_neg_3d.shape}  (structured 3-D)")
    else:
        print(f"train_neg : {train_neg_flat.shape}  (flat)")
    print(f"valid_pos : {valid_pos.shape}")
    print(f"valid_neg : {valid_neg.shape}")
    print(f"Node features : {x.shape}")

    # -- Model -----------------------------------------------------------------
    model = LinkPredModel(
        in_channels=x.shape[1],
        hidden=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # -- Optimiser & scheduler -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    # Cosine annealing with warm restarts - helps escape local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(args.epochs // 4, 50), T_mult=2
    )

    best_hits        = 0.0
    best_model_state = None
    patience_count   = 0

    # -- Training loop ---------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        n_pos = train_pos.shape[0]

        # Sample hard negatives for this epoch
        if train_neg_3d is not None:
            # [M, K, 2] - sample one negative per positive
            K_neg     = train_neg_3d.shape[1]
            neg_k_idx = torch.randint(K_neg, (n_pos,), device=device)
            neg_sample = train_neg_3d[
                torch.arange(n_pos, device=device), neg_k_idx
            ]                                           # [M, 2]
            # Also mix in some random negatives from elsewhere
            extra_idx = torch.randperm(n_pos, device=device)[:n_pos // 4]
            extra_k   = torch.randint(K_neg, (extra_idx.shape[0],),
                                      device=device)
            extra_neg  = train_neg_3d[extra_idx, extra_k]
            neg_sample = torch.cat([neg_sample, extra_neg], dim=0)
        else:
            # Flat negatives - subsample to match positives * ratio
            neg_idx    = torch.randperm(
                train_neg_flat.shape[0], device=device
            )[:n_pos * args.neg_ratio]
            neg_sample = train_neg_flat[neg_idx]

        # Encode once and reuse for both pos and neg scoring
        h = model.encode(x, edge_index)

        pos_scores       = model.predictor(
            h[train_pos[:, 0]], h[train_pos[:, 1]]
        )                                               # [M]
        neg_scores_train = model.predictor(
            h[neg_sample[:, 0]], h[neg_sample[:, 1]]
        )                                               # [M * ratio]

        # Binary cross-entropy with label smoothing
        scores   = torch.cat([pos_scores, neg_scores_train])
        n_neg    = neg_scores_train.shape[0]
        # Label smoothing: 0.95 for pos, 0.05 for neg
        pos_labels = torch.full((n_pos,), 0.95, device=device)
        neg_labels = torch.full((n_neg,), 0.05, device=device)
        labels     = torch.cat([pos_labels, neg_labels])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # -- Train Hits@50 -- reuse this epoch's scores, no extra forward pass
        with torch.no_grad():
            _ps     = pos_scores.detach()
            n_ratio = neg_scores_train.shape[0] // n_pos
            if n_ratio > 0:
                _ns = neg_scores_train.detach()[:n_pos * n_ratio].view(n_pos, n_ratio)
                train_hits = hits_at_k(_ps, _ns, k=50)
            else:
                train_hits = 0.0

        # -- Validation -- only every eval_every epochs -----------------------
        val_hits = math.nan
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                h_val = model.encode(x, edge_index)

                vp_scores = model.predictor(
                    h_val[valid_pos[:, 0]], h_val[valid_pos[:, 1]]
                )                                       # [V]

                V, K, _ = valid_neg.shape
                vn_flat   = valid_neg.view(V * K, 2)
                vn_scores = model.predictor(
                    h_val[vn_flat[:, 0]], h_val[vn_flat[:, 1]]
                ).view(V, K)                            # [V, K]

                val_hits = hits_at_k(vp_scores, vn_scores, k=50)

            if val_hits > best_hits:
                best_hits        = val_hits
                best_model_state = {k: v.clone()
                                    for k, v in model.state_dict().items()}
                patience_count   = 0
            else:
                patience_count += 1

        # -- Print every epoch ------------------------------------------------
        val_str   = f"{val_hits:.4f}" if not math.isnan(val_hits) else "  --  "
        best_flag = "  *" if (not math.isnan(val_hits) and val_hits >= best_hits) else ""
        print(
            f"Epoch {epoch:4d}  loss={loss.item():.4f}  "
            f"train_hits@50={train_hits:.4f}  val_hits@50={val_str}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}{best_flag}"
        )

        if patience_count >= args.patience:
            print(f"Early stopping (patience={args.patience} eval intervals)")
            break

    print(f"\nBest validation Hits@50: {best_hits:.4f}")

    # -- Save ------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    model.cpu()

    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(
        args.model_dir, f"{args.kerberos}_model_C.pt"
    )
    torch.save(model, save_path)
    print(f"Saved model → {save_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train link-prediction model for COL761 A3 dataset C."
    )
    parser.add_argument("--data_dir",   required=True,
                        help="Absolute path to datasets directory (contains C/)")
    parser.add_argument("--model_dir",  required=True,
                        help="Directory where the model will be saved")
    parser.add_argument("--kerberos",   required=True,
                        help="Your Kerberos ID — used to name the saved model")

    # Model hyper-parameters
    parser.add_argument("--hidden",     type=int,   default=512,
                        help="Hidden dimension of the GAT encoder (default 256)")
    parser.add_argument("--num_layers", type=int,   default=3,
                        help="Number of GAT convolution layers (default 3)")
    parser.add_argument("--heads",      type=int,   default=4,
                        help="Number of attention heads per GAT layer (default 4); "
                             "hidden must be divisible by heads")
    parser.add_argument("--dropout",    type=float, default=0.3,
                        help="Dropout probability (default 0.3)")

    # Optimiser hyper-parameters
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Initial learning rate (default 3e-4)")
    parser.add_argument("--wd",         type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4)")
    parser.add_argument("--epochs",     type=int,   default=500,
                        help="Maximum training epochs (default 500)")
    parser.add_argument("--patience",   type=int,   default=30,
                        help="Early-stopping patience in eval intervals (default 30)")
    parser.add_argument("--eval_every", type=int,   default=5,
                        help="Evaluate on validation set every N epochs (default 5)")

    # Negative sampling
    parser.add_argument("--neg_ratio",  type=int,   default=2,
                        help="Negatives per positive for flat train_neg (default 2)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()