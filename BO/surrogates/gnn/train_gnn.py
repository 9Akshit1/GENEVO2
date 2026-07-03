#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNN surrogate training script.

Trains a TorchMPNN on biosensor simulation data and saves weights for
NumpyMPNN inference. Requires: pip install torch

Training data: the same CSV format as surrogate v4 (master_index.csv).
The GNN should achieve comparable or better rank-rho than the GBM surrogate
(v4 rank-rho = 0.517) because it explicitly captures cross-channel interactions.

Usage:
    python BO/surrogates/gnn/train_gnn.py --data BO/data/data_v10/master_index.csv
    python BO/surrogates/gnn/train_gnn.py --data BO/data/data_v10/master_index.csv --epochs 50 --batch-size 64
"""

import sys
import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path

_GNN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_GNN_DIR))))  # project root
sys.path.insert(0, _GNN_DIR)  # allow absolute imports from gnn package dir when run as script

logger = logging.getLogger(__name__)


def train_gnn(
    data_path: str,
    out_dir: str = "BO/surrogates/gnn/weights",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    seed: int = 42,
):
    """
    Full training pipeline.

    Args:
        data_path: path to master_index.csv
        out_dir:   output directory for weights
        epochs:    training epochs
        batch_size: mini-batch size (graphs)
        lr:        learning rate
        val_frac:  validation fraction
        seed:      random seed
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("[FAIL] PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        from .graph_biosensor import dataset_from_csv, BiosensorGraph
        from .gnn_surrogate import build_torch_mpnn, NumpyMPNN
    except ImportError:
        from graph_biosensor import dataset_from_csv, BiosensorGraph
        from gnn_surrogate import build_torch_mpnn, NumpyMPNN

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # --- Load data ---
    logger.info(f"Loading graphs from {data_path} ...")
    graphs = dataset_from_csv(data_path)
    logger.info(f"Loaded {len(graphs)} graphs")

    # Filter to graphs with targets
    graphs = [g for g in graphs if g.targets is not None]
    logger.info(f"  {len(graphs)} graphs with targets")

    # Train/val split
    idx = rng.permutation(len(graphs))
    n_val = max(1, int(len(graphs) * val_frac))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    logger.info(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    class GraphDataset(Dataset):
        def __init__(self, graphs):
            self.graphs = graphs
        def __len__(self):
            return len(self.graphs)
        def __getitem__(self, i):
            return self.graphs[i]

    def collate_fn(batch):
        return batch   # keep as list — graphs have different node/edge counts in general

    train_loader = DataLoader(GraphDataset(train_graphs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(GraphDataset(val_graphs),   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Model ---
    model, optimizer = build_torch_mpnn(hidden_dim=64, n_layers=3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss: MSE on [DR, FNR, TTD_norm] with per-output weights
    OUTPUT_WEIGHTS = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)  # DR most important

    def compute_loss(y_pred, y_true):
        diff = (y_pred - y_true) ** 2
        return (diff * OUTPUT_WEIGHTS).sum()

    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch = 0
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses   = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            for graph in batch:
                if graph.targets is None:
                    continue
                y_pred = model(graph)
                y_true = torch.tensor(graph.targets, dtype=torch.float32)
                loss   = compute_loss(y_pred, y_true)
                batch_loss += loss
            if n_batches > 0 or len(batch) > 0:
                (batch_loss / max(len(batch), 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(batch_loss.item())
                n_batches  += len(batch)

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                for graph in batch:
                    if graph.targets is None:
                        continue
                    y_pred = model(graph)
                    y_true = torch.tensor(graph.targets, dtype=torch.float32)
                    val_loss += float(compute_loss(y_pred, y_true).item())
                    n_val_samples += 1

        avg_val = val_loss / max(n_val_samples, 1)
        val_losses.append(avg_val)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_path = out_path / "gnn_best.pt"
            torch.save(model.state_dict(), str(best_path))

    logger.info(f"Training done. Best val loss={best_val_loss:.4f} at epoch {best_epoch}")

    # --- Validation metrics ---
    model.load_state_dict(torch.load(str(out_path / "gnn_best.pt"), weights_only=True))
    model.eval()

    # Export weights to .npz format for NumpyMPNN inference
    numpy_weights_path = out_path / "gnn_best.npz"
    model.save_numpy(str(numpy_weights_path))
    logger.info(f"[OK] Exported numpy weights: {numpy_weights_path}")

    all_dr_pred, all_dr_true = [], []
    with torch.no_grad():
        for g in val_graphs:
            if g.targets is None:
                continue
            y_pred_np = model(g).detach().numpy()
            y_true = g.targets
            # DR is first output; apply sigmoid to get probability
            dr_pred = float(1.0 / (1.0 + np.exp(-np.clip(y_pred_np[0], -20, 20))))
            dr_true = float(y_true[0])
            all_dr_pred.append(dr_pred)
            all_dr_true.append(dr_true)

    if len(all_dr_pred) > 10:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(all_dr_true, all_dr_pred)
        logger.info(f"Validation DR rank-rho = {rho:.3f} (p={pval:.4f})")
        logger.info(f"  Comparison: v4 GBM rank-rho = 0.517")
    else:
        rho = float("nan")
        pval = float("nan")

    # Save training history
    history = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "val_dr_rank_rho": float(rho) if not np.isnan(rho) else None,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "numpy_weights": str(numpy_weights_path),
    }
    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"[OK] GNN training complete. Weights: {out_path / 'gnn_best.pt'}")
    logger.info(f"     Numpy weights: {numpy_weights_path}")
    return str(numpy_weights_path), history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN surrogate")
    parser.add_argument("--data", required=True, help="Path to master_index.csv")
    parser.add_argument("--out-dir", default="BO/surrogates/gnn/weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_gnn(
        data_path=args.data,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
