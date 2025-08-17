"""
Minimal training loop for link prediction with GraphSAGE.
Assumes you've pre-sampled a labeled dataset with columns: src_user_id, dst_user_id, label.
"""
from __future__ import annotations
import argparse
import json
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from src.data.loaders import load_graph
from src.models.gnn import GraphSageLinkPredictor, build_pyg_data
from src.utils.io import save_torch, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_dataset(df: pd.DataFrame, uid2idx: dict) -> TensorDataset:
    u = torch.tensor(df["src_user_id"].map(uid2idx).values, dtype=torch.long)
    v = torch.tensor(df["dst_user_id"].map(uid2idx).values, dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    pairs = torch.stack([u, v], dim=1)
    return TensorDataset(pairs, y)


def evaluate(model, data_obj, loader, device="cpu"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        emb = model.encoder(data_obj.x.to(device), data_obj.edge_index.to(device))
        for pairs, y in loader:
            s = model.score_pairs(emb, pairs.to(device)).cpu()
            preds.append(s)
            labels.append(y)
    y_true = torch.cat(labels).numpy()
    y_score = torch.sigmoid(torch.cat(preds)).numpy()
    auc = roc_auc_score(y_true, y_score)
    return auc


def main(args):
    # Load base graph (edges only)
    edges, users, maps, edge_index = load_graph(args.edges_csv, args.users_csv)
    num_nodes = len(maps["idx2uid"])

    # Load labeled train/val CSVs
    train_df = pd.read_csv(args.train_pairs_csv)
    val_df = pd.read_csv(args.val_pairs_csv)

    train_ds = make_dataset(train_df, maps["uid2idx"])
    val_ds = make_dataset(val_df, maps["uid2idx"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphSageLinkPredictor(num_nodes, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout).to(device)
    data_obj = build_pyg_data(num_nodes, edge_index).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()

    best_auc, best_state = -1.0, None
    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        emb = model.encoder(data_obj.x, data_obj.edge_index)  # precompute per epoch
        for pairs, y in train_loader:
            opt.zero_grad()
            scores = model.decoder(emb[pairs[:, 0].to(device)], emb[pairs[:, 1].to(device)])
            loss = bce(scores, y.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item() * pairs.size(0)

        val_auc = evaluate(model, data_obj, val_loader, device)
        logger.info(f"Epoch {epoch:03d} | loss={total_loss/len(train_ds):.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping")
                break

    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.out_dir, "model.pt"))
    save_json(maps, os.path.join(args.out_dir, "id_maps.json"))

    # Export final embeddings for serving
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.encoder(data_obj.x, data_obj.edge_index).cpu()
    save_torch(emb, os.path.join(args.out_dir, "node_embeddings.pt"))

    metrics = {"best_val_auc": float(best_auc)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved artifacts to {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--edges_csv", type=str, required=True)
    p.add_argument("--users_csv", type=str, default="")
    p.add_argument("--train_pairs_csv", type=str, required=True)
    p.add_argument("--val_pairs_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    main(p.parse_args())
