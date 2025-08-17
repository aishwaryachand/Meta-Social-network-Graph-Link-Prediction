"""
Evaluation utilities: AUC, Precision@K, Recall@K on test pairs.
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
from src.utils.io import load_json, load_torch
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_dataset(df: pd.DataFrame, uid2idx: dict) -> TensorDataset:
    u = torch.tensor(df["src_user_id"].map(uid2idx).values, dtype=torch.long)
    v = torch.tensor(df["dst_user_id"].map(uid2idx).values, dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    return TensorDataset(torch.stack([u, v], dim=1), y)


def precision_recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 10) -> tuple[float, float]:
    """
    scores/labels are grouped by src_user_id externally (recommended for accurate P@K).
    For simplicity here, we compute global P@K/R@K by top-k over all pairs.
    """
    k = min(k, scores.numel())
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    topk_labels = labels[topk_idx]
    precision = float(topk_labels.mean().item())
    recall = float(topk_labels.sum().item() / labels.sum().clamp(min=1).item())
    return precision, recall


def main(args):
    # Load id maps and base graph
    maps = load_json(os.path.join(args.artifacts_dir, "id_maps.json"))
    edges, users, _, edge_index = load_graph(args.edges_csv, args.users_csv)
    num_nodes = len(maps["idx2uid"])
    data_obj = build_pyg_data(num_nodes, edge_index)

    # Load model + embeddings (if present)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphSageLinkPredictor(num_nodes).to(device)
    state = torch.load(os.path.join(args.artifacts_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_df = pd.read_csv(args.test_pairs_csv)
    ds = make_dataset(test_df, maps["uid2idx"])
    loader = DataLoader(ds, batch_size=8192, shuffle=False)

    with torch.no_grad():
        emb = model.encoder(data_obj.x.to(device), data_obj.edge_index.to(device))
        preds, labels = [], []
        for pairs, y in loader:
            s = model.score_pairs(emb, pairs.to(device))
            preds.append(s.cpu())
            labels.append(y)
        scores = torch.cat(preds)
        y_true = torch.cat(labels)

    auc = roc_auc_score(y_true.numpy(), torch.sigmoid(scores).numpy())
    p_at_10, r_at_10 = precision_recall_at_k(torch.sigmoid(scores), y_true, k=10)

    metrics = {"auc": float(auc), "precision@10": p_at_10, "recall@10": r_at_10}
    with open(os.path.join(args.artifacts_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Test AUC={auc:.4f} | P@10={p_at_10:.4f} | R@10={r_at_10:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--edges_csv", type=str, required=True)
    p.add_argument("--users_csv", type=str, default="")
    p.add_argument("--test_pairs_csv", type=str, required=True)
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    main(p.parse_args())
