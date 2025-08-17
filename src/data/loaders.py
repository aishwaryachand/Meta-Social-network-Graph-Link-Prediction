"""
Data loaders for edges/users and basic graph tensors.

Expected files:
- edges.csv: src_user_id,dst_user_id,timestamp (ISO or int)
- users.csv: user_id,<optional attrs...>
"""
from __future__ import annotations
import os
import pandas as pd
import torch
from typing import Dict, Tuple


def load_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"src_user_id", "dst_user_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"edges.csv missing columns: {missing}")
    return df


def load_users(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "user_id" not in df.columns:
        raise ValueError("users.csv must contain 'user_id'")
    return df


def build_id_mappings(edges: pd.DataFrame, users: pd.DataFrame | None = None) -> Dict[str, Dict]:
    """
    Build a contiguous id space [0..N-1] for all user ids seen in edges (and users if provided).
    Returns dict with:
      - "uid2idx": {user_id: idx}
      - "idx2uid": list[idx] -> user_id
    """
    ids = set(edges["src_user_id"]).union(set(edges["dst_user_id"]))
    if users is not None:
        ids |= set(users["user_id"])
    idx2uid = sorted(ids)
    uid2idx = {u: i for i, u in enumerate(idx2uid)}
    return {"uid2idx": uid2idx, "idx2uid": idx2uid}


def to_edge_index(edges: pd.DataFrame, uid2idx: Dict) -> torch.Tensor:
    """
    Convert edges to a PyG-style edge_index (shape [2, E]) using the provided mapping.
    Directed: u->v kept as-is.
    """
    src = edges["src_user_id"].map(uid2idx).astype(int).values
    dst = edges["dst_user_id"].map(uid2idx).astype(int).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def load_graph(
    edges_csv: str,
    users_csv: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, Dict, torch.Tensor]:
    edges = load_edges(edges_csv)
    users = load_users(users_csv) if users_csv and os.path.exists(users_csv) else None
    maps = build_id_mappings(edges, users)
    edge_index = to_edge_index(edges, maps["uid2idx"])
    return edges, users, maps, edge_index
