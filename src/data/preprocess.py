"""
Preprocessing utilities: temporal splits and negative sampling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict


def to_timestamp(series: pd.Series) -> pd.Series:
    """Coerce to int timestamp (seconds). Accepts ISO or int-like."""
    try:
        # If already int-ish, return as is
        return pd.to_datetime(series, errors="coerce").astype("int64") // 10**9
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def temporal_split(
    edges: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Temporal split by timestamp cutoffs.
    - train: t <= train_end
    - val:   train_end < t <= val_end
    - test:  t > val_end
    """
    e = edges.copy()
    if not np.issubdtype(e["timestamp"].dtype, np.number):
        e["timestamp"] = to_timestamp(e["timestamp"])
    t_train = pd.to_datetime(train_end).value // 10**9
    t_val = pd.to_datetime(val_end).value // 10**9

    train = e[e["timestamp"] <= t_train]
    val = e[(e["timestamp"] > t_train) & (e["timestamp"] <= t_val)]
    test = e[e["timestamp"] > t_val]
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def negative_sampling(
    edges_pos: pd.DataFrame,
    all_users: np.ndarray,
    negatives_per_pos: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample negatives for each (u,v) by corrupting v: (u, v_neg) where (u, v_neg) is not in positives.
    Returns a DataFrame with columns: src_user_id, dst_user_id, label (1 for pos, 0 for neg).
    """
    rng = np.random.default_rng(seed)
    # Build set of positives per src for fast exclusion
    pos_pairs = set(zip(edges_pos["src_user_id"].tolist(), edges_pos["dst_user_id"].tolist()))
    grouped = edges_pos.groupby("src_user_id")["dst_user_id"].apply(set).to_dict()

    neg_rows = []
    for u, vs in grouped.items():
        num_pos = len(vs)
        num_neg = num_pos * negatives_per_pos
        # rejection sampling
        samples = []
        while len(samples) < num_neg:
            candidates = rng.choice(all_users, size=(num_neg - len(samples)), replace=True)
            for v in candidates:
                if (u, v) not in pos_pairs and u != v:
                    samples.append((u, v))
                    if len(samples) >= num_neg:
                        break
        for v in samples:
            neg_rows.append({"src_user_id": u, "dst_user_id": v[1], "label": 0})

    pos_rows = edges_pos.assign(label=1)
    all_rows = pd.concat([pos_rows, pd.DataFrame(neg_rows)], axis=0, ignore_index=True)
    return all_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
