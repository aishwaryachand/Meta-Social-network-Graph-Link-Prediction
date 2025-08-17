"""
FastAPI inference service:
- /health
- /recommendations/{user_id}?k=20
- /score (batch pair scoring)

It uses pretrained node embeddings (artifacts/node_embeddings.pt) and id_maps.
Recommendation strategy:
1) Compute cosine similarity to all nodes.
2) Exclude self and already-followed accounts (from edges.csv).
3) Return top-k.
"""
from __future__ import annotations
import os
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from src.utils.io import load_json, load_torch
from src.data.loaders import load_edges

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
EDGES_CSV = os.environ.get("EDGES_CSV", "data/edges.csv")

app = FastAPI(title="Link Prediction Service", version="0.1.0")

# Load artifacts
id_maps = load_json(os.path.join(ART_DIR, "id_maps.json"))
emb = load_torch(os.path.join(ART_DIR, "node_embeddings.pt"))  # [N, D]
uid2idx = {k: int(v) for k, v in id_maps["uid2idx"].items()}
idx2uid = [int(u) for u in id_maps["idx2uid"]]

edges_df = load_edges(EDGES_CSV)
followed = edges_df.groupby("src_user_id")["dst_user_id"].apply(set).to_dict()


class ScoreRequest(BaseModel):
    pairs: List[Tuple[int, int]]  # [(u, v), ...]


@app.get("/health")
def health():
    return {"status": "ok", "num_nodes": emb.shape[0], "dim": emb.shape[1]}


@app.get("/recommendations/{user_id}")
def recommend(user_id: int, k: int = 20):
    if user_id not in uid2idx:
        raise HTTPException(status_code=404, detail="user_id not found")
    uidx = uid2idx[user_id]
    uvec = emb[uidx:uidx + 1]  # [1, D]
    sims = F.cosine_similarity(uvec, emb)  # [N]
    sims[uidx] = -1e9  # exclude self
    # Exclude already-followed
    existing = followed.get(user_id, set())
    if existing:
        idx_existing = [uid2idx[v] for v in existing if v in uid2idx]
        sims[idx_existing] = -1e9
    topk = torch.topk(sims, k=min(k, sims.numel()), largest=True)
    recs = [{"user_id": int(idx2uid[i]), "score": float(sims[i].item())} for i in topk.indices.tolist()]
    return {"user_id": user_id, "recommendations": recs}


@app.post("/score")
def score(req: ScoreRequest):
    pairs = req.pairs
    uidx = torch.tensor([uid2idx[u] for u, _ in pairs], dtype=torch.long)
    vidx = torch.tensor([uid2idx[v] for _, v in pairs], dtype=torch.long)
    s = (emb[uidx] * emb[vidx]).sum(dim=-1)  # dot product
    return {"scores": [float(x) for x in s.tolist()]}
