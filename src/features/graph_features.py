"""
Graph-topology features for candidate (u,v) pairs.
- Common Neighbors (directed variants approximated via undirected projection)
- Adamic–Adar
- Personalized PageRank (PPR) proximity
"""
from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
from typing import Iterable, Tuple


def build_nx_graph(edges: pd.DataFrame, directed: bool = True) -> nx.Graph:
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges[["src_user_id", "dst_user_id"]].itertuples(index=False, name=None))
    return G


def pairwise_features(
    G: nx.DiGraph,
    pairs: Iterable[Tuple[int, int]],
    ppr_alpha: float = 0.15,
    use_undirected_projection: bool = True,
) -> pd.DataFrame:
    """
    Compute CN/AA over undirected projection (stable signal) and PPR(u->v).
    """
    if use_undirected_projection:
        UG = G.to_undirected(as_view=True)
    else:
        UG = G

    # Precompute degrees to speed up AA
    deg = dict(UG.degree())

    def adamic_adar(u, v):
        cn = set(UG.neighbors(u)).intersection(UG.neighbors(v))
        if not cn:
            return 0.0
        return float(sum(1.0 / np.log1p(deg[w]) for w in cn if deg[w] > 1))

    # Cache PPR per unique src u to avoid recomputation
    pairs = list(pairs)
    unique_src = sorted({u for u, _ in pairs})
    ppr_cache = {}
    for u in unique_src:
        try:
            ppr_cache[u] = nx.pagerank(G, alpha=1 - ppr_alpha, personalization={u: 1.0})
        except Exception:
            ppr_cache[u] = {}

    rows = []
    for u, v in pairs:
        cn = len(set(UG.neighbors(u)).intersection(UG.neighbors(v))) if UG.has_node(u) and UG.has_node(v) else 0
        aa = adamic_adar(u, v) if UG.has_node(u) and UG.has_node(v) else 0.0
        ppr = ppr_cache.get(u, {}).get(v, 0.0)
        rows.append({"src_user_id": u, "dst_user_id": v, "feat_cn": cn, "feat_aa": aa, "feat_ppr": ppr})
    return pd.DataFrame(rows)
