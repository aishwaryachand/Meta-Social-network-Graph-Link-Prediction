import pandas as pd
import networkx as nx
from src.features.graph_features import build_nx_graph, pairwise_features

def test_cn_aa_ppr_basic():
    edges = pd.DataFrame({"src_user_id": [1, 2], "dst_user_id": [2, 3]})
    G = build_nx_graph(edges)
    pairs = [(1, 3), (2, 1)]
    df = pairwise_features(G, pairs)
    assert "feat_cn" in df.columns
    assert "feat_aa" in df.columns
    assert "feat_ppr" in df.columns
    assert df.shape[0] == 2
