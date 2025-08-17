# Meta Social Network — Graph Link Prediction

Predict missing directed edges (“who should follow whom?”) to power **People You May Know**–style recommendations at scale.

---

## 🧭 Overview
This project implements a **link prediction system** on a **directed social graph**.  

It supports:
- Offline training with heuristics, embeddings, and GNNs
- Batch scoring for large candidate sets
- Online inference API (FastAPI)
- Evaluation metrics (AUC, Precision@K, Recall@K)
- Scalable data pipeline (Spark-ready)
- Dockerized deployment

---
**Tech Stack:** Python · PyTorch · PyTorch Geometric/DGL · FastAPI · Spark · Redis · FAISS · Docker · MLflow

---

## Problem Statement
Given a directed social graph \( G=(V,E) \) where edge \((u→v)\) means *user u follows user v*, predict **missing links** for recommendation.

- **Input**: Edges with timestamps; optional user attributes  
- **Output**: Ranked candidate list for each user  
- **Challenges**:
  - Directionality matters
  - Cold-start users with few connections
  - Temporal leakage avoidance
  - Scale: millions of users, billions of edges  

---

## Data
Example files:
- `edges.csv`: `src_user_id,dst_user_id,timestamp`
- `users.csv`: `user_id,country,age,joined_at,...`

### Splitting
- **Temporal**: train/val/test by cutoff timestamps  
- **Negatives**: sample K non-edges per positive edge  
- **Cold-start bucket**: hold out users with ≤N edges  

---

##  Features
- **Graph topology**: in/out degree, reciprocity, common neighbors (CN), Adamic–Adar, Katz, Personalized PageRank, 2-hop paths  
- **Node attributes**: demographics, activity, embeddings from bio text  
- **Edge attributes**: recency, frequency, mutual follows, interaction counts  

---

##  Models
- **Heuristics**: Common Neighbors, Adamic–Adar, Katz, PPR  
- **Embeddings**: Node2Vec, DeepWalk, LINE  
- **GNNs**: GraphSAGE, GAT with edge decoders (dot/MLP/bilinear)  
- **Hybrid**: Heuristics for recall, GNN/MLP for ranking  

---

##  Evaluation Metrics
- **AUC-ROC**, **Average Precision (AP)**  
- **Precision@K, Recall@K, NDCG@K**  
- **Coverage/Diversity** for fairness & business goals  
- **Latency p95/p99** for serving  

---

##  Setup
### Requirements
- Python 3.10+  
- PyTorch + PyTorch Geometric (or DGL)  
- FastAPI, Uvicorn  
- Optional: Spark, Redis, FAISS  

```bash
pip install -r requirements.txt
pytest -q   # run tests
