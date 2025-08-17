FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

CMD ["python", "-m", "src.training.train", \
     "--edges_csv", "data/edges.csv", \
     "--train_pairs_csv", "data/train_pairs.csv", \
     "--val_pairs_csv", "data/val_pairs.csv", \
     "--out_dir", "artifacts"]
