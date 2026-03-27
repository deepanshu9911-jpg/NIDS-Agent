# Self-Evolving Network Security Agent (NIDS)

GNN + MAML meta-learning system for zero-day intrusion detection.

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 - Download CICIDS2017

Download from: https://www.unb.ca/cic/datasets/ids-2017.html
Place `*_Flow.csv` files in `data/raw/`

## Step 2 - Build graph dataset (run once)

```bash
python -c "
from src.data.dataset import CICIDS2017Dataset
ds = CICIDS2017Dataset(root='data/processed', csv_dir='data/raw').process()
train, val, test = ds.split()
print(train.summary(), val.summary(), test.summary())
"
```

## Step 3 - Train GNN baseline

```bash
python -m src.models.trainer \
    --processed_dir data/processed \
    --epochs 50 --hidden 128 --lr 0.001
```

## Step 4 - Meta-train MAML

```bash
python -m src.models.maml_trainer \
    --processed_dir data/processed \
    --meta_epochs 100 --n_way 2 --k_shot 5
```

## Step 5 - Start API backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Step 6 - Start React dashboard

```bash
cd dashboard && npm install && npm run dev
```

## Project structure

```text
nids-agent/
|-- data/raw/             <- CICIDS2017 CSVs (download separately)
|-- data/processed/       <- built graph .pt files (auto-generated)
|-- src/
|   |-- data/
|   |   |-- graph_builder.py   <- flow CSV -> PyG graph
|   |   `-- dataset.py         <- windowing + train/val/test split
|   |-- models/
|   |   |-- gnn.py             <- GraphSAGE model
|   |   |-- trainer.py         <- baseline training loop
|   |   `-- maml_trainer.py    <- MAML meta-learning
|   |-- agent/
|   |   `-- drift_detector.py  <- ADWIN + mutation engine
|   `-- api/
|       `-- main.py            <- FastAPI backend
|-- dashboard/                 <- React SIEM UI (Week 6-7)
|-- experiments/checkpoints/   <- saved model weights
`-- requirements.txt
```

## MLflow

```bash
mlflow ui  # open http://localhost:5000
```
