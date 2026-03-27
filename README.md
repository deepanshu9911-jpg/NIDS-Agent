# Self-Evolving Network Security Agent

<p align="center">
  <strong>Adaptive intrusion detection with Graph Neural Networks, meta-learning, and concept-drift awareness.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-Dashboard-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#key-features">Features</a> |
  <a href="#dashboard-preview">Dashboard</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#running-the-project">Run</a> |
  <a href="#roadmap">Roadmap</a>
</p>

---

## Overview

The Self-Evolving Network Security Agent is an experimental Network Intrusion Detection System designed to detect and adapt to evolving threats. It combines graph-based traffic modeling, MAML-inspired meta-learning, and drift-aware adaptation to explore how modern ML systems can respond more effectively to zero-day and shifting attack patterns.

This repository includes:

- A graph-learning pipeline for intrusion detection
- A meta-learning workflow for faster adaptation
- A drift detection and model-evolution layer
- A FastAPI backend for serving and orchestration
- A React dashboard for live operational visibility

## Why This Project Stands Out

Traditional NIDS solutions often degrade when network behavior changes. This project is built around a different idea: the model should not remain static.

Instead, the system is designed to:

- represent traffic as graph-structured data
- learn transferable patterns across intrusion scenarios
- detect distribution shifts in real time
- trigger adaptation workflows when the environment changes
- expose model behavior through a security-focused dashboard

## Key Features

- Graph Neural Network baseline for flow classification
- MAML-style meta-learning for rapid few-shot adaptation
- Drift detection pipeline for identifying concept shift
- Model evolution tracking across adaptation cycles
- Real-time dashboard with alerts, timeline, drift log, and adaptation summaries
- Replay benchmarking and experiment tracking with MLflow

## Dashboard Preview

The frontend is designed as an analyst-facing monitoring surface for adaptive intrusion detection.

Current views include:

- Live traffic timeline with attack and drift overlays
- Alert feed for streaming model outputs
- Drift log showing accepted and rejected adaptation events
- Model evolution graph for tracking version behavior
- Summary cards for windows processed, attack rate, confidence, and acceptance rate

You can add dashboard screenshots here later by placing images in a folder like `docs/screenshots/` and referencing them below.

```md
![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![Drift Log](docs/screenshots/drift-log.png)
```

## Tech Stack

| Layer | Tools |
|---|---|
| ML / Training | PyTorch, PyTorch Geometric, Learn2Learn, scikit-learn |
| Drift Detection | River |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, D3, Recharts, Lucide React |
| Tracking | MLflow |
| Data | CICIDS2017 |

## Architecture

1. Raw CICIDS2017 flow records are loaded from `data/raw/`.
2. The dataset layer converts traffic windows into graph-ready samples.
3. A GNN baseline is trained on the processed graph dataset.
4. A MAML-based training stage prepares the model for fast adaptation.
5. The drift detector monitors changes in incoming traffic behavior.
6. Adaptation decisions are surfaced through the backend and dashboard.

## Repository Structure

```text
nids-agent/
|-- data/
|   |-- raw/                  <- CICIDS2017 CSV files (download separately)
|   `-- processed/            <- generated graph datasets and caches
|-- dashboard/                <- React dashboard
|   |-- src/components/       <- UI widgets and visualizations
|   `-- src/hooks/            <- frontend streaming hooks
|-- experiments/
|   |-- checkpoints/          <- saved model weights
|   `-- results/              <- replay benchmark outputs
|-- src/
|   |-- agent/                <- adaptation and drift logic
|   |-- api/                  <- FastAPI application
|   |-- data/                 <- dataset loading and graph construction
|   |-- evaluation/           <- benchmarking and replay evaluation
|   `-- models/               <- GNN and meta-learning training code
|-- requirements.txt
`-- README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/deepanshu9911-jpg/NIDS-agent.git
cd NIDS-agent
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd dashboard
npm install
cd ..
```

## Dataset

This project uses the CICIDS2017 dataset.

- Dataset source: https://www.unb.ca/cic/datasets/ids-2017.html
- Place the required CSV files inside `data/raw/`
- Large raw and processed data artifacts are excluded from GitHub

## Running the Project

### Build the processed graph dataset

```bash
python -c "
from src.data.dataset import CICIDS2017Dataset
ds = CICIDS2017Dataset(root='data/processed', csv_dir='data/raw').process()
train, val, test = ds.split()
print(train.summary(), val.summary(), test.summary())
"
```

### Train the GNN baseline

```bash
python -m src.models.trainer \
    --processed_dir data/processed \
    --epochs 50 --hidden 128 --lr 0.001
```

### Meta-train the adaptive model

```bash
python -m src.models.maml_trainer \
    --processed_dir data/processed \
    --meta_epochs 100 --n_way 2 --k_shot 5
```

### Start the backend API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Start the dashboard

```bash
cd dashboard
npm run dev
```

## Experiment Tracking

To inspect experiments locally:

```bash
mlflow ui
```

Then open `http://localhost:5000`.

## Future Improvements

- Add reproducible training and evaluation pipelines
- Introduce automated testing and CI
- Expand dashboard analytics for adaptation quality
- Package deployment-ready backend endpoints
- Add richer experiment comparison workflows

## Roadmap

- Improve documentation for setup and data preparation
- Add screenshot assets and a demo walkthrough
- Support larger-scale replay evaluation
- Improve robustness of drift-triggered adaptation
- Refine dashboard UX for security analyst workflows

## Author

**Deepanshu**

- GitHub: https://github.com/deepanshu9911-jpg
