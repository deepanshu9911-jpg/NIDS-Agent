"""
trainer.py — GNN baseline training loop with MLflow tracking.

Features: class-weighted loss, early stopping, best checkpoint save.

Run:
    python -m src.models.trainer --processed_dir data/processed --epochs 50
"""

import argparse, logging, os
import torch, torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report
import mlflow

from src.data.dataset import CICIDS2017Dataset
from src.models.gnn import build_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(data.x, data.edge_index), data.y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(DEVICE)
        preds.extend(model(data.x, data.edge_index).argmax(1).cpu().numpy())
        labels.extend(data.y.cpu().numpy())
    return f1_score(labels, preds, average="binary", zero_division=0), preds, labels


def train(args):
    ds = CICIDS2017Dataset(root=args.processed_dir, csv_dir=args.csv_dir,
                           window_size=args.window_size).process()
    train_ds, val_ds, test_ds = ds.split()
    for s in (train_ds, val_ds, test_ds):
        log.info(s.summary())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    in_ch = train_ds[0].x.shape[1]
    model = build_model(in_ch, task="node", hidden=args.hidden,
                        num_classes=2, layers=args.layers, dropout=args.dropout).to(DEVICE)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    r = train_ds.attack_ratio()
    crit = nn.CrossEntropyLoss(weight=torch.tensor([r, 1 - r], dtype=torch.float).to(DEVICE))
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    os.makedirs("experiments/checkpoints", exist_ok=True)
    mlflow.set_experiment("nids-gnn-baseline")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        best_f1, patience_ct = 0, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, opt, crit)
            val_f1, _, _ = evaluate(model, val_loader)
            sched.step(1 - val_f1)
            mlflow.log_metrics({"train_loss": loss, "val_f1": val_f1}, step=epoch)
            log.info(f"Epoch {epoch:3d} | loss={loss:.4f} | val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1, patience_ct = val_f1, 0
                ckpt = "experiments/checkpoints/best_gnn.pt"
                torch.save(model.state_dict(), ckpt)
            else:
                patience_ct += 1
                if patience_ct >= args.patience:
                    log.info(f"Early stop @ epoch {epoch}")
                    break

        model.load_state_dict(torch.load("experiments/checkpoints/best_gnn.pt"))
        test_f1, preds, labels = evaluate(model, test_loader)
        mlflow.log_metric("test_f1", test_f1)
        log.info(f"\nTest F1: {test_f1:.4f}")
        log.info("\n" + classification_report(labels, preds, target_names=["BENIGN","ATTACK"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--csv_dir",       default="data/raw")
    p.add_argument("--window_size",   type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--layers",        type=int,   default=3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--patience",      type=int,   default=10)
    train(p.parse_args())
