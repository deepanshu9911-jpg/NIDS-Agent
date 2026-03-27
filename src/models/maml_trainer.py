"""
maml_trainer.py — MAML meta-learning on the GNN.
Each attack class = one few-shot task (N-way K-shot).
Meta-tests on held-out attack type → simulates zero-day detection.

Run: python -m src.models.maml_trainer --processed_dir data/processed
"""

import argparse, copy, json, logging, os, random
from typing import Iterable, Optional

import torch, torch.nn as nn
import mlflow
from sklearn.metrics import f1_score

from src.models.gnn import NodeClassifier
from src.data.dataset import CICIDS2017Dataset

try:
    import learn2learn as l2l
except ImportError:
    l2l = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAMLTrainer:
    """
    Small compatibility layer for the online self-evolving agent.

    The project originally planned to use a full MAML adapter, but the live
    backend needs a fast-adaptation primitive today. This class clones the
    current node classifier and runs a short inner-loop fine-tune pass on the
    support windows, which keeps the agent/API contract intact while the full
    meta-learning loop continues to exist below for offline experiments.
    """

    def __init__(
        self,
        model: NodeClassifier,
        inner_lr: float = 0.001,
        inner_steps: int = 4,
        weight_decay: float = 1e-4,
        head_only: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.weight_decay = weight_decay
        self.head_only = head_only
        self.device = device or next(model.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def score(self, model: NodeClassifier, graphs: Iterable) -> float:
        graphs = list(graphs)
        if not graphs:
            return 0.0
        model.eval()
        preds, labels = [], []
        for graph in graphs:
            data = graph.to(self.device)
            logits = model(data.x, data.edge_index)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(data.y.view(-1).cpu().numpy())
        return f1_score(labels, preds, average="binary", zero_division=0) if labels else 0.0

    def fast_adapt(
        self,
        support_graphs: Iterable,
        steps: Optional[int] = None,
        validation_graphs: Optional[Iterable] = None,
    ):
        support_graphs = list(support_graphs)
        if not support_graphs:
            return copy.deepcopy(self.model).to(self.device)
        validation_graphs = list(validation_graphs or [])

        with torch.enable_grad():
            adapted = copy.deepcopy(self.model).to(self.device)
            if self.head_only and hasattr(adapted, "encoder"):
                for param in adapted.encoder.parameters():
                    param.requires_grad = False
            adapted.train()
            optimizer = torch.optim.Adam(
                [param for param in adapted.parameters() if param.requires_grad],
                lr=self.inner_lr,
                weight_decay=self.weight_decay,
            )

            adapt_steps = steps or self.inner_steps
            best_state = copy.deepcopy(adapted.state_dict())
            best_score = self.score(adapted, validation_graphs) if validation_graphs else float("-inf")
            for _ in range(adapt_steps):
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.device)
                for graph in support_graphs:
                    data = graph.to(self.device)
                    logits = adapted(data.x, data.edge_index)
                    total_loss = total_loss + self.loss_fn(logits, data.y.view(-1))
                (total_loss / len(support_graphs)).backward()
                optimizer.step()

                if validation_graphs:
                    candidate_score = self.score(adapted, validation_graphs)
                    if candidate_score >= best_score:
                        best_score = candidate_score
                        best_state = copy.deepcopy(adapted.state_dict())

        if validation_graphs:
            adapted.load_state_dict(best_state)

        adapted.eval()
        return adapted

    def export_config(self) -> dict:
        return {
            "inner_lr": self.inner_lr,
            "inner_steps": self.inner_steps,
            "weight_decay": self.weight_decay,
            "head_only": self.head_only,
        }

    @classmethod
    def from_config(cls, model: NodeClassifier, config: Optional[dict] = None):
        return cls(model=model, **(config or {}))


def save_adapter_artifact(path: str, model: NodeClassifier, config: dict, extra: Optional[dict] = None):
    payload = {
        "state_dict": model.state_dict(),
        "adapter_config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_adapter_artifact(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict):
        return artifact
    return None


def _full_edges(n: int) -> torch.Tensor:
    r, c = zip(*[(i, j) for i in range(n) for j in range(n) if i != j]) if n > 1 else ([], [])
    return torch.tensor([list(r), list(c)], dtype=torch.long, device=DEVICE) if r else \
           torch.zeros(2, 0, dtype=torch.long, device=DEVICE)


def sample_episode(graphs, n_way, k_shot, q_query):
    g = random.choice(graphs).to(DEVICE)
    classes = g.y.unique().tolist()
    if len(classes) < 2: return None
    chosen = random.sample(classes, min(n_way, len(classes)))
    sx, sy, qx, qy = [], [], [], []
    for lbl, cls in enumerate(chosen):
        idx = (g.y == cls).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))]
        if len(idx) < k_shot + 1: continue
        sx.append(g.x[idx[:k_shot]])
        sy.append(torch.full((k_shot,), lbl, device=DEVICE))
        qx.append(g.x[idx[k_shot:k_shot+q_query]])
        qy.append(torch.full((min(q_query, len(idx)-k_shot),), lbl, device=DEVICE))
    if not sx: return None
    return torch.cat(sx), torch.cat(sy), torch.cat(qx), torch.cat(qy)


def _sample_graph_episode(graphs, support_windows: int = 8, query_windows: int = 8):
    if not graphs:
        return None
    shuffled = graphs[:]
    random.shuffle(shuffled)
    support = shuffled[:support_windows]
    query = shuffled[support_windows:support_windows + query_windows]
    if len(support) < 2 or len(query) < 2:
        return None
    return support, query


@torch.no_grad()
def _window_f1(model, graphs) -> float:
    preds, labels = [], []
    model.eval()
    for graph in graphs:
        data = graph.to(DEVICE)
        logits = model(data.x, data.edge_index)
        node_preds = logits.argmax(dim=1)
        preds.append(int((node_preds == 1).any().item()))
        labels.append(int((data.y.view(-1) == 1).any().item()))
    return f1_score(labels, preds, zero_division=0) if labels else 0.0


def _first_order_meta_train(args):
    ds = CICIDS2017Dataset(root=args.processed_dir, csv_dir=args.csv_dir).process()
    train_ds, val_ds, _ = ds.split()
    train_g, val_g = list(train_ds), list(val_ds)
    in_ch = train_g[0].x.shape[1]

    model = NodeClassifier(
        in_ch,
        hidden=args.hidden,
        num_classes=2,
        layers=3,
        dropout=0.2,
    ).to(DEVICE)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("experiments/checkpoints", exist_ok=True)
    mlflow.set_experiment("nids-maml-fallback")

    best_f1 = 0.0
    best_config = {
        "inner_lr": args.inner_lr,
        "inner_steps": args.inner_steps,
        "weight_decay": 1e-4,
        "head_only": True,
        "meta_algorithm": "first_order_fallback",
    }

    with mlflow.start_run():
        mlflow.log_params({**vars(args), "meta_backend": "first_order_fallback"})

        for epoch in range(1, args.meta_epochs + 1):
            model.train()
            meta_optimizer.zero_grad()
            total_query_loss = torch.tensor(0.0, device=DEVICE)
            count = 0

            for _ in range(args.meta_batch):
                episode = _sample_graph_episode(
                    train_g,
                    support_windows=max(4, args.k_shot),
                    query_windows=max(4, args.q_query // 2),
                )
                if episode is None:
                    continue
                support_graphs, query_graphs = episode
                adapted = copy.deepcopy(model).to(DEVICE).train()
                inner_opt = torch.optim.SGD(adapted.parameters(), lr=args.inner_lr)

                for _ in range(args.inner_steps):
                    inner_opt.zero_grad()
                    support_loss = torch.tensor(0.0, device=DEVICE)
                    for graph in support_graphs:
                        data = graph.to(DEVICE)
                        support_loss = support_loss + loss_fn(adapted(data.x, data.edge_index), data.y.view(-1))
                    (support_loss / len(support_graphs)).backward()
                    inner_opt.step()

                query_loss = torch.tensor(0.0, device=DEVICE)
                for graph in query_graphs:
                    data = graph.to(DEVICE)
                    query_loss = query_loss + loss_fn(adapted(data.x, data.edge_index), data.y.view(-1))
                total_query_loss = total_query_loss + (query_loss / len(query_graphs))
                count += 1

                with torch.no_grad():
                    for param, adapted_param in zip(model.parameters(), adapted.parameters()):
                        param.data.mul_(1 - args.meta_lr).add_(adapted_param.data * args.meta_lr)

            if count == 0:
                continue

            val_sample = val_g[: min(len(val_g), 128)]
            val_f1 = _window_f1(model, val_sample)
            avg_query_loss = (total_query_loss / count).item()
            mlflow.log_metrics({"meta_loss": avg_query_loss, "val_f1": val_f1}, step=epoch)
            log.info(
                "Fallback meta epoch %4d | meta_loss=%.4f | val_f1=%.4f",
                epoch,
                avg_query_loss,
                val_f1,
            )

            if val_f1 >= best_f1:
                best_f1 = val_f1
                save_adapter_artifact(
                    "experiments/checkpoints/best_maml_adapter.pt",
                    model,
                    best_config,
                    {"val_f1": round(val_f1, 4), "backend": "first_order_fallback"},
                )
                with open("experiments/checkpoints/best_maml_config.json", "w", encoding="utf-8") as fh:
                    json.dump({**best_config, "val_f1": round(val_f1, 4)}, fh, indent=2)

        log.info("Best fallback meta val F1: %.4f", best_f1)


def meta_train(args):
    if l2l is None:
        log.warning("learn2learn is unavailable; falling back to first-order episodic meta-training")
        _first_order_meta_train(args)
        return
    ds = CICIDS2017Dataset(root=args.processed_dir, csv_dir=args.csv_dir).process()
    train_ds, val_ds, _ = ds.split()
    train_g, val_g = list(train_ds), list(val_ds)
    in_ch = train_g[0].x.shape[1]

    base = NodeClassifier(in_ch, hidden=args.hidden, num_classes=args.n_way, layers=2, dropout=0.2).to(DEVICE)
    maml = l2l.algorithms.MAML(base, lr=args.inner_lr, first_order=args.first_order)
    opt  = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("experiments/checkpoints", exist_ok=True)
    mlflow.set_experiment("nids-maml")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        best_acc = 0

        for epoch in range(1, args.meta_epochs + 1):
            opt.zero_grad()
            meta_loss = torch.tensor(0.0, device=DEVICE)
            count = 0

            for _ in range(args.meta_batch):
                ep = sample_episode(train_g, args.n_way, args.k_shot, args.q_query)
                if ep is None: continue
                sx, sy, qx, qy = ep
                learner = maml.clone()
                for _ in range(args.inner_steps):
                    learner.adapt(loss_fn(learner(sx, _full_edges(len(sx))), sy))
                meta_loss += loss_fn(learner(qx, _full_edges(len(qx))), qy)
                count += 1

            if count == 0: continue
            (meta_loss / count).backward()
            opt.step()

            # Validation
            correct = total = 0
            for _ in range(20):
                ep = sample_episode(val_g, args.n_way, args.k_shot, args.q_query)
                if ep is None: continue
                sx, sy, qx, qy = ep
                learner = maml.clone()
                with torch.no_grad():
                    for _ in range(args.inner_steps):
                        learner.adapt(loss_fn(learner(sx, _full_edges(len(sx))), sy))
                    preds = learner(qx, _full_edges(len(qx))).argmax(1)
                correct += (preds == qy).sum().item(); total += len(qy)
            val_acc = correct / total if total else 0

            mlflow.log_metrics({"meta_loss": (meta_loss/count).item(), "val_acc": val_acc}, step=epoch)
            log.info(f"Epoch {epoch:4d} | meta_loss={(meta_loss/count).item():.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(maml.state_dict(), "experiments/checkpoints/best_maml.pt")
                save_adapter_artifact(
                    "experiments/checkpoints/best_maml_adapter.pt",
                    base,
                    {
                        "inner_lr": args.inner_lr,
                        "inner_steps": args.inner_steps,
                        "weight_decay": 1e-4,
                        "head_only": False,
                        "meta_algorithm": "learn2learn_maml",
                    },
                    {"val_acc": round(val_acc, 4), "backend": "learn2learn"},
                )
                with open("experiments/checkpoints/best_maml_config.json", "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "inner_lr": args.inner_lr,
                            "inner_steps": args.inner_steps,
                            "meta_lr": args.meta_lr,
                            "hidden": args.hidden,
                            "n_way": args.n_way,
                            "k_shot": args.k_shot,
                            "q_query": args.q_query,
                            "first_order": args.first_order,
                        },
                        fh,
                        indent=2,
                    )

        log.info(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--csv_dir",       default="data/raw")
    p.add_argument("--n_way",         type=int,   default=2)
    p.add_argument("--k_shot",        type=int,   default=5)
    p.add_argument("--q_query",       type=int,   default=15)
    p.add_argument("--meta_batch",    type=int,   default=8)
    p.add_argument("--inner_steps",   type=int,   default=5)
    p.add_argument("--inner_lr",      type=float, default=0.01)
    p.add_argument("--meta_lr",       type=float, default=1e-3)
    p.add_argument("--meta_epochs",   type=int,   default=100)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--first_order",   action="store_true")
    meta_train(p.parse_args())
