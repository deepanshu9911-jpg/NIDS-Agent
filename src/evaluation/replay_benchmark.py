"""
Replay benchmark for comparing baseline vs adaptive replay on real graph windows.

Usage:
    python -m src.evaluation.replay_benchmark --limit 1000
"""

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.agent.agent import DriftDetector, SelfEvolvingAgent
from src.api.main import _load_real_graphs, _load_real_model
from src.models.maml_trainer import MAMLTrainer, load_adapter_artifact


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ReplayMetrics:
    mode: str
    windows_processed: int
    window_accuracy: float
    window_f1: float
    attack_precision: float
    attack_recall: float
    avg_confidence: float
    drift_events: int
    hypotheses_tried: int
    hypotheses_accepted: int
    final_model_version: int
    runtime_seconds: float
    version_change_points: List[dict]


class BaselineReplayRunner:
    def __init__(self, model):
        self.model = model.to(DEVICE).eval()
        self.detector = DriftDetector()
        self.window_truths = []
        self.window_preds = []
        self.confidences = []
        self.drifts = []
        self.version_history = []

    @torch.no_grad()
    def process(self, graph, window_id: int):
        data = graph.to(DEVICE)
        logits = self.model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)
        node_preds = probs.argmax(dim=1)

        truth = int((data.y.view(-1) == 1).any().item())
        pred = int((node_preds == 1).any().item())
        conf = float(probs[:, 1].max().item()) if pred else float(probs[:, 0].max().item())

        error_rate = float((node_preds.cpu() != data.y.view(-1).cpu()).float().mean().item())
        drift_signal = 0.7 * error_rate + 0.3 * (1.0 - conf)
        trigger = self.detector.update(drift_signal, window_id)
        if trigger:
            self.drifts.append(trigger)

        self.window_truths.append(truth)
        self.window_preds.append(pred)
        self.confidences.append(conf)
        self.version_history.append(0)

    def summary(self, runtime_seconds: float) -> ReplayMetrics:
        return ReplayMetrics(
            mode="baseline",
            windows_processed=len(self.window_truths),
            window_accuracy=accuracy_score(self.window_truths, self.window_preds),
            window_f1=f1_score(self.window_truths, self.window_preds, zero_division=0),
            attack_precision=precision_score(self.window_truths, self.window_preds, zero_division=0),
            attack_recall=recall_score(self.window_truths, self.window_preds, zero_division=0),
            avg_confidence=sum(self.confidences) / len(self.confidences) if self.confidences else 0.0,
            drift_events=len(self.drifts),
            hypotheses_tried=0,
            hypotheses_accepted=0,
            final_model_version=0,
            runtime_seconds=runtime_seconds,
            version_change_points=[{"window_id": 1, "model_version": 0}],
        )


def _adaptive_summary(agent: SelfEvolvingAgent, truths: List[int], runtime_seconds: float) -> ReplayMetrics:
    preds = [alert.prediction for alert in agent.alert_history]
    confidences = [alert.confidence for alert in agent.alert_history]
    version_change_points = []
    last_version = None
    for alert in agent.alert_history:
        if alert.model_version != last_version:
            version_change_points.append({
                "window_id": alert.window_id,
                "model_version": alert.model_version,
            })
            last_version = alert.model_version
    return ReplayMetrics(
        mode="adaptive",
        windows_processed=len(truths),
        window_accuracy=accuracy_score(truths, preds),
        window_f1=f1_score(truths, preds, zero_division=0),
        attack_precision=precision_score(truths, preds, zero_division=0),
        attack_recall=recall_score(truths, preds, zero_division=0),
        avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        drift_events=len(agent.drift_history),
        hypotheses_tried=len(agent.mutation_eng.hypothesis_log),
        hypotheses_accepted=sum(1 for hyp in agent.mutation_eng.hypothesis_log if hyp.accepted),
        final_model_version=agent.model_version,
        runtime_seconds=runtime_seconds,
        version_change_points=version_change_points or [{"window_id": 1, "model_version": 0}],
    )


def _to_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _write_markdown(path: str, baseline: ReplayMetrics, adaptive: ReplayMetrics, meta: dict):
    lines = [
        "# Replay Benchmark",
        "",
        f"- Timestamp: `{meta['timestamp']}`",
        f"- Graph source: `{meta['graphs_path']}`",
        f"- Model checkpoint: `{meta['model_path']}`",
        f"- Windows evaluated: `{meta['window_limit']}`",
        "",
        "| Metric | Baseline | Adaptive |",
        "| --- | ---: | ---: |",
        f"| Window accuracy | {_to_pct(baseline.window_accuracy)} | {_to_pct(adaptive.window_accuracy)} |",
        f"| Window F1 | {baseline.window_f1:.4f} | {adaptive.window_f1:.4f} |",
        f"| Attack precision | {baseline.attack_precision:.4f} | {adaptive.attack_precision:.4f} |",
        f"| Attack recall | {baseline.attack_recall:.4f} | {adaptive.attack_recall:.4f} |",
        f"| Avg confidence | {baseline.avg_confidence:.4f} | {adaptive.avg_confidence:.4f} |",
        f"| Drift events | {baseline.drift_events} | {adaptive.drift_events} |",
        f"| Hypotheses tried | {baseline.hypotheses_tried} | {adaptive.hypotheses_tried} |",
        f"| Hypotheses accepted | {baseline.hypotheses_accepted} | {adaptive.hypotheses_accepted} |",
        f"| Final model version | {baseline.final_model_version} | {adaptive.final_model_version} |",
        f"| Runtime (s) | {baseline.runtime_seconds:.2f} | {adaptive.runtime_seconds:.2f} |",
        "",
        "## Notes",
        "",
        "- Baseline uses the same checkpoint for the entire replay and only records drift triggers.",
        "- Adaptive replay uses the self-evolving agent and can update the model version during the run.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    graphs, active_graphs_path, graph_source = _load_real_graphs(
        args.graphs_path,
        args.raw_csv_dir,
        prefer_cache=args.use_replay_cache,
    )
    graphs = graphs[: args.limit] if args.limit else graphs
    in_channels = int(graphs[0].x.shape[1])

    baseline_model = _load_real_model(args.model_path, in_channels)
    adaptive_model = copy.deepcopy(baseline_model)

    truths = [int((graph.y.view(-1) == 1).any().item()) for graph in graphs]

    baseline = BaselineReplayRunner(baseline_model)
    t0 = time.time()
    for idx, graph in enumerate(graphs, start=1):
        baseline.process(graph, idx)
    baseline_metrics = baseline.summary(time.time() - t0)

    adapter_artifact = load_adapter_artifact(args.adapter_path) if args.adapter_path else None
    adapter_config = (adapter_artifact or {}).get("adapter_config") if adapter_artifact else None
    trainer = MAMLTrainer.from_config(model=adaptive_model, config=adapter_config) if adapter_config else MAMLTrainer(
        model=adaptive_model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        head_only=not args.full_finetune,
    )
    adaptive = SelfEvolvingAgent(model=adaptive_model, maml_trainer=trainer)
    t1 = time.time()
    for idx, graph in enumerate(graphs, start=1):
        adaptive.process_window(graph, window_id=idx)
    adaptive_metrics = _adaptive_summary(adaptive, truths, time.time() - t1)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": timestamp,
        "graphs_path": active_graphs_path,
        "graphs_source": graph_source,
        "model_path": os.path.abspath(args.model_path),
        "adapter_path": os.path.abspath(args.adapter_path) if args.adapter_path else None,
        "window_limit": len(graphs),
    }
    payload = {
        "meta": meta,
        "baseline": asdict(baseline_metrics),
        "adaptive": asdict(adaptive_metrics),
        "adaptive_drift_events": [
            {
                "window_id": event.window_id,
                "detector": event.detector,
                "trigger_reason": event.trigger_reason,
                "trigger_score": round(event.trigger_score, 4),
                "accepted": event.accepted,
                "model_version": event.model_version,
                "before_f1": round(event.before_f1, 4),
                "after_f1": round(event.after_f1, 4),
            }
            for event in adaptive.drift_history
        ],
    }

    json_path = os.path.join(args.output_dir, f"replay_benchmark_{timestamp}.json")
    md_path = os.path.join(args.output_dir, f"replay_benchmark_{timestamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _write_markdown(md_path, baseline_metrics, adaptive_metrics, meta)

    print(json.dumps(payload, indent=2))
    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved Markdown summary to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", default="data/processed/graphs.pt")
    parser.add_argument("--raw_csv_dir", default="data/raw")
    parser.add_argument("--model_path", default="experiments/checkpoints/best_gnn.pt")
    parser.add_argument("--adapter_path", default="experiments/checkpoints/best_maml_adapter.pt")
    parser.add_argument("--output_dir", default="experiments/results")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--inner_lr", type=float, default=0.001)
    parser.add_argument("--inner_steps", type=int, default=4)
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--use_replay_cache", action="store_true")
    run(parser.parse_args())
