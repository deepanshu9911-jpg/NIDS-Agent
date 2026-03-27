"""
agent.py
Online self-evolving agent for the dashboard/backend.

Pipeline:
  graph window
      ->
  node-level GNN prediction
      ->
  ADWIN watches window error/confidence
      -> (drift detected)
  collect recent windows as support set
      ->
  fast-adapt cloned model
      ->
  compare F1 on validation buffer
      -> (if improved)
  accept model, increment version, log drift event
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data

try:
    from river.drift import ADWIN
except ImportError:
    ADWIN = None

from src.models.gnn import NodeClassifier
from src.models.maml_trainer import MAMLTrainer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Alert:
    timestamp: float
    window_id: int
    prediction: int
    confidence: float
    attack_type: Optional[str] = None
    model_version: int = 0


@dataclass
class DriftEvent:
    timestamp: float
    window_id: int
    detector: str
    action: str
    trigger_reason: str = ""
    trigger_score: float = 0.0
    before_f1: float = 0.0
    after_f1: float = 0.0
    support_f1_before: float = 0.0
    support_f1_after: float = 0.0
    support_loss_before: float = 0.0
    support_loss_after: float = 0.0
    val_loss_before: float = 0.0
    val_loss_after: float = 0.0
    tune_f1_before: float = 0.0
    tune_f1_after: float = 0.0
    adapt_steps: int = 0
    accepted: bool = False
    model_version: int = 0


@dataclass
class Hypothesis:
    hypothesis_id: int
    created_at: float
    support_window_ids: List[int]
    adapt_steps: int
    support_f1_before: float
    support_f1_after: float
    support_loss_before: float
    support_loss_after: float
    val_f1_before: float
    val_f1_after: float
    val_loss_before: float
    val_loss_after: float
    tune_f1_before: float
    tune_f1_after: float
    accepted: bool


class DriftDetector:
    def __init__(
        self,
        delta: float = 0.002,
        baseline_window: int = 96,
        recent_window: int = 24,
        min_shift: float = 0.012,
        cooldown: int = 80,
    ):
        self.detector = ADWIN(delta=delta) if ADWIN is not None else None
        self.drift_count = 0
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.min_shift = min_shift
        self.cooldown = cooldown
        self.history = deque(maxlen=baseline_window + recent_window)
        self.last_trigger_window = -cooldown

    def update(self, value: float, window_id: int) -> Optional[dict]:
        self.history.append(float(value))

        if self.detector is not None:
            self.detector.update(value)
            if self.detector.drift_detected and (window_id - self.last_trigger_window) >= self.cooldown:
                self.drift_count += 1
                self.last_trigger_window = window_id
                return {
                    "detected": True,
                    "detector": "ADWIN",
                    "score": float(value),
                    "reason": "ADWIN detected a distribution change in the drift signal",
                }

        if len(self.history) < self.baseline_window + self.recent_window:
            return None
        if (window_id - self.last_trigger_window) < self.cooldown:
            return None

        baseline = list(self.history)[: self.baseline_window]
        recent = list(self.history)[-self.recent_window :]
        baseline_mean = sum(baseline) / len(baseline)
        recent_mean = sum(recent) / len(recent)
        shift = recent_mean - baseline_mean

        if shift >= self.min_shift:
            self.drift_count += 1
            self.last_trigger_window = window_id
            return {
                "detected": True,
                "detector": "SHIFT",
                "score": round(shift, 4),
                "reason": f"Recent drift score rose from {baseline_mean:.3f} to {recent_mean:.3f}",
            }

        return None

    def reset(self):
        if ADWIN is not None:
            self.detector = ADWIN()
        self.history.clear()
        self.last_trigger_window = -self.cooldown


class MutationEngine:
    def __init__(
        self,
        maml_trainer: MAMLTrainer,
        support_size: int = 20,
        min_support: int = 8,
        min_improvement: float = 0.0005,
        max_holdout_drop: float = 0.0015,
        min_tune_gain: float = 0.0,
        min_support_gain: float = 0.002,
        min_support_loss_drop: float = 0.0002,
        max_holdout_loss_increase: float = 0.001,
        adapt_steps: int = 4,
    ):
        self.maml_trainer = maml_trainer
        self.support_size = support_size
        self.min_support = min_support
        self.min_improvement = min_improvement
        self.max_holdout_drop = max_holdout_drop
        self.min_tune_gain = min_tune_gain
        self.min_support_gain = min_support_gain
        self.min_support_loss_drop = min_support_loss_drop
        self.max_holdout_loss_increase = max_holdout_loss_increase
        self.adapt_steps = adapt_steps
        self.hypothesis_log: List[Hypothesis] = []
        self._hyp_counter = 0

    def _is_attack_window(self, graph: Data) -> bool:
        labels = getattr(graph, "y", None)
        return bool(labels is not None and int(labels.view(-1).max().item()) == 1)

    def _build_support_set(self, recent_windows: List[Data]) -> List[Data]:
        pool = recent_windows[-max(self.support_size * 3, self.min_support):]
        attack_windows = [graph for graph in pool if self._is_attack_window(graph)]
        benign_windows = [graph for graph in pool if not self._is_attack_window(graph)]

        attack_target = min(len(attack_windows), max(1, int(self.support_size * 0.65)))
        benign_target = min(len(benign_windows), self.support_size - attack_target)

        support = attack_windows[-attack_target:] + benign_windows[-benign_target:]

        if len(support) < self.support_size:
            selected_ids = {id(graph) for graph in support}
            leftovers = [
                graph for graph in reversed(pool)
                if id(graph) not in selected_ids
            ]
            support.extend(list(reversed(leftovers[: self.support_size - len(support)])))

        return support[-self.support_size:]

    def _split_validation_windows(self, val_windows: List[Data]):
        if len(val_windows) < self.min_support * 2:
            return val_windows, val_windows

        tuning = val_windows[::2]
        holdout = val_windows[1::2]
        if len(holdout) < self.min_support:
            holdout = val_windows[-self.min_support:]
        if len(tuning) < self.min_support:
            tuning = val_windows[: self.min_support]
        return tuning, holdout

    def mutate(
        self,
        current_model: NodeClassifier,
        recent_windows: List[Data],
        val_windows: List[Data],
        window_ids: List[int],
    ):
        self._hyp_counter += 1

        support = self._build_support_set(recent_windows)
        tuning_windows, holdout_windows = self._split_validation_windows(val_windows)

        if (
            len(support) < self.min_support
            or len(tuning_windows) < self.min_support
            or len(holdout_windows) < self.min_support
        ):
            log.info("Skipping adaptation: not enough support/validation windows yet")
            return False, None, None

        support_before = self._quick_f1(current_model, support)
        support_loss_before = self._quick_loss(current_model, support)
        f1_before = self._quick_f1(current_model, holdout_windows)
        holdout_loss_before = self._quick_loss(current_model, holdout_windows)
        tune_before = self._quick_f1(current_model, tuning_windows)
        t0 = time.time()
        adapted = self.maml_trainer.fast_adapt(
            support,
            steps=self.adapt_steps,
            validation_graphs=tuning_windows,
        )
        adapt_time = round(time.time() - t0, 2)
        support_after = self._quick_f1(adapted, support)
        support_loss_after = self._quick_loss(adapted, support)
        f1_after = self._quick_f1(adapted, holdout_windows)
        holdout_loss_after = self._quick_loss(adapted, holdout_windows)
        tune_after = self._quick_f1(adapted, tuning_windows)
        support_gain = support_after - support_before
        holdout_gain = f1_after - f1_before
        tune_gain = tune_after - tune_before
        support_loss_drop = support_loss_before - support_loss_after
        holdout_loss_increase = holdout_loss_after - holdout_loss_before
        accepted = (
            holdout_gain >= self.min_improvement
            or (
                holdout_gain >= -self.max_holdout_drop
                and tune_gain >= self.min_tune_gain
                and support_gain >= self.min_support_gain
            )
            or (
                holdout_gain >= -self.max_holdout_drop
                and holdout_loss_increase <= self.max_holdout_loss_increase
                and support_loss_drop >= self.min_support_loss_drop
            )
        )

        hyp = Hypothesis(
            hypothesis_id=self._hyp_counter,
            created_at=time.time(),
            support_window_ids=window_ids[-len(support):],
            adapt_steps=self.adapt_steps,
            support_f1_before=round(support_before, 4),
            support_f1_after=round(support_after, 4),
            support_loss_before=round(support_loss_before, 4),
            support_loss_after=round(support_loss_after, 4),
            val_f1_before=round(f1_before, 4),
            val_f1_after=round(f1_after, 4),
            val_loss_before=round(holdout_loss_before, 4),
            val_loss_after=round(holdout_loss_after, 4),
            tune_f1_before=round(tune_before, 4),
            tune_f1_after=round(tune_after, 4),
            accepted=accepted,
        )
        self.hypothesis_log.append(hyp)

        log.info(
            "Hypothesis #%s: support F1 %.4f -> %.4f loss %.4f -> %.4f | holdout F1 %.4f -> %.4f loss %.4f -> %.4f | tuning F1 %.4f -> %.4f | %s | %ss",
            self._hyp_counter,
            support_before,
            support_after,
            support_loss_before,
            support_loss_after,
            f1_before,
            f1_after,
            holdout_loss_before,
            holdout_loss_after,
            tune_before,
            tune_after,
            "ACCEPTED" if accepted else "REJECTED",
            adapt_time,
        )

        return accepted, adapted if accepted else None, hyp

    @torch.no_grad()
    def _quick_f1(self, model, windows: List[Data]) -> float:
        if not windows:
            return 0.0
        model.eval()
        preds, labels = [], []
        for graph in windows:
            data = graph.to(DEVICE)
            logits = model(data.x, data.edge_index)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(data.y.view(-1).cpu().numpy())
        return f1_score(labels, preds, average="binary", zero_division=0)

    @torch.no_grad()
    def _quick_loss(self, model, windows: List[Data]) -> float:
        if not windows:
            return 0.0
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for graph in windows:
            data = graph.to(DEVICE)
            logits = model(data.x, data.edge_index)
            total_loss += float(loss_fn(logits, data.y.view(-1)).item())
        return total_loss / len(windows)


class SelfEvolvingAgent:
    def __init__(
        self,
        model: NodeClassifier,
        maml_trainer: MAMLTrainer,
        buffer_size: int = 200,
        val_buffer_size: int = 64,
        drift_delta: float = 0.002,
        on_alert: Optional[Callable[[Alert], None]] = None,
        on_drift: Optional[Callable[[DriftEvent], None]] = None,
    ):
        self.model = model.to(DEVICE).eval()
        self.maml_trainer = maml_trainer
        self.mutation_eng = MutationEngine(maml_trainer)
        self.drift_det = DriftDetector(delta=drift_delta)

        self.replay_buffer: deque = deque(maxlen=buffer_size)
        self.val_buffer: deque = deque(maxlen=val_buffer_size)

        self.on_alert = on_alert
        self.on_drift = on_drift

        self.alert_history: List[Alert] = []
        self.drift_history: List[DriftEvent] = []
        self.model_version = 0
        self._window_counter = 0

    @torch.no_grad()
    def process_window(self, window: Data, window_id: Optional[int] = None) -> Alert:
        self._window_counter += 1
        wid = self._window_counter if window_id is None else window_id

        data = window.to(DEVICE)
        self.model.eval()
        logits = self.model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)
        node_preds = probs.argmax(dim=1)

        attack_nodes = int((node_preds == 1).sum().item())
        prediction = int(attack_nodes > 0)
        confidence = float(probs[:, 1].max().item()) if prediction else float(probs[:, 0].max().item())

        alert = Alert(
            timestamp=time.time(),
            window_id=wid,
            prediction=prediction,
            confidence=confidence,
            model_version=self.model_version,
        )
        self.alert_history.append(alert)
        if self.on_alert:
            self.on_alert(alert)

        self.replay_buffer.append(data.cpu())
        if wid % 4 == 0:
            self.val_buffer.append(data.cpu())

        error_rate = float((node_preds.cpu() != data.y.view(-1).cpu()).float().mean().item())
        drift_signal = 0.7 * error_rate + 0.3 * (1.0 - confidence)
        trigger = self.drift_det.update(drift_signal, wid)
        if trigger and len(self.val_buffer) >= self.mutation_eng.min_support:
            self._handle_drift(
                wid,
                detector=trigger["detector"],
                trigger_reason=trigger["reason"],
                trigger_score=trigger["score"],
            )

        return alert

    def _handle_drift(self, window_id: int, detector: str, trigger_reason: str, trigger_score: float):
        recent_windows = list(self.replay_buffer)
        val_windows = list(self.val_buffer)
        window_ids = list(range(max(0, window_id - len(recent_windows)), window_id))

        before_f1 = self.mutation_eng._quick_f1(self.model, val_windows)
        accepted, new_model, hyp = self.mutation_eng.mutate(
            current_model=self.model,
            recent_windows=recent_windows,
            val_windows=val_windows,
            window_ids=window_ids,
        )
        after_f1 = hyp.val_f1_after if hyp else before_f1

        if accepted and new_model is not None:
            self.model = new_model.to(DEVICE).eval()
            self.maml_trainer.model = self.model
            self.model_version += 1
            action = "adapt"
        else:
            action = "skip"

        event = DriftEvent(
            timestamp=time.time(),
            window_id=window_id,
            detector=detector,
            action=action,
            trigger_reason=trigger_reason,
            trigger_score=trigger_score,
            before_f1=before_f1,
            after_f1=after_f1,
            support_f1_before=hyp.support_f1_before if hyp else 0.0,
            support_f1_after=hyp.support_f1_after if hyp else 0.0,
            support_loss_before=hyp.support_loss_before if hyp else 0.0,
            support_loss_after=hyp.support_loss_after if hyp else 0.0,
            val_loss_before=hyp.val_loss_before if hyp else 0.0,
            val_loss_after=hyp.val_loss_after if hyp else 0.0,
            tune_f1_before=hyp.tune_f1_before if hyp else 0.0,
            tune_f1_after=hyp.tune_f1_after if hyp else 0.0,
            adapt_steps=self.mutation_eng.adapt_steps,
            accepted=accepted,
            model_version=self.model_version,
        )
        self.drift_history.append(event)
        if self.on_drift:
            self.on_drift(event)

        if accepted:
            log.info("[DRIFT] Model updated to version %s", self.model_version)
        else:
            log.info("[DRIFT] Adaptation rejected; keeping current model")

    def get_stats(self) -> dict:
        return {
            "windows_processed": self._window_counter,
            "total_alerts": len(self.alert_history),
            "attack_alerts": sum(1 for a in self.alert_history if a.prediction == 1),
            "drift_events": len(self.drift_history),
            "model_version": self.model_version,
            "hypotheses_tried": len(self.mutation_eng.hypothesis_log),
            "hypotheses_accepted": sum(1 for h in self.mutation_eng.hypothesis_log if h.accepted),
        }
