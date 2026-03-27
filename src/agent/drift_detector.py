"""
drift_detector.py — Concept drift detection + self-evolution agent loop.

- Uses ADWIN (river library) to detect distribution shift in prediction error
- On drift trigger: samples recent flows, fast-adapts the GNN via MAML
- Logs hypothesis history (what changed, did it improve F1?)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np

try:
    from river.drift import ADWIN
except ImportError:
    raise ImportError("pip install river")

log = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """One candidate model adaptation."""
    timestamp: float
    trigger_reason: str
    pre_f1: float
    post_f1: float
    accepted: bool
    adapter_steps: int


class DriftDetector:
    """
    Wraps ADWIN to monitor a sliding stream of per-sample prediction errors.
    Call update(error) every time the model makes a prediction.
    """

    def __init__(self, delta: float = 0.002):
        self._adwin = ADWIN(delta=delta)
        self._drift_count = 0

    def update(self, error: float) -> bool:
        """
        Args:
            error: 0.0 (correct prediction) or 1.0 (wrong prediction)
        Returns:
            True if drift detected this step
        """
        self._adwin.update(error)
        if self._adwin.drift_detected:
            self._drift_count += 1
            log.warning(f"Drift detected! (total drifts: {self._drift_count})")
            return True
        return False

    @property
    def drift_count(self):
        return self._drift_count


class MutationEngine:
    """
    Self-evolving agent loop.

    When drift is detected, it:
    1. Collects a buffer of recent flows
    2. Fast-adapts the model (MAML inner loop)
    3. Evaluates adapted model vs original on the buffer
    4. Accepts or discards the mutation
    5. Logs the hypothesis
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        buffer_size: int = 500,
        accept_threshold: float = 0.01,   # min F1 improvement to accept
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.buffer = deque(maxlen=buffer_size)    # stores (x, y, edge_index) tuples
        self.detector = DriftDetector()
        self.history: List[Hypothesis] = []
        self.accept_threshold = accept_threshold
        self._loss_fn = nn.CrossEntropyLoss()
        self.device = next(model.parameters()).device

    # ----------------------------------------------------------------- ingest
    def ingest(self, graph_data) -> Optional[dict]:
        """
        Process one incoming graph. Returns an alert dict if attack detected.
        Also checks for drift and triggers mutation if needed.

        Args:
            graph_data: PyG Data object (one traffic window)
        Returns:
            dict with alert info, or None if benign
        """
        self.model.eval()
        with torch.no_grad():
            data = graph_data.to(self.device)
            logits = self.model(data.x, data.edge_index)
            probs  = logits.softmax(dim=1)
            preds  = logits.argmax(dim=1)

        # Update drift detector with error rate for this window
        error_rate = (preds != data.y).float().mean().item()
        drift = self.detector.update(error_rate)

        # Buffer this window for potential adaptation
        self.buffer.append(data)

        # On drift: attempt to mutate
        if drift and len(self.buffer) >= 50:
            self._mutate()

        # Build alert if any attack nodes detected
        attack_mask = (preds == 1)
        if attack_mask.any():
            return {
                "timestamp": time.time(),
                "attack_nodes": int(attack_mask.sum()),
                "total_nodes": int(len(preds)),
                "max_confidence": float(probs[:, 1].max()),
                "drift_detected": drift,
            }
        return None

    # ---------------------------------------------------------------- mutate
    def _mutate(self):
        """Fast-adapt the model on buffered data, accept if F1 improves."""
        recent = list(self.buffer)[-100:]   # last 100 windows

        pre_f1 = self._estimate_f1(recent)
        log.info(f"Mutation attempt | pre-F1={pre_f1:.4f}")

        # Clone model for adaptation (don't modify original yet)
        adapted = self._clone_and_adapt(recent)

        post_f1 = self._estimate_f1(recent, model=adapted)
        accepted = (post_f1 - pre_f1) >= self.accept_threshold

        hyp = Hypothesis(
            timestamp=time.time(),
            trigger_reason="ADWIN drift",
            pre_f1=pre_f1,
            post_f1=post_f1,
            accepted=accepted,
            adapter_steps=self.inner_steps,
        )
        self.history.append(hyp)

        if accepted:
            self.model.load_state_dict(adapted.state_dict())
            log.info(f"Mutation ACCEPTED: F1 {pre_f1:.4f} → {post_f1:.4f}")
        else:
            log.info(f"Mutation REJECTED: F1 {pre_f1:.4f} → {post_f1:.4f} (below threshold)")

    def _clone_and_adapt(self, windows: list) -> nn.Module:
        import copy
        adapted = copy.deepcopy(self.model).train()
        opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            for data in windows:
                logits = adapted(data.x, data.edge_index)
                total_loss += self._loss_fn(logits, data.y)
            (total_loss / len(windows)).backward()
            opt.step()
        return adapted

    @torch.no_grad()
    def _estimate_f1(self, windows: list, model=None) -> float:
        from sklearn.metrics import f1_score
        m = model if model is not None else self.model
        m.eval()
        all_preds, all_labels = [], []
        for data in windows:
            preds = m(data.x, data.edge_index).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())
        return f1_score(all_labels, all_preds, average="binary", zero_division=0)

    def hypothesis_log(self) -> list:
        """Return history as list of dicts (for API / dashboard)."""
        return [
            {
                "timestamp": h.timestamp,
                "trigger": h.trigger_reason,
                "pre_f1": round(h.pre_f1, 4),
                "post_f1": round(h.post_f1, 4),
                "accepted": h.accepted,
                "steps": h.adapter_steps,
            }
            for h in self.history
        ]
