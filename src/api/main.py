"""
FastAPI backend for the NIDS dashboard.

By default this backend now expects real CICIDS2017 artifacts:
- processed graph windows at data/processed/graphs.pt
- a trained checkpoint at experiments/checkpoints/best_gnn.pt

Set NIDS_DEMO_MODE=1 only when you explicitly want the fake demo stream.
"""

import asyncio
import glob
import json
import os
import threading
import time
from dataclasses import dataclass
import logging

import numpy as np
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="NIDS Self-Evolving Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global state
agent = None
graphs = None
is_running = False

alert_store = []
drift_store = []
runtime_info = {
    "mode": "uninitialized",
    "graphs_path": None,
    "graphs_source": None,
    "model_path": None,
    "adapter_path": None,
    "graph_count": 0,
}


class StatsSchema(BaseModel):
    windows_processed: int
    total_alerts: int
    attack_alerts: int
    drift_events: int
    model_version: int
    hypotheses_tried: int
    hypotheses_accepted: int


@dataclass
class ReplayAlert:
    timestamp: float
    window_id: int
    prediction: int
    confidence: float
    model_version: int = 0


@dataclass
class ReplayDriftEvent:
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


def on_alert(alert):
    alert_store.append({
        "type": "alert",
        "timestamp": alert.timestamp,
        "window_id": alert.window_id,
        "prediction": int(alert.prediction),
        "confidence": round(float(alert.confidence), 4),
        "model_version": int(alert.model_version),
    })


def on_drift(event):
    drift_store.append({
        "type": "drift",
        "timestamp": event.timestamp,
        "window_id": event.window_id,
        "detector": event.detector,
        "action": event.action,
        "trigger_reason": getattr(event, "trigger_reason", ""),
        "trigger_score": round(float(getattr(event, "trigger_score", 0.0)), 4),
        "before_f1": round(float(event.before_f1), 4),
        "after_f1": round(float(event.after_f1), 4),
        "support_f1_before": round(float(getattr(event, "support_f1_before", 0.0)), 4),
        "support_f1_after": round(float(getattr(event, "support_f1_after", 0.0)), 4),
        "support_loss_before": round(float(getattr(event, "support_loss_before", 0.0)), 4),
        "support_loss_after": round(float(getattr(event, "support_loss_after", 0.0)), 4),
        "val_loss_before": round(float(getattr(event, "val_loss_before", 0.0)), 4),
        "val_loss_after": round(float(getattr(event, "val_loss_after", 0.0)), 4),
        "tune_f1_before": round(float(getattr(event, "tune_f1_before", 0.0)), 4),
        "tune_f1_after": round(float(getattr(event, "tune_f1_after", 0.0)), 4),
        "adapt_steps": int(getattr(event, "adapt_steps", 0)),
        "accepted": bool(event.accepted),
        "model_version": int(getattr(event, "model_version", 0)),
    })


class RealReplayAgent:
    """
    Replays real graph windows through a trained node classifier.

    This replaces the old fake stream so the dashboard reflects real data and a
    real checkpoint, even before the MAML adapter is fully wired in.
    """

    def __init__(self, model, on_alert=None, on_drift=None, drift_delta: float = 0.002):
        self.model = model.to(DEVICE).eval()
        self.on_alert = on_alert
        self.on_drift = on_drift

        self.model_version = 0
        self._window_counter = 0
        self.alert_history = []
        self.drift_history = []
        self.mutation_eng = type("obj", (object,), {"hypothesis_log": []})()

        self._drift_detector = None
        try:
            from river.drift import ADWIN
            self._drift_detector = ADWIN(delta=drift_delta)
        except Exception as exc:
            log.warning(f"ADWIN unavailable for live drift tracking: {exc}")

    @torch.no_grad()
    def process_window(self, graph, window_id: Optional[int] = None):
        self._window_counter += 1
        wid = self._window_counter if window_id is None else window_id

        data = graph.to(DEVICE)
        logits = self.model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)
        node_preds = probs.argmax(dim=1)

        attack_nodes = int((node_preds == 1).sum().item())
        prediction = int(attack_nodes > 0)
        confidence = float(probs[:, 1].max().item()) if prediction else float(probs[:, 0].max().item())

        alert = ReplayAlert(
            timestamp=time.time(),
            window_id=wid,
            prediction=prediction,
            confidence=confidence,
            model_version=self.model_version,
        )
        self.alert_history.append(alert)
        if self.on_alert:
            self.on_alert(alert)

        labels = getattr(data, "y", None)
        if labels is not None and self._drift_detector is not None:
            labels = labels.view(-1)
            if labels.numel() == 1 and node_preds.numel() > 1:
                labels = labels.repeat(node_preds.numel())

            error_rate = float((node_preds.cpu() != labels.cpu()).float().mean().item())
            self._drift_detector.update(error_rate)
            if self._drift_detector.drift_detected:
                score = max(0.0, 1.0 - error_rate)
                event = ReplayDriftEvent(
                    timestamp=time.time(),
                    window_id=wid,
                    detector="ADWIN",
                    action="skip",
                    trigger_reason="Replay-only mode does not adapt the model",
                    trigger_score=score,
                    before_f1=score,
                    after_f1=score,
                    adapt_steps=0,
                    accepted=False,
                    model_version=self.model_version,
                )
                self.drift_history.append(event)
                if self.on_drift:
                    self.on_drift(event)

        return alert

    def get_stats(self):
        return {
            "windows_processed": self._window_counter,
            "total_alerts": len(self.alert_history),
            "attack_alerts": sum(1 for a in self.alert_history if a.prediction == 1),
            "drift_events": len(self.drift_history),
            "model_version": self.model_version,
            "hypotheses_tried": len(self.mutation_eng.hypothesis_log),
            "hypotheses_accepted": sum(
                1 for h in self.mutation_eng.hypothesis_log if getattr(h, "accepted", False)
            ),
        }


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_path(path: str) -> str:
    return os.path.abspath(path)


def _prepare_graph(graph):
    for attr in ("x", "edge_index", "y"):
        if not hasattr(graph, attr):
            raise ValueError(f"Graph window is missing required attribute '{attr}'")
    return graph


def _load_real_graphs(graphs_path: str, raw_csv_dir: str, prefer_cache: bool = True):
    requested_graphs_path = _resolve_path(graphs_path)
    resolved_raw_dir = _resolve_path(raw_csv_dir)
    candidate_paths = []

    if prefer_cache:
        cache_path = os.path.join(os.path.dirname(requested_graphs_path), "replay_cache.pt")
        candidate_paths.append(cache_path)
    candidate_paths.append(requested_graphs_path)

    loaded_graphs = None
    resolved_graphs_path = requested_graphs_path
    for candidate in candidate_paths:
        resolved_candidate = _resolve_path(candidate)
        if os.path.exists(resolved_candidate):
            resolved_graphs_path = resolved_candidate
            log.info(f"Loading processed graphs from {resolved_candidate}")
            loaded_graphs = _torch_load(resolved_candidate)
            break

    if loaded_graphs is None:
        if not os.path.exists(requested_graphs_path):
            csv_files = glob.glob(os.path.join(resolved_raw_dir, "*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"Processed graphs not found at {requested_graphs_path} and no CICIDS2017 CSVs found in {resolved_raw_dir}."
                )

            log.info(f"Processed graphs missing. Building them from raw CSVs in {resolved_raw_dir}")
            from src.data.dataset import CICIDS2017Dataset

            dataset = CICIDS2017Dataset(
                root=os.path.dirname(requested_graphs_path),
                csv_dir=resolved_raw_dir,
            ).process(force=False)
            loaded_graphs = dataset.graphs
            resolved_graphs_path = _resolve_path(os.path.join(os.path.dirname(requested_graphs_path), "replay_cache.pt"))
            if not os.path.exists(resolved_graphs_path):
                resolved_graphs_path = requested_graphs_path
        else:
            log.info(f"Loading requested processed graphs from {requested_graphs_path}")
            loaded_graphs = _torch_load(requested_graphs_path)
            resolved_graphs_path = requested_graphs_path

    if not loaded_graphs:
        raise RuntimeError("No graph windows were loaded for backend replay.")

    source = "replay_cache" if os.path.basename(resolved_graphs_path) == "replay_cache.pt" else "full_graphs"
    return [_prepare_graph(graph) for graph in loaded_graphs], resolved_graphs_path, source


def _load_real_model(model_path: str, in_channels: int):
    resolved_model_path = _resolve_path(model_path)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {resolved_model_path}. Run the baseline trainer first."
        )

    from src.models.gnn import build_model

    model = build_model(
        in_channels,
        task="node",
        hidden=128,
        num_classes=2,
        layers=3,
        dropout=0.3,
    )

    state = _torch_load(resolved_model_path)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for key, value in state.items():
        cleaned_state[key[7:] if key.startswith("module.") else key] = value

    model.load_state_dict(cleaned_state)
    model.eval()

    log.info(f"Loaded checkpoint from {resolved_model_path}")
    return model


def _load_adapter_config(adapter_path: str):
    resolved_adapter_path = _resolve_path(adapter_path)
    if not os.path.exists(resolved_adapter_path):
        return None, None

    from src.models.maml_trainer import load_adapter_artifact

    artifact = load_adapter_artifact(resolved_adapter_path)
    if artifact is None:
        return None, None
    return artifact.get("adapter_config"), resolved_adapter_path


def _build_demo_graphs():
    from torch_geometric.data import Data

    log.info("Generating dummy graphs for demo mode")
    dummy_graphs = []
    for i in range(300):
        n = 50
        x = torch.randn(n, 20)
        edge_index = torch.randint(0, n, (2, 100))
        label = torch.tensor([1 if i % 3 == 0 else 0], dtype=torch.long)
        node_y = label.repeat(n)
        dummy_graphs.append(Data(x=x, edge_index=edge_index, y=node_y, graph_y=label))
    return dummy_graphs


def _build_demo_agent():
    class DummyAgent:
        def __init__(self):
            self.model_version = 0
            self._counter = 0
            self.mutation_eng = type("obj", (object,), {"hypothesis_log": []})()

        def process_window(self, graph, window_id=0):
            self._counter += 1
            pred = 1 if window_id % 3 == 0 else 0
            conf = round(0.75 + np.random.uniform(-0.15, 0.2), 4)
            on_alert(ReplayAlert(
                timestamp=time.time(),
                window_id=window_id,
                prediction=pred,
                confidence=conf,
                model_version=self.model_version,
            ))

        def get_stats(self):
            return {
                "windows_processed": self._counter,
                "total_alerts": len(alert_store),
                "attack_alerts": sum(1 for a in alert_store if a["prediction"] == 1),
                "drift_events": len(drift_store),
                "model_version": self.model_version,
                "hypotheses_tried": 0,
                "hypotheses_accepted": 0,
            }

    return DummyAgent()


def replay_loop(delay: float = 0.3):
    global agent, graphs, is_running
    if agent is None or graphs is None:
        return

    is_running = True
    idx = 0
    log.info(f"Replay loop started over {len(graphs)} graphs")

    while is_running:
        try:
            graph = graphs[idx % len(graphs)]
            agent.process_window(graph, window_id=idx)
            idx += 1
            time.sleep(delay)
        except Exception as exc:
            log.error(f"Replay error: {exc}")
            time.sleep(1)


@app.on_event("startup")
async def startup():
    global agent, graphs

    alert_store.clear()
    drift_store.clear()

    graphs_path = os.environ.get("GRAPHS_PATH", "data/processed/graphs.pt")
    raw_csv_dir = os.environ.get("RAW_CSV_DIR", "data/raw")
    model_path = os.environ.get("MODEL_PATH", "experiments/checkpoints/best_gnn.pt")
    adapter_path = os.environ.get("ADAPTER_PATH", "experiments/checkpoints/best_maml_adapter.pt")
    prefer_cache = os.environ.get("USE_REPLAY_CACHE", "1") != "0"
    demo_mode = os.environ.get("NIDS_DEMO_MODE", "0") == "1"

    try:
        graphs, active_graphs_path, graph_source = _load_real_graphs(graphs_path, raw_csv_dir, prefer_cache=prefer_cache)
        in_channels = int(graphs[0].x.shape[1])
        model = _load_real_model(model_path, in_channels)
        try:
            from src.agent.agent import SelfEvolvingAgent
            from src.models.maml_trainer import MAMLTrainer

            adapter_config, resolved_adapter_path = _load_adapter_config(adapter_path)
            if adapter_config:
                maml_trainer = MAMLTrainer.from_config(model=model, config=adapter_config)
                log.info(f"Loaded adapter config from {resolved_adapter_path}")
            else:
                maml_trainer = MAMLTrainer(model=model, inner_lr=0.003, inner_steps=5)
                resolved_adapter_path = None
            agent = SelfEvolvingAgent(
                model=model,
                maml_trainer=maml_trainer,
                on_alert=on_alert,
                on_drift=on_drift,
            )
            runtime_mode = "real_adaptive"
            log.info("Backend initialized with self-evolving adaptive agent")
        except Exception as adaptive_exc:
            log.warning("Adaptive agent unavailable, falling back to replay-only agent: %s", adaptive_exc)
            agent = RealReplayAgent(model=model, on_alert=on_alert, on_drift=on_drift)
            runtime_mode = "real"

        runtime_info.update({
            "mode": runtime_mode,
            "graphs_path": active_graphs_path,
            "graphs_source": graph_source,
            "model_path": _resolve_path(model_path),
            "adapter_path": resolved_adapter_path,
            "graph_count": len(graphs),
        })
        log.info(f"Backend initialized in {runtime_mode.upper()} mode with {len(graphs)} graph windows")
    except Exception as exc:
        if not demo_mode:
            log.error("Real backend initialization failed", exc_info=exc)
            raise RuntimeError(
                "Backend startup requires real CICIDS2017 artifacts. "
                "Provide processed graphs/checkpoints or set NIDS_DEMO_MODE=1 for demo mode."
            ) from exc

        log.warning(f"Real backend initialization failed ({exc}) - demo mode enabled")
        graphs = _build_demo_graphs()
        agent = _build_demo_agent()
        runtime_info.update({
            "mode": "demo",
            "graphs_path": None,
            "graphs_source": "demo",
            "model_path": None,
            "adapter_path": None,
            "graph_count": len(graphs),
        })

    thread = threading.Thread(target=replay_loop, args=(0.3,), daemon=True)
    thread.start()


@app.on_event("shutdown")
async def shutdown():
    global is_running
    is_running = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        **runtime_info,
    }


@app.get("/stats")
def stats():
    if agent is None:
        return {
            "windows_processed": 0,
            "total_alerts": 0,
            "attack_alerts": 0,
            "drift_events": 0,
            "model_version": 0,
            "hypotheses_tried": 0,
            "hypotheses_accepted": 0,
        }
    return agent.get_stats()


@app.get("/alerts")
def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    attack_only: bool = False,
):
    data = alert_store
    if attack_only:
        data = [a for a in data if a["prediction"] == 1]
    return {"total": len(data), "alerts": data[skip: skip + limit]}


@app.get("/drift-events")
def get_drift_events():
    return {"total": len(drift_store), "events": drift_store}


@app.get("/model-metrics")
def model_metrics():
    if not alert_store:
        return {"metrics": []}

    window = 50
    metrics = []
    for i in range(window, len(alert_store), 10):
        chunk = alert_store[i - window: i]
        attack_rate = sum(1 for a in chunk if a["prediction"] == 1) / len(chunk)
        avg_conf = sum(a["confidence"] for a in chunk) / len(chunk)
        metrics.append({
            "window_id": chunk[-1]["window_id"],
            "timestamp": chunk[-1]["timestamp"],
            "attack_rate": round(attack_rate, 3),
            "avg_conf": round(avg_conf, 3),
        })

    return {"metrics": metrics}


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    log.info("WebSocket client connected")

    alert_cursor = len(alert_store)
    drift_cursor = len(drift_store)

    for record in alert_store[-20:]:
        try:
            await websocket.send_text(json.dumps(record))
        except Exception:
            return

    try:
        while True:
            await asyncio.sleep(0.5)
            sent_message = False

            new_alerts = alert_store[alert_cursor:]
            for record in new_alerts:
                await websocket.send_text(json.dumps(record))
                sent_message = True
            alert_cursor += len(new_alerts)

            new_drifts = drift_store[drift_cursor:]
            for record in new_drifts:
                await websocket.send_text(json.dumps(record))
                sent_message = True
            drift_cursor += len(new_drifts)

            if not sent_message:
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": time.time(),
                }))

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as exc:
        log.error(f"WebSocket error: {exc}")
