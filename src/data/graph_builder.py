"""
graph_builder.py — Converts CICIDS2017 flow CSVs into PyTorch Geometric Data objects.

Each graph = one time window of traffic.
  Nodes  : individual flows (1 row = 1 node)
  Edges  : flows sharing a source IP, destination IP, or destination port
  Labels : 0=BENIGN, 1..N=attack class (binary + multiclass stored)
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging, pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

try:
    from torch_geometric.data import Data
except ImportError:
    # Stub so the file is importable without torch-geometric installed
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

FEATURE_COLS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
    "URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio",
    "Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
]
BENIGN_LABEL = "BENIGN"


class FlowGraphBuilder:
    """
    Fit on training data once, then call build() on any window DataFrame.

    Example:
        builder = FlowGraphBuilder(window_size=200, edge_strategy="both")
        builder.fit(train_df)
        graph = builder.build(window_df)   # returns PyG Data object
        builder.save("builder_state.pkl")
    """

    def __init__(self, window_size: int = 200, edge_strategy: str = "shared_ip"):
        self.window_size = window_size
        self.edge_strategy = edge_strategy   # "shared_ip" | "shared_port" | "both"
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "FlowGraphBuilder":
        df = self._clean(df)
        self.scaler.fit(self._features(df))
        self.label_encoder.fit(df["Label"].values)
        self._fitted = True
        log.info(f"Fitted on {len(df)} rows | classes: {list(self.label_encoder.classes_)}")
        return self

    # ---------------------------------------------------------------- build
    def build(self, df: pd.DataFrame) -> Data:
        assert self._fitted, "Call .fit() on training data first."
        df = self._clean(df).head(self.window_size).reset_index(drop=True)
        n = len(df)

        x = torch.tensor(
            self.scaler.transform(self._features(df)), dtype=torch.float
        )

        raw = self.label_encoder.transform(df["Label"].values)
        benign_idx = list(self.label_encoder.classes_).index(BENIGN_LABEL)
        y_binary = torch.tensor((raw != benign_idx).astype(int), dtype=torch.long)
        y_multi  = torch.tensor(raw, dtype=torch.long)

        edge_index = self._edges(df, n)
        return Data(x=x, edge_index=edge_index, y=y_binary, y_multi=y_multi, num_nodes=n)

    # ------------------------------------------------------------ internals
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        if "Label" not in df.columns and " Label" in df.columns:
            df.rename(columns={" Label": "Label"}, inplace=True)
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        if len(cols) < 10:
            raise ValueError(f"Only {len(cols)} feature columns found. Check CSV format.")
        return df[cols].values.astype(np.float32)

    def _edges(self, df: pd.DataFrame, n: int):
        srcs, dsts = [], []

        def add_group(col):
            if col not in df.columns:
                return
            for idxs in df.groupby(col).groups.values():
                idxs = list(idxs)
                if len(idxs) > 100:   # skip degenerate hubs
                    continue
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        srcs.extend([idxs[i], idxs[j]])
                        dsts.extend([idxs[j], idxs[i]])

        if self.edge_strategy in ("shared_ip", "both"):
            add_group("Source IP")
            add_group("Destination IP")
        if self.edge_strategy in ("shared_port", "both"):
            add_group("Destination Port")

        if not srcs:                       # fallback: self-loops only
            srcs = dsts = list(range(n))

        ei = torch.tensor([srcs, dsts], dtype=torch.long)
        return torch.unique(ei, dim=1)

    # ---------------------------------------------------------- persistence
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "le": self.label_encoder,
                         "window_size": self.window_size,
                         "edge_strategy": self.edge_strategy}, f)
        log.info(f"Builder saved → {path}")

    def load(self, path: str) -> "FlowGraphBuilder":
        with open(path, "rb") as f:
            s = pickle.load(f)
        self.scaler = s["scaler"]
        self.label_encoder = s["le"]
        self.window_size = s.get("window_size", self.window_size)
        self.edge_strategy = s.get("edge_strategy", self.edge_strategy)
        self._fitted = True
        return self
