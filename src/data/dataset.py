"""
dataset.py - CICIDS2017 PyG dataset with sliding-window graph generation.

Usage:
    from src.data.dataset import CICIDS2017Dataset
    ds = CICIDS2017Dataset(root="data/processed", csv_dir="data/raw")
    ds.process()
    train_ds, val_ds, test_ds = ds.split()
"""

import glob
import json
import logging
import os
import random
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from src.data.graph_builder import FlowGraphBuilder

log = logging.getLogger(__name__)


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {path} with supported encodings")


def _torch_load_graphs(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


class CICIDS2017Dataset:
    """
    Loads all CICIDS2017 CSVs from csv_dir, slices them into fixed-size
    windows, builds a PyG graph per window, and saves to disk.
    """

    def __init__(
        self,
        root: str = "data/processed",
        csv_dir: str = "data/raw",
        window_size: int = 200,
        stride: int = 100,
        edge_strategy: str = "shared_ip",
    ):
        self.root = root
        self.csv_dir = csv_dir
        self.window_size = window_size
        self.stride = stride
        self.edge_strategy = edge_strategy
        self.graphs: list = []
        os.makedirs(root, exist_ok=True)

    def process(self, force: bool = False):
        save_path = os.path.join(self.root, "graphs.pt")
        builder_path = os.path.join(self.root, "builder.pkl")
        meta_path = os.path.join(self.root, "dataset_meta.json")
        replay_cache_path = os.path.join(self.root, "replay_cache.pt")

        if os.path.exists(save_path) and not force:
            log.info(f"Found cached graphs at {save_path}. Loading...")
            self.graphs = _torch_load_graphs(save_path)
            if not os.path.exists(meta_path):
                self._write_metadata(meta_path)
            if not os.path.exists(replay_cache_path):
                self._write_replay_cache(replay_cache_path)
            return self

        csv_files = glob.glob(os.path.join(self.csv_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.csv_dir}.\n"
                "Download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html\n"
                "Place the *_Flow.csv files in data/raw/"
            )

        log.info(f"Found {len(csv_files)} CSV files.")
        all_dfs = [_read_csv_with_fallback(path) for path in csv_files]
        full_df = pd.concat(all_dfs, ignore_index=True)
        log.info(f"Total rows: {len(full_df)}")

        train_end = int(len(full_df) * 0.7)
        builder = FlowGraphBuilder(window_size=self.window_size, edge_strategy=self.edge_strategy)
        builder.fit(full_df.iloc[:train_end])
        builder.save(builder_path)

        self.graphs = []
        for start in range(0, len(full_df) - self.window_size, self.stride):
            window = full_df.iloc[start : start + self.window_size]
            try:
                self.graphs.append(builder.build(window))
            except Exception as exc:
                log.warning(f"Skipping window at {start}: {exc}")

        log.info(f"Built {len(self.graphs)} graphs.")
        torch.save(self.graphs, save_path)
        log.info(f"Saved -> {save_path}")
        self._write_metadata(meta_path)
        self._write_replay_cache(replay_cache_path)
        return self

    def _write_metadata(self, meta_path: str):
        meta = {
            "graph_count": len(self.graphs),
            "window_size": self.window_size,
            "stride": self.stride,
            "edge_strategy": self.edge_strategy,
            "feature_dim": int(self.graphs[0].x.shape[1]) if self.graphs else 0,
            "cache_files": {
                "graphs": os.path.abspath(os.path.join(self.root, "graphs.pt")),
                "replay_cache": os.path.abspath(os.path.join(self.root, "replay_cache.pt")),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        log.info(f"Saved metadata -> {meta_path}")

    def _write_replay_cache(self, replay_cache_path: str, max_graphs: int = 1500):
        if not self.graphs:
            return
        if len(self.graphs) <= max_graphs:
            replay_graphs = self.graphs
        else:
            step = max(1, len(self.graphs) // max_graphs)
            replay_graphs = self.graphs[::step][:max_graphs]
        torch.save(replay_graphs, replay_cache_path)
        log.info(f"Saved replay cache -> {replay_cache_path} ({len(replay_graphs)} graphs)")

    def split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        stratify: bool = True,
    ) -> Tuple["GraphSubset", "GraphSubset", "GraphSubset"]:
        assert self.graphs, "Call .process() first."

        if not stratify:
            n = len(self.graphs)
            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))
            return (
                GraphSubset(self.graphs[:train_end], name="train"),
                GraphSubset(self.graphs[train_end:val_end], name="val"),
                GraphSubset(self.graphs[val_end:], name="test"),
            )

        rng = random.Random(seed)
        benign_graphs = []
        attack_graphs = []

        for graph in self.graphs:
            if int((graph.y == 1).any().item()) == 1:
                attack_graphs.append(graph)
            else:
                benign_graphs.append(graph)

        rng.shuffle(benign_graphs)
        rng.shuffle(attack_graphs)

        def split_bucket(bucket: list):
            n = len(bucket)
            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))
            return bucket[:train_end], bucket[train_end:val_end], bucket[val_end:]

        train_benign, val_benign, test_benign = split_bucket(benign_graphs)
        train_attack, val_attack, test_attack = split_bucket(attack_graphs)

        train_graphs = train_benign + train_attack
        val_graphs = val_benign + val_attack
        test_graphs = test_benign + test_attack

        rng.shuffle(train_graphs)
        rng.shuffle(val_graphs)
        rng.shuffle(test_graphs)

        return (
            GraphSubset(train_graphs, name="train"),
            GraphSubset(val_graphs, name="val"),
            GraphSubset(test_graphs, name="test"),
        )

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class GraphSubset(TorchDataset):
    def __init__(self, graphs: list, name: str = ""):
        self.graphs = graphs
        self.name = name

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def attack_ratio(self) -> float:
        total = attack = 0
        for graph in self.graphs:
            total += graph.y.numel()
            attack += graph.y.sum().item()
        return attack / total if total else 0.0

    def summary(self) -> str:
        n_graphs = len(self.graphs)
        avg_nodes = sum(graph.num_nodes for graph in self.graphs) / n_graphs if n_graphs else 0
        avg_edges = sum(graph.edge_index.shape[1] for graph in self.graphs) / n_graphs if n_graphs else 0
        graph_attack_ratio = (
            sum(1 for graph in self.graphs if int((graph.y == 1).any().item()) == 1) / n_graphs
            if n_graphs
            else 0
        )
        return (
            f"[{self.name}] {n_graphs} graphs | "
            f"avg nodes: {avg_nodes:.0f} | avg edges: {avg_edges:.0f} | "
            f"node attack ratio: {self.attack_ratio():.2%} | "
            f"graph attack ratio: {graph_attack_ratio:.2%}"
        )
