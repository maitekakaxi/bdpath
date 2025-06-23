# =========================================
# path_time_diffusion.py (Step‑1 + Step‑2)
# -----------------------------------------
#   * PathTimeTokenizer – int‑ID & log‑bin time bucket
#   * PathTimeDataset   – produces fixed‑length blocks ready for Block Diffusion
#   * BlockDiffusionModel – minimal dual‑channel Transformer backbone
#       ‑ supports unconditional forward returning (logits_path, logits_time)
#   * debug CLI tests dataset & model fwd pass
# -------------------------------------------------
from __future__ import annotations
import math, argparse, pathlib, pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -----------------------------
# Tokenizer
# -----------------------------
class PathTimeTokenizer:
    def __init__(self,
                 max_path_id: int = 8263,
                 num_time_bins: int = 256,
                 max_time: float = 3600.0):
        self.num_time_bins = num_time_bins
        self.max_time = max_time
        self.time_bins = np.logspace(np.log10(1.0), np.log10(max_time), num=255)
        self.road_offset = 0
        self.time_offset = self.road_offset + max_path_id + 1
        self.vocab_size = self.time_offset + num_time_bins

    # ------------- encode / decode -------------
    def _time_to_bucket(self, t: float) -> int:
        idx = np.searchsorted(self.time_bins, t, side="right")
        return int(min(idx, self.num_time_bins - 1))

    def _bucket_to_time(self, idx: int) -> float:
        if idx == 0:
            return self.time_bins[0]
        if idx >= len(self.time_bins):
            return self.time_bins[-1]
        return float((self.time_bins[idx-1] + self.time_bins[idx]) / 2)

    def encode_pair(self, road_ids: List[int], dt: List[float]) -> Tuple[List[int], List[int]]:
        assert len(road_ids) == len(dt)
        roads = [self.road_offset + r for r in road_ids]
        times = [self.time_offset + self._time_to_bucket(x) for x in dt]
        return roads, times

    def decode_pair(self, roads: List[int], times: List[int]):
        roads = [r -self.road_offset for r in roads]
        times = [self._bucket_to_time(t-self.time_offset) for t in times]
        return roads, times


class PathTimeDataset(Dataset):
    def __init__(self, paths_csv: str, times_csv: str,
                 tokenizer: PathTimeTokenizer,
                 block_size: int = 16,
                 normalize_time: bool = True):
        self.tok = tokenizer
        self.block = block_size
        self.normalize_time = normalize_time
        block_size = int(block_size) if isinstance(block_size, str) else block_size

        df_p = pd.read_csv(paths_csv, header=None, dtype=str)
        df_t = pd.read_csv(times_csv, header=None, dtype=str)

        self.path_blocks, self.time_blocks, self.attn = [], [], []
        all_dts = []  # 用于统计 mean/std
        skipped = 0

        for pstr, tstr in zip(df_p[0], df_t[0]):
            if pd.isna(pstr) or pd.isna(tstr):
                skipped += 1; continue
            roads = pstr.strip().split(); times = tstr.strip().split()
            if roads and roads[-1] == '0':
                roads = roads[:-1]
            if not roads or len(times) != len(roads)+1:  # first ts absolute
                skipped += 1; continue
            try:
                road_ids = list(map(int, roads))
                dts = list(map(float, times[1:]))  # delta_t
            except ValueError:
                skipped += 1; continue

            all_dts.extend(dts)

            # block 切分（不 encode time 为 token）
            nblk = len(road_ids) // block_size
            for i in range(nblk):
                seg = slice(i * block_size, (i + 1) * block_size)
                r_seg = road_ids[seg]
                t_seg = dts[seg]
                self.path_blocks.append(torch.tensor(r_seg, dtype=torch.long))
                self.time_blocks.append(torch.tensor(t_seg, dtype=torch.float32))
                self.attn.append(torch.ones(block_size, dtype=torch.long))

        #self.time_mean = np.mean(all_dts)
        #self.time_std = np.std(all_dts) if np.std(all_dts) > 1e-6 else 1.0
        #print(f"[Dataset] blocks={len(self.path_blocks)}  skipped={skipped}")
        #print(f"[Time Normalize] mean={self.time_mean:.4f}, std={self.time_std:.4f}")
        if normalize_time:
            # Compute log domain mean/std
            flat_times = torch.cat(self.time_blocks)
            eps = 1e-2
            log_times = torch.log(flat_times + eps)
            self.log_time_mean = log_times.mean()
            self.log_time_std = log_times.std()
    def __len__(self):
        return len(self.path_blocks)

    def __getitem__(self, idx):
        raw_time = self.time_blocks[idx]  # Tensor of shape [L]
        eps = 1e-2
        log_time = torch.log(raw_time + eps)

        if self.normalize_time:
            log_time = (log_time - self.log_time_mean) / self.log_time_std

        return {
            "path_ids": self.path_blocks[idx],
            "time_values": log_time,
            "attn_mask": self.attn[idx],
        }

    def get_time_stats(self):
        return self.log_time_mean.item(), self.log_time_std.item()
