"""
Star-Graph GAT + Weighted Multi-Task BCE Clinical Outcome Predictor
====================================================================

Overview
--------
This script implements a single-index star-graph Graph Attention Network
(GATSingleIndex) trained with binary cross-entropy (BCE) on clinical admission
data. Each admission's ICD-10 diagnosis codes are converted to fixed-length
prefix tokens, embedded, and represented as token nodes connected through a
learnable CLS node in a star graph. Two additive graph-attention layers update
the CLS and token representations, after which the CLS node is used as the
pooled admission representation. A rho MLP maps this representation to a single
scalar logit score.

Training follows a two-stage weighted multi-task strategy. Stage 1 trains one
fresh single-task GAT model per clinical target to estimate validation BCE. These
Stage-1 validation BCEs are then converted into task weights for Stage 2 using a
log(2)-based improvement score. Tasks that are single-class in TRAIN are skipped
by zeroing their masks before Stage 2. The final Stage-2 model is a single
scalar-output model trained with normalized weighted masked BCE and selected by
lowest weighted validation BCE.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs and diagnosis ICD-10 CSV.gz files for the
    CORE dataset.
3.  Validate required admission columns, including hadm_id and all configured
    train/eval targets.
4.  Build label matrices Y (N, T) and validity masks M (N, T):
        - mortality, mortality_30d, and icu_transfer use non-missing labels.
        - long_stay validity is controlled by long_stay_defined.
5.  Detect single-class TRAIN targets and skip them by zeroing their TRAIN and
    VAL masks before Stage 2.
6.  Load diagnosis ICD-10 files in chunks of 2M rows and convert each ICD code to
    a normalized 4-character prefix token.
7.  Build a TRAIN-only vocabulary from ICD-10 prefix tokens:
        - PAD token has ID 0.
        - UNK token has ID 1.
        - Tokens are collected from TRAIN diagnoses only.
8.  Encode each admission's diagnosis token list to integer IDs:
        - Preserve duplicate diagnosis codes.
        - Preserve encounter/file order.
        - Truncate to MAX_CODES_PER_ADMISSION=256.
        - Use an empty sequence for admissions with no diagnosis codes.
9.  Create PyTorch datasets and padded dataloaders using collate_pad:
        - PAD ID is 0.
        - Minimum padded sequence length is 1.
10. Stage 1 — Per-target single-task BCE training:
        - Train a fresh GATSingleIndex model for each non-skipped target.
        - Use AdamW with LR=1e-3 for 10 epochs.
        - Compute masked BCE for the selected task only.
        - Select the best checkpoint by lowest per-task validation BCE.
        - Store best validation BCE for task-weight derivation.
11. Derive Stage-2 task weights from Stage-1 validation BCE:
        score_t  = max(log(2) - val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, eps))^alpha, 1/cap, cap)
    Skipped tasks receive weight 0. Active weights are normalized to mean 1.
    If all scores are zero, the script falls back to uniform active weights.
12. Stage 2 — Weighted multi-task BCE training:
        - Train one GATSingleIndex model that outputs a single scalar score.
        - Expand the scalar logit across all tasks for BCE computation.
        - Use normalized weighted masked BCE:
              weighted sum of task BCE losses / weighted sum of valid labels.
        - Select the best checkpoint by lowest weighted validation BCE.
13. Score the TEST split using the best Stage-2 scalar-output model.
14. Report Distance Correlation, and Mutual Information between the
    scalar test score and each binary test label.

Architecture: GATSingleIndex Star-Graph Attention Model
-------------------------------------------------------
- Embedding layer:
      vocab_size -> EMB_DIM=128, padding_idx=0
- Learnable CLS node embedding:
      one trainable vector of size EMB_DIM
- Graph structure:
      node 0 is CLS
      nodes 1..L are ICD-10 token nodes
      CLS attends to all valid nodes
      each token attends to itself and CLS
- StarGATLayer attention:
      Linear projection W without bias
      additive attention e_ij = LeakyReLU(a_l^T Wh_i + a_r^T Wh_j)
      invalid PAD token nodes are masked out for CLS attention
      invalid token outputs are zeroed after token update
- Two stacked StarGATLayer blocks:
      StarGATLayer -> ELU -> StarGATLayer -> ELU
- Pooling:
      use final CLS node representation H[:, 0, :]
- Layer normalization:
      LayerNorm(EMB_DIM)
- rho output network:
      Linear(128->128) -> ReLU -> Linear(128->128) -> ReLU -> Linear(128->1)
- Output:
      single scalar logit per admission with shape (B,)

Loss Functions
--------------
Single-task masked BCE:
    Computes BCEWithLogitsLoss for one selected task only, multiplied by that
    task's validity mask and averaged over valid examples.

Weighted multi-task masked BCE:
    Expands the scalar model output across all tasks, computes BCE for each
    task, applies task masks, weights each task by the Stage-2 weight vector,
    and normalizes by the weighted number of valid labels.

Evaluation Metrics
------------------
For each TEST target, the script computes metrics using only valid rows:
    - Distance correlation
    - Mutual information in nats
    
Mutual information uses mutual_info_classif on the z-scored scalar score within
valid rows only, with a fixed EVAL_SEED for reproducibility.

Key Configuration
-----------------
  SEED: controlled via SEED env var (default 1001); EVAL_SEED=12345 fixed
  PREFIX_LEN=4, EMB_DIM=128, BATCH_SIZE=256, EPOCHS=10
  LR=1e-3, WEIGHT_DECAY=0.0
  NUM_WORKERS=4, PIN_MEMORY=True
  MI_BASE_NEIGHBORS=5
  MAX_CODES_PER_ADMISSION=256
  CHUNK_DX=2_000_000
  WEIGHT_EPS=0.02, WEIGHT_ALPHA=0.25, WEIGHT_CAP=3.0

Expected Directory Layout
-------------------------
    <data_root>/
    └── core/
        ├── admissions_core_train.csv
        ├── admissions_core_val.csv
        ├── admissions_core_test.csv
        ├── diagnoses_icd10_core_train.csv.gz
        ├── diagnoses_icd10_core_val.csv.gz
        └── diagnoses_icd10_core_test.csv.gz

Required admission columns:
    hadm_id, mortality, mortality_30d, long_stay, long_stay_defined, icu_transfer

Required diagnosis columns:
    hadm_id, icd_code

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor torch

Running
-------
    python starGAT_mimiciv.py --data_root /path/to/final_datasets

Override training seed:
    SEED=42 python starGAT_mimiciv.py --data_root /path/to/final_datasets

Example:
    python starGAT_mimiciv.py --data_root ./data/final_datasets
"""

################################################################
# IMPORTS
################################################################

import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor
import math


################################################################
# CONFIG
################################################################

SEED = int(os.environ.get("SEED", 1001))   # training varies per run
EVAL_SEED = 12345                        # fixed for MI (and any subsampling, if added)
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)
PREFIX_LEN = 4              # ICD-10 prefix length used as token (e.g., "I509" from "I50.9")
EMB_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 4
PIN_MEMORY = True
MI_BASE_NEIGHBORS = 5       # for mutual_info_classif
MAX_CODES_PER_ADMISSION = 256
WEIGHT_EPS   = 0.02   # floor for score when computing weights (try 0.01–0.05)
WEIGHT_ALPHA = 0.25   # gentler than sqrt (sqrt would be 0.5)
WEIGHT_CAP   = 3.0    # max ratio cap (prevents domination)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to final_datasets directory containing core/admissions_core_train.csv, etc.",
    )
    return parser.parse_args()


ARGS = parse_args()
FINAL_ROOT = Path(ARGS.data_root)

DATASETS = {
    "CORE": {
        "adm_train": FINAL_ROOT / "core" / "admissions_core_train.csv",
        "adm_val":   FINAL_ROOT / "core" / "admissions_core_val.csv",
        "adm_test":  FINAL_ROOT / "core" / "admissions_core_test.csv",
        "dx_train":  FINAL_ROOT / "core" / "diagnoses_icd10_core_train.csv.gz",
        "dx_val":    FINAL_ROOT / "core" / "diagnoses_icd10_core_val.csv.gz",
        "dx_test":   FINAL_ROOT / "core" / "diagnoses_icd10_core_test.csv.gz",
        "train_targets": ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
        "eval_targets":  ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
    },
}

################################################################
# UTILITY HELPERS
################################################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_icd(code) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()

def to_prefix(code: str, k: int) -> str:
    c = norm_icd(code)
    if not c:
        return ""
    return c[:k]

def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def is_single_class_train(Y: np.ndarray, M: np.ndarray, j: int) -> bool:
    valid = M[:, j].astype(bool)
    if valid.sum() == 0:
        return True
    yy = Y[valid, j].astype(int)
    return np.unique(yy).size < 2

def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

################################################################
# Data loading & preprocessing 
################################################################

CHUNK_DX = 2_000_000  # add near config

def load_dx_prefix_map_chunked(dx_path: Path, prefix_len: int, chunksize: int = CHUNK_DX) -> dict[int, list[str]]:
    hadm_to_list: dict[int, list[str]] = {}
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=chunksize):
        hadm_arr = chunk["hadm_id"].astype(int).to_numpy()
        code_arr = chunk["icd_code"].to_numpy()

        for hadm, raw in zip(hadm_arr, code_arr):
            tok = to_prefix(raw, prefix_len)
            if tok == "":
                continue
            lst = hadm_to_list.get(hadm)
            if lst is None:
                hadm_to_list[hadm] = [tok]
            else:
                lst.append(tok)
    return hadm_to_list

# 1) vocab: add PAD + UNK
def build_vocab_from_train(dx_map):
    toks = sorted({t for lst in dx_map.values() for t in lst})
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, t in enumerate(toks, start=2):
        vocab[t] = i
    return vocab

def encode_hadm_tokens(adm_df: pd.DataFrame, dx_map: dict[int, list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    unk = vocab["<UNK>"]
    out = []
    for h in adm_df["hadm_id"].astype(int).tolist():
        toks = dx_map.get(int(h), [])

        # apples-to-apples: missing dx -> empty list
        if not toks:
            out.append(np.asarray([], dtype=np.int64))
            continue

        # keep duplicates + order (NO set(), NO sorting)
        ids = [vocab.get(t, unk) for t in toks]

        # truncate first 256 in encounter order
        if MAX_CODES_PER_ADMISSION is not None and len(ids) > MAX_CODES_PER_ADMISSION:
            ids = ids[:MAX_CODES_PER_ADMISSION]

        out.append(np.asarray(ids, dtype=np.int64))
    return out

def build_Y_and_mask(df: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Y:    float32 (N,T) with 0/1 targets (NaN filled with 0)
      Mask: float32 (N,T) where 1 indicates valid label for loss
    """
    N = len(df)
    T = len(targets)
    Y = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)

    for j, t in enumerate(targets):
        if t == "long_stay":
            require_cols(df, ["long_stay", "long_stay_defined"], "admissions")
            valid = df["long_stay_defined"].fillna(0).astype(int).to_numpy().astype(bool)
            y = df["long_stay"].fillna(0).astype(int).to_numpy()
            Y[:, j] = y.astype(np.float32)
            M[:, j] = valid.astype(np.float32)
        else:
            require_cols(df, [t], "admissions")
            s = pd.to_numeric(df[t], errors="coerce")
            valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()
            Y[:, j] = y.astype(np.float32)
            M[:, j] = valid.astype(np.float32)

    return Y, M


################################################################
# DATASET + COLLATE
################################################################

class TokenDataset(Dataset):
    def __init__(self, seqs: list[np.ndarray], Y: np.ndarray, M: np.ndarray):
        self.seqs = seqs
        self.Y = Y
        self.M = M

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.Y[idx], self.M[idx]

def collate_pad(batch, pad_id: int = 0):
    seqs, Y, M = zip(*batch)
    B = len(seqs)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    max_len = int(lengths.max().item()) if B > 0 else 0
    max_len = max(max_len, 1)  # <-- ADD THIS (apples-to-apples)

    X = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        X[i, :len(s)] = torch.from_numpy(s).long()

    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
    M = torch.tensor(np.stack(M, axis=0), dtype=torch.float32)
    return X, lengths, Y, M


################################################################
# MODEL: GAT (Star Graph) Single-Index Baseline
################################################################

class StarGATLayer(nn.Module):
    """
    Graph Attention on a star graph:
      - Node 0 is CLS
      - Nodes 1..L are tokens (some may be PAD)
      - CLS attends to all valid nodes
      - Each token attends to {itself, CLS}
    Uses additive attention: e_ij = LeakyReLU(a_l^T Wh_i + a_r^T Wh_j)
    """
    def __init__(self, dim: int, dropout: float = 0.0, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_l = nn.Parameter(torch.empty(dim))
        self.a_r = nn.Parameter(torch.empty(dim))
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        H:    (B, N, d) where N = 1 + L (CLS + tokens)
        mask: (B, N)    1 for valid nodes, 0 for PAD tokens (CLS should be 1)
        """
        B, N, d = H.shape
        Wh = self.W(H)  # (B, N, d)

        # Precompute projections for additive attention
        # s_i = a_l^T Wh_i, t_j = a_r^T Wh_j
        s = (Wh * self.a_l.view(1, 1, d)).sum(dim=-1)  # (B, N)
        t = (Wh * self.a_r.view(1, 1, d)).sum(dim=-1)  # (B, N)

        # -------------------------
        # CLS update: node 0 attends to all nodes
        # e_0j = LeakyReLU(s_0 + t_j)
        # -------------------------
        e_cls = self.leakyrelu(s[:, 0:1] + t)  # (B, N)

        # mask out invalid token nodes (keep CLS valid)
        # set invalid logits to -inf so softmax ignores them
        neg_inf = torch.finfo(e_cls.dtype).min
        e_cls = torch.where(mask > 0, e_cls, torch.full_like(e_cls, neg_inf))

        alpha_cls = torch.softmax(e_cls, dim=1)          # (B, N)
        alpha_cls = self.dropout(alpha_cls)
        h_cls_new = torch.sum(alpha_cls.unsqueeze(-1) * Wh, dim=1)  # (B, d)

        # -------------------------
        # Token updates: node i attends to {i, CLS}
        # e_ii = LeakyReLU(s_i + t_i)
        # e_i0 = LeakyReLU(s_i + t_0)
        # -------------------------
        if N > 1:
            Wh_tok = Wh[:, 1:, :]         # (B, L, d)
            s_tok  = s[:, 1:]             # (B, L)
            t_tok  = t[:, 1:]             # (B, L)
            t_cls  = t[:, 0:1]            # (B, 1)
            mask_tok = mask[:, 1:]        # (B, L)

            e_self = self.leakyrelu(s_tok + t_tok)       # (B, L)
            e_to_cls = self.leakyrelu(s_tok + t_cls)     # (B, L)

            # stack two neighbors: [self, CLS]
            e_pair = torch.stack([e_self, e_to_cls], dim=-1)  # (B, L, 2)

            # If token is invalid, force its attention to zero output later
            alpha_pair = torch.softmax(e_pair, dim=-1)        # (B, L, 2)
            alpha_pair = self.dropout(alpha_pair)

            # neighbor features: self uses Wh_tok, CLS uses Wh_cls broadcast
            Wh_cls = Wh[:, 0:1, :]  # (B, 1, d)
            h_tok_new = alpha_pair[..., 0:1] * Wh_tok + alpha_pair[..., 1:2] * Wh_cls  # (B, L, d)

            # zero out invalid tokens
            h_tok_new = h_tok_new * mask_tok.unsqueeze(-1)
            H_new = torch.cat([h_cls_new.unsqueeze(1), h_tok_new], dim=1)  # (B, N, d)
        else:
            H_new = h_cls_new.unsqueeze(1)

        return H_new


class GATSingleIndex(nn.Module):
    """
    Star-graph GAT over tokens + CLS, then rho MLP -> scalar logit.
    Interface matches DeepSetsSingleIndex: forward(x_ids, lengths) -> (B,)
    """
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # learnable CLS node embedding
        self.cls = nn.Parameter(torch.zeros(emb_dim))

        # 2-layer GAT
        self.gat1 = StarGATLayer(emb_dim, dropout=dropout)
        self.gat2 = StarGATLayer(emb_dim, dropout=dropout)

        self.act = nn.ELU()
        self.ln = nn.LayerNorm(emb_dim)

        # same rho head as DeepSets
        self.rho = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x_ids: (B, L) with PAD=0
        B, L = x_ids.shape

        tok = self.emb(x_ids)  # (B, L, d)

        # Build node tensor with CLS at position 0
        cls = self.cls.view(1, 1, -1).expand(B, 1, -1)  # (B, 1, d)
        H = torch.cat([cls, tok], dim=1)                # (B, 1+L, d)

        # Node mask: CLS is always valid; tokens valid if x_ids != 0
        tok_mask = (x_ids != 0).float()                 # (B, L)
        mask = torch.cat([torch.ones(B, 1, device=x_ids.device), tok_mask], dim=1)  # (B, 1+L)

        # GAT layers
        H = self.gat1(H, mask)
        H = self.act(H)
        H = self.gat2(H, mask)
        H = self.act(H)

        # Use CLS node as pooled representation
        pooled = H[:, 0, :]           # (B, d)
        pooled = self.ln(pooled)

        s = self.rho(pooled).squeeze(-1)  # (B,)
        return s


################################################################
# Loss functions (BCE variants)
################################################################

def masked_bce_loss_single_task(logits: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, task_idx: int) -> torch.Tensor:
    """
    logits: (B,) scalar s(x)
    Y:      (B,T)
    M:      (B,T) 0/1 mask
    task_idx: which task to compute loss for
    Loss: average BCE for this task only
    """
    y_t = Y[:, task_idx]
    m_t = M[:, task_idx]
    
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction="none")  # (B,)
    loss = loss * m_t
    
    denom = m_t.sum()
    if denom.item() <= 0:
        return logits.sum() * 0.0

    return loss.sum() / denom

def masked_multi_bce_loss_weighted(logits: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    NORMALIZED weighted masked BCE:
      per_elem = BCE(logits, Y) * M
      per_task_sum = sum_i per_elem(i,t)
      valid_counts = sum_i M(i,t)
      loss = sum_t w_t * per_task_sum[t] / sum_t w_t * valid_counts[t]
    """
    B, T = Y.shape
    logit_mat = logits.unsqueeze(1).expand(B, T)  # (B,T)

    per_elem = nn.functional.binary_cross_entropy_with_logits(logit_mat, Y, reduction="none")
    per_elem = per_elem * M

    per_task_sum = per_elem.sum(dim=0)        # (T,)
    valid_counts = M.sum(dim=0)               # (T,)

    num = (weights * per_task_sum).sum()
    den = (weights * valid_counts).sum()

    if den.item() <= 0:
        return logits.sum() * 0.0
    return num / den

################################################################
# TRAINING HELPERS
################################################################
@torch.no_grad()
def compute_task_loss(model, loader, task_idx: int) -> float:
    """Compute BCE loss for a specific task on a loader."""
    model.eval()
    losses = []
    for x_ids, lengths, Y, M in loader:
        x_ids = x_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        Y = Y.to(DEVICE)
        M = M.to(DEVICE)

        s = model(x_ids, lengths)
        loss = masked_bce_loss_single_task(s, Y, M, task_idx)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")

@torch.no_grad()
def compute_epoch_loss_weighted(model, loader, weights: torch.Tensor) -> float:
    model.eval()
    losses = []
    for x_ids, lengths, Y, M in loader:
        x_ids = x_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        Y = Y.to(DEVICE)
        M = M.to(DEVICE)

        s = model(x_ids, lengths)
        loss = masked_multi_bce_loss_weighted(s, Y, M, weights=weights)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")

@torch.no_grad()
def predict_scores(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for x_ids, lengths, _Y, _M in loader:
        x_ids = x_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        s = model(x_ids, lengths).detach().cpu().numpy()
        out.append(s)
    return np.concatenate(out, axis=0)

def train_single_task_bce(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_idx: int,
) -> tuple[dict, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x_ids, lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids, lengths)
            loss = masked_bce_loss_single_task(s, Y, M, task_idx)
            loss.backward()
            opt.step()

        val_loss = compute_task_loss(model, val_loader, task_idx)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, float(best_val)


################################################################
# Evaluation metrics 
################################################################

def pearson_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
   """
   Compute Pearson correlation between scalar score and binary outcome.
   """
   valid = valid_mask.astype(bool)
   if valid.sum() < 3:
       return 0.0
  
   yy = y[valid].astype(float)
   ss = score[valid].astype(float)
  
   # Check for constant values (no variance)
   if np.std(ss) < 1e-8 or np.std(yy) < 1e-8:
       return 0.0
  
   corr, _ = pearsonr(ss, yy)  # returns (correlation, p-value)
   return float(corr)


def dcor_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
   """
   Compute distance correlation between scalar score and binary outcome.
   """
   valid = valid_mask.astype(bool)
   if valid.sum() < 3:
       return 0.0
  
   yy = y[valid].astype(float)
   ss = score[valid].astype(float)
  
   dc = dcor.distance_correlation(ss, yy)
   return float(dc)


def mi_nats_masked(
    score: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    seed: int = EVAL_SEED,
    base_neighbors: int = MI_BASE_NEIGHBORS
) -> float:
    valid = valid.astype(bool)
    if valid.sum() < 3:
        return 0.0

    yy = y[valid].astype(int)
    if np.unique(yy).size < 2:
        return 0.0

    nn = int(min(base_neighbors, valid.sum() - 1))
    if nn < 1:
        return 0.0

    # z-score ONLY within valid rows (apples-to-apples with NHSIC scripts)
    s_valid = zscore_1d(score[valid]).reshape(-1, 1)
    return float(mutual_info_classif(s_valid, yy, random_state=seed, n_neighbors=nn)[0])



################################################################
# MAIN PIPELINE
################################################################

def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]

    print("\n" + "=" * 100)
    print(f"DATASET: {ds_name}")
    print("=" * 100)

    tr = pd.read_csv(cfg["adm_train"])
    va = pd.read_csv(cfg["adm_val"])
    te = pd.read_csv(cfg["adm_test"])

    train_targets = cfg["train_targets"]
    eval_targets  = cfg["eval_targets"]

    require_cols(tr, ["hadm_id"], f"{ds_name}/TRAIN")
    require_cols(va, ["hadm_id"], f"{ds_name}/VAL")
    require_cols(te, ["hadm_id"], f"{ds_name}/TEST")
    require_cols(tr, train_targets, f"{ds_name}/TRAIN")
    require_cols(va, train_targets, f"{ds_name}/VAL")
    require_cols(te, eval_targets,  f"{ds_name}/TEST")

    print(f"TRAIN n={len(tr):,} | VAL n={len(va):,} | TEST n={len(te):,}")

    # Build labels + masks
    Ytr, Mtr = build_Y_and_mask(tr, train_targets)
    Yva, Mva = build_Y_and_mask(va, train_targets)

    # -------------------------
    # SKIP TASKS THAT ARE SINGLE-CLASS IN TRAIN (apples-to-apples with HSIC)
    # -------------------------
    skipped = np.zeros(len(train_targets), dtype=bool)

    for j, t in enumerate(train_targets):
        if is_single_class_train(Ytr, Mtr, j):
            skipped[j] = True
            Mtr[:, j] = 0.0   # remove from Stage-2 TRAIN loss
            Mva[:, j] = 0.0   # remove from Stage-2 VAL loss (early stopping)


    # Diagnostics: positive rates on valid rows
    print("\nTRAIN label positive rates (valid-only):")
    for j, t in enumerate(train_targets):
        valid = Mtr[:, j].astype(bool)
        if valid.sum() == 0:
            print(f"  {t:16s}: n_valid=0")
        else:
            print(f"  {t:16s}: {float(Ytr[valid, j].mean()):.6f} (n_valid={int(valid.sum())})")

    # Load dx maps
    print("\nLoading diagnoses + building TRAIN vocab...")
    dx_tr = load_dx_prefix_map_chunked(cfg["dx_train"], PREFIX_LEN)
    dx_va = load_dx_prefix_map_chunked(cfg["dx_val"], PREFIX_LEN)
    dx_te = load_dx_prefix_map_chunked(cfg["dx_test"], PREFIX_LEN)

    vocab = build_vocab_from_train(dx_tr)
    vocab_size = len(vocab)   # (already includes PAD/UNK)
    print(f"Vocab size (TRAIN ICD10 prefixes): {len(vocab):,} (+PAD)")

    # Encode sequences
    tr_seqs = encode_hadm_tokens(tr, dx_tr, vocab)
    va_seqs = encode_hadm_tokens(va, dx_va, vocab)
    te_seqs = encode_hadm_tokens(te, dx_te, vocab)

    # Dataloaders
    train_ds = TokenDataset(tr_seqs, Ytr, Mtr)
    val_ds   = TokenDataset(va_seqs, Yva, Mva)

    # For test we don't need Y/M in loader, but keep placeholders
    Yte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    Mte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    test_ds  = TokenDataset(te_seqs, Yte0, Mte0)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )

    # =========================
    # STAGE 1: Single-task models
    # =========================
    print("\n=== Stage 1: Single-task BCE models ===")
    stage1_val = {}

    for j, t in enumerate(train_targets):
        if skipped[j]:
            print(f"  Skipping single-task model for {t}: single-class in TRAIN")
            stage1_val[t] = float("inf")   # recommended for skipped
            continue

        print(f"  Training single-task model for: {t}")
        model = GATSingleIndex(vocab_size=vocab_size, emb_dim=EMB_DIM, dropout=0.0).to(DEVICE)


        _, best_val = train_single_task_bce(model, train_loader, val_loader, j)
        print(f"    best VAL BCE: {best_val:.6f}")

        stage1_val[t] = best_val


    # =========================
    # Compute weights from Stage-1 VAL BCE (NEW: weighted rule)
    # =========================
    vals = np.array([float(stage1_val[t]) for t in train_targets], dtype=np.float64)

    # score_t = max(log(2) - val_bce_t, 0)
    LOG2 = float(np.log(2.0))
    score = np.maximum(LOG2 - vals, 0.0)

    # skipped tasks => score=0
    score[skipped] = 0.0
    score[~np.isfinite(score)] = 0.0

    max_score = float(score.max()) if score.size > 0 else 0.0

    eps   = WEIGHT_EPS
    alpha = WEIGHT_ALPHA
    cap   = WEIGHT_CAP

    if max_score <= 0.0:
        # if everything is "impossible", use uniform weights on active tasks
        weights = np.ones_like(score, dtype=np.float32)
    else:
        denom = np.maximum(score, eps)
        weights = (np.maximum(max_score, eps) / denom) ** alpha
        weights = np.clip(weights, 1.0 / cap, cap).astype(np.float32)

    # skipped => weight 0
    weights[skipped] = 0.0

    # normalize active weights to mean 1 (stabilizes objective scale)
    active = weights > 0
    if active.any():
        weights[active] = weights[active] / weights[active].mean()

    W_t = torch.tensor(weights, device=DEVICE, dtype=torch.float32)

    if float(W_t.sum().detach().cpu().item()) == 0.0:
        print("All Stage-2 weights are zero. Skipping Stage-2.")
        return

    print("\nDerived Stage-2 weights (after cap + mean-normalize):")
    for j, t in enumerate(train_targets):
        tag = " (SKIPPED)" if skipped[j] else ""
        print(f"  {t:16s}: w={float(weights[j]):.6f}{tag}")

    # =========================
    # STAGE 2: Weighted multi-task BCE
    # =========================
    print("\n=== Stage 2: Weighted multi-task BCE model (ONE scalar) ===")
    model = GATSingleIndex(vocab_size=vocab_size, emb_dim=EMB_DIM, dropout=0.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x_ids, lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids, lengths)
            loss = masked_multi_bce_loss_weighted(s, Y, M, weights=W_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(losses)) if losses else float("inf")
        va_loss = compute_epoch_loss_weighted(model, val_loader, weights=W_t)


        print(f"Epoch {ep:02d}/{EPOCHS} | train_BCE={tr_loss:.6f} | val_BCE={va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict scalar score on TEST
    score_test = predict_scores(model, test_loader)

    # Metrics on TEST vs eval_targets
    print("\nTEST Metrics (DistCorr, MI_nats) using GATSingleIndex 2-stage WEIGHTED scalar score s(x):")

    for t in eval_targets:
        if t == "long_stay":
            require_cols(te, ["long_stay", "long_stay_defined"], f"{ds_name}/TEST")
            valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            y = te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            s = pd.to_numeric(te[t], errors="coerce")
            valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()

        pearson = pearson_from_score(score_test, y, valid)
        dcorr = dcor_from_score(score_test, y, valid)
        mi = mi_nats_masked(score_test, y, valid)


        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )
    print("=" * 100)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    for ds in ["CORE"]:
        run_dataset(ds)