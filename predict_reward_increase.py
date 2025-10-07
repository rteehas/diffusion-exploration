"""
RewardIncreasePredictor
- Uses the Stable Diffusion (CLIP) text encoder from a HF repo to embed prompts
- Concatenates text embedding with a learned projection of a fixed-length reward
  history vector
- Feeds the concatenated vector through an MLP to predict the next H reward deltas

Requirements:
  pip install torch transformers
Optionally (for testing tokenization parity with SD1.x):
  pip install diffusers

Notes
- Loads tokenizer and CLIPTextModel from a Stable Diffusion repo via subfolders
  (e.g., "runwayml/stable-diffusion-v1-5": subfolder="tokenizer", "text_encoder").
- History length L and prediction horizon H are configurable.
- Text encoder can be frozen or fine-tuned.
- Minimal, framework-agnostic trainer stub included for reference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from regression import get_all_data
import wandb
# ------------------------------
# Utilities
# ------------------------------
import os
from pathlib import Path
import matplotlib.pyplot as plt
import copy

def plot_loss_curves(
    losses_by_timestep: dict,
    save_dir: Union[str, os.PathLike],
    combined_filename: str = "loss_curves_combined.png",
    dpi: int = 150,
):
    """
    Plot per-timestep loss curves and a combined subplot figure.

    losses_by_timestep: dict[int, Sequence[float]]
        Maps timestep t -> iterable of length K with the loss value at each GD step 1..K
    save_dir: directory to save outputs into. Created if not present.
    combined_filename: name of the combined figure file.

    Saves:
      - One PNG per timestep: loss_t{t}.png
      - One combined PNG with subplots.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Individual figures ---
    filepaths = []
    for t, losses in sorted(losses_by_timestep.items()):
        xs = list(range(1, len(losses) + 1))
        fig = plt.figure()
        plt.plot(xs, losses)
        plt.xlabel("GD step")
        plt.ylabel("Loss")
        plt.title(f"Loss curve @ timestep t={t}")
        out = save_path / f"loss_t{t}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        filepaths.append(out)

    # --- Combined subplots ---
    T = len(losses_by_timestep)
    if T == 0:
        return []

    # Choose a near-square grid
    import math
    ncols = math.ceil(math.sqrt(T))
    nrows = math.ceil(T / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    # Normalize axes to iterable
    if isinstance(axes, plt.Axes):
        axes = [[axes]]
    elif isinstance(axes, (list, tuple)):
        # already a 1D list of axes in some matplotlib versions
        if all(isinstance(a, plt.Axes) for a in axes):
            axes = [axes]

    idx = 0
    sorted_items = sorted(losses_by_timestep.items())
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c] if isinstance(axes[r], (list, tuple)) else axes[r,c]
            if idx < T:
                t, losses = sorted_items[idx]
                xs = list(range(1, len(losses) + 1))
                ax.plot(xs, losses)
                ax.set_title(f"t={t}")
                ax.set_xlabel("GD step")
                ax.set_ylabel("Loss")
            else:
                ax.axis("off")
            idx += 1

    combined_out = save_path / combined_filename
    fig.tight_layout()
    fig.savefig(combined_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return [str(p) for p in filepaths] + [str(combined_out)]

def plot_rollout_mse_curves(
    mse_by_timestep: dict,
    save_dir: Union[str, os.PathLike],
    combined_filename: str = "rollout_mse_combined.png",
    dpi: int = 150,
):
    """
    Plot per-timestep rollout MSE curves ("mse@k") and a combined subplot figure.

    mse_by_timestep: dict[int, Sequence[float]]
        Maps timestep t -> iterable with mse@k for k=1..L_val
    save_dir: directory to save outputs into. Created if not present.

    Saves:
      - One PNG per timestep: mse_at_k_t{t}.png
      - One combined PNG with subplots.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Individual figures ---
    filepaths = []
    for t, mse_list in sorted(mse_by_timestep.items()):
        xs = list(range(1, len(mse_list) + 1))
        fig = plt.figure()
        plt.plot(xs, mse_list)
        plt.xlabel("Rollout step k")
        plt.ylabel("MSE@k")
        plt.title(f"Rollout MSE@k @ timestep t={t}")
        out = save_path / f"mse_at_k_t{t}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        filepaths.append(out)

    # --- Combined subplots ---
    T = len(mse_by_timestep)
    if T == 0:
        return []

    import math
    ncols = math.ceil(math.sqrt(T))
    nrows = math.ceil(T / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    if isinstance(axes, plt.Axes):
        axes = [[axes]]
    elif isinstance(axes, (list, tuple)):
        if all(isinstance(a, plt.Axes) for a in axes):
            axes = [axes]

    idx = 0
    sorted_items = sorted(mse_by_timestep.items())
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c] if isinstance(axes[r], (list, tuple)) else axes[r,c]
            if idx < T:
                t, mse_list = sorted_items[idx]
                xs = list(range(1, len(mse_list) + 1))
                ax.plot(xs, mse_list)
                ax.set_title(f"t={t}")
                ax.set_xlabel("Rollout step k")
                ax.set_ylabel("MSE@k")
            else:
                ax.axis("off")
            idx += 1

    combined_out = save_path / combined_filename
    fig.tight_layout()
    fig.savefig(combined_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return [str(p) for p in filepaths] + [str(combined_out)]

def plot_mse_at_k_over_time(
    mse_by_timestep: dict,
    k: int,
    save_dir: Union[str, os.PathLike],
    filename: Optional[str] = None,
    dpi: int = 150,
    history=None
):
    """
    Plot a single curve: for a fixed rollout step k, show MSE@k across timesteps t.

    Args
      mse_by_timestep: dict[int, Sequence[float]] mapping t -> [mse@1, mse@2, ...]
      k: 1-indexed rollout step to extract (mse@k)
      save_dir: directory to save output figure
      filename: override output filename; default: f"mse_at_k_{k}_over_time.png"

    Returns
      str path to saved PNG
    """
    assert k >= 1, "k is 1-indexed and must be >= 1"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"mse_at_k_{k}_over_time.png"

    # Collect (t, mse@k) pairs; skip timesteps without k steps of rollout
    ts = []
    ys = []
    for t, mse_list in sorted(mse_by_timestep.items()):
        if len(mse_list) >= k:
            ts.append(t)
            ys.append(float(mse_list[k - 1]))

    fig = plt.figure()
    plt.plot(list(range(len(ys))), ys)
    if history is not None:
        per_timestep_mse1, mse_by_timestep_baseline = baseline_mean_mse1(history, B=11)
        ts_baseline = []
        ys_baseline = []
        for t, mse_list in sorted(mse_by_timestep_baseline.items()):
            ts_baseline.append(t)
            ys_baseline.append(float(mse_list[k - 1]))
        
        plt.plot(list(range(len(ys_baseline))), ys_baseline, label="mean prediction baseline")
    plt.xlabel("Timestep t")
    plt.ylabel(f"MSE@{k}")
    plt.title(f"MSE at rollout step k={k} across timesteps")
    if history is not None:
        plt.legend()

    out = save_path / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if history is not None:
        deltas = []
        for (y, y_baseline) in zip(ys, ys_baseline):
            deltas.append(y_baseline - y)
        
        plt.cla()
        plt.clf()
        fig = plt.figure()
        plt.plot(list(range(len(deltas))), deltas)
        plt.xlabel("Timestep t")
        plt.ylabel(f"Delta MSE@{k}")
        plt.title(f"Mean Prediction MSE - Model MSE at rollout step k={k}")
        delta_filename = "delta_" + filename
        fig.savefig(save_path / delta_filename,dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return str(out)


def mean_pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool non-pad tokens from CLIP text hidden states.
    hidden: (B, T, D)
    attention_mask: (B, T) with 1 for real tokens, 0 for pad
    returns: (B, D)
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------
# Core model
# ------------------------------

@dataclass
class PredictorConfig:
    sd_repo_id: str = "runwayml/stable-diffusion-v1-5"
    freeze_text_encoder: bool = True
    max_text_tokens: int = 77
    history_len: int = 32  # L
    horizon: int = 8       # H
    reward_proj_hidden: Tuple[int, ...] = (128, 128)
    head_hidden: Tuple[int, ...] = (512, 256)
    dropout: float = 0.1


class RewardIncreasePredictor(nn.Module):
    """Predicts the next H reward increases given a prompt and L-step history.

    Inputs
    - prompts: List[str]  (via forward_from_text) or tokenized tensors (via forward)
    - reward_history: Float tensor of shape (B, L), ordered oldest->most recent

    Output
    - next_deltas: Float tensor of shape (B, H), for steps T_k+1 ... T_k+H
    """
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        # Load tokenizer + text encoder from SD repo (subfolders)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_repo_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.sd_repo_id, subfolder="text_encoder")

        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        text_width = self.text_encoder.config.hidden_size  # typically 768 for SD1.x

        # Project reward history (L,) -> r_feat
        self.reward_proj = MLP(cfg.history_len, cfg.reward_proj_hidden, out_dim=256, dropout=cfg.dropout)

        # Optional projection for text embedding (identity-sized if desired)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_width),
            nn.Linear(text_width, 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Final MLP head over [text_feat || reward_feat]
        head_in = 512 + 256
        self.head = MLP(head_in, cfg.head_hidden, out_dim=cfg.horizon, dropout=cfg.dropout)

    # --------------------------
    # Text encoding helpers
    # --------------------------
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # CLIPTextModel outputs: last_hidden_state (B, T, D)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool_hidden(out.last_hidden_state, attention_mask)
        return self.text_proj(pooled)

    def tokenize(self, prompts: List[str], device: Optional[torch.device] = None):
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_tokens,
            return_tensors="pt",
        )
        if device is not None:
            toks = {k: v.to(device) for k, v in toks.items()}
        return toks

    # --------------------------
    # Forwards
    # --------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reward_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids: (B, T), attention_mask: (B, T)
        reward_history: (B, L)
        returns: (B, H)
        """
        assert reward_history.ndim == 2 and reward_history.size(1) == self.cfg.history_len, \
            f"reward_history must be (B,{self.cfg.history_len})"

        text_feat = self.encode_text(input_ids, attention_mask)  # (B, 512)
        r_feat = self.reward_proj(reward_history)                # (B, 256)
        x = torch.cat([text_feat, r_feat], dim=-1)              # (B, 768)
        return self.head(x)                                     # (B, H)

    @torch.no_grad()
    def forward_from_text(self, prompts: List[str], reward_history: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        toks = self.tokenize(prompts, device=device)
        return self.forward(toks["input_ids"], toks["attention_mask"], reward_history.to(device))

class RewardIncreasePredictorTextOnly(nn.Module):
    """Predicts the next H reward increases given a prompt and L-step history.

    Inputs
    - prompts: List[str]  (via forward_from_text) or tokenized tensors (via forward)
    - reward_history: Float tensor of shape (B, L), ordered oldest->most recent

    Output
    - next_deltas: Float tensor of shape (B, H), for steps T_k+1 ... T_k+H
    """
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        # Load tokenizer + text encoder from SD repo (subfolders)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_repo_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.sd_repo_id, subfolder="text_encoder")

        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        text_width = self.text_encoder.config.hidden_size  # typically 768 for SD1.x


        self.head = MLP(text_width, cfg.head_hidden, out_dim=cfg.horizon, dropout=cfg.dropout)

    # --------------------------
    # Text encoding helpers
    # --------------------------
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # CLIPTextModel outputs: last_hidden_state (B, T, D)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool_hidden(out.last_hidden_state, attention_mask)
        return pooled

    def tokenize(self, prompts: List[str], device: Optional[torch.device] = None):
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_tokens,
            return_tensors="pt",
        )
        if device is not None:
            toks = {k: v.to(device) for k, v in toks.items()}
        return toks

    # --------------------------
    # Forwards
    # --------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids: (B, T), attention_mask: (B, T)
        reward_history: (B, L)
        returns: (B, H)
        """
        text_feat = self.encode_text(input_ids, attention_mask)  # (B, 512)
        return self.head(text_feat)                                     # (B, H)

    @torch.no_grad()
    def forward_from_text(self, prompts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        toks = self.tokenize(prompts, device=device)
        return self.forward(toks["input_ids"], toks["attention_mask"])


class RewardIncreasePredictorSeq(nn.Module):
    """Predicts next H reward deltas using a progressively growing history.

    Uses a sequence model over the reward history so B, B+1, B+2, ... are all valid
    without changing tensor shapes. Supports GRU (default) or TransformerEncoder.
    """
    def __init__(self, cfg: PredictorConfig, rnn_hidden: int = 256, use_transformer: bool = False, n_layers: int = 2, n_heads: int = 4, tfm_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.use_transformer = use_transformer
        # Load SD text encoder/tokenizer as before
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_repo_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.sd_repo_id, subfolder="text_encoder")
        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        text_width = self.text_encoder.config.hidden_size

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_width),
            nn.Linear(text_width, 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # --- Reward sequence encoder ---
        if not use_transformer:
            # GRU over scalar reward deltas
            self.rnn = nn.GRU(input_size=1, hidden_size=rnn_hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
            r_feat_dim = rnn_hidden
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=rnn_hidden, nhead=n_heads, dim_feedforward=tfm_ff, dropout=dropout, batch_first=True)
            self.rnn_in = nn.Linear(1, rnn_hidden)
            self.pos = PositionalEncoding(rnn_hidden, dropout=dropout)
            self.tfm = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            r_feat_dim = rnn_hidden

        # Final head
        self.head = MLP(512 + r_feat_dim, cfg.head_hidden, out_dim=cfg.horizon, dropout=cfg.dropout)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool_hidden(out.last_hidden_state, attention_mask)
        return self.text_proj(pooled)

    def tokenize(self, prompts: List[str], device: Optional[torch.device] = None):
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_tokens,
            return_tensors="pt",
        )
        if device is not None:
            toks = {k: v.to(device) for k, v in toks.items()}
        return toks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reward_history: torch.Tensor,
        reward_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        reward_history: (B, T) variable T allowed (padded on the right if batching multiple T)
        reward_mask:   (B, T) with 1 for valid, 0 for pad. If None, assumes all valid.
        Returns: (B, H)
        """
        B, T = reward_history.shape
        text_feat = self.encode_text(input_ids, attention_mask)  # (B, 512)

        x = reward_history.unsqueeze(-1)  # (B, T, 1)
        if reward_mask is None:
            reward_mask = torch.ones(B, T, device=reward_history.device, dtype=torch.bool)

        if not self.use_transformer:
            # Pack sequences for GRU
            lengths = reward_mask.sum(dim=1).to(torch.int64).cpu()
            lengths_sorted, idx = torch.sort(lengths, descending=True)
            x_sorted = x[idx]
            packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted, batch_first=True, enforce_sorted=True)
            _, h_n = self.rnn(packed)  # (n_layers, B, hidden)
            h_last = h_n[-1]  # (B, hidden)
            inv_idx = torch.argsort(idx)
            r_feat = h_last[inv_idx]
        else:
            # Transformer with key padding mask (True = pad)
            key_padding_mask = ~reward_mask  # (B, T)
            z = self.rnn_in(x)
            z = self.pos(z)
            z = self.tfm(z, src_key_padding_mask=key_padding_mask)
            idx_last = reward_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
            r_feat = z[torch.arange(B, device=z.device), idx_last]

        feat = torch.cat([text_feat, r_feat], dim=-1)
        return self.head(feat)

    @torch.no_grad()
    def forward_from_text(self, prompts: List[str], reward_history: torch.Tensor, reward_mask: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        toks = self.tokenize(prompts, device=device)
        return self.forward(toks["input_ids"], toks["attention_mask"], reward_history.to(device), None if reward_mask is None else reward_mask.to(device))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)

def baseline_mean_mse1(
    rewards: torch.Tensor,
    B: int,
) -> Tuple[dict, dict]:
    """
    Compute MSE@1 for a simple baseline that predicts, for each prompt p at time t,
    the mean of that prompt's past reward deltas over the prefix [0..t-1].

    Args
      rewards: (N, P) tensor of reward increases
      B: initial prefix length; first evaluated target is t=B
      S: number of consecutive timesteps to evaluate (defaults to N - B)

    Returns
      per_timestep_mse1: {t: float}
      mse_by_timestep_like: {t: [float]}  # same schema as model's mse@k dicts
    """
    assert rewards.ndim == 2, "rewards must be (N, P)"
    N, P = rewards.shape
    assert 1 <= B <= N - 1, "B must be in [1, N-1]"

    per_timestep_mse1 = {}
    mse_by_timestep_like = {}
    for t in range(B-1, N):
        hist = rewards[:t, :]  # (t, P)
        pred = hist.mean(dim=0)  # (P,)
        gt = rewards[t, :]       # (P,)
        mse = torch.mean((pred - gt) ** 2).item()
        per_timestep_mse1[int(t)] = float(mse)
        mse_by_timestep_like[int(t)] = [float(mse)]  # so it can plug into plot_rollout_mse_curves

    return per_timestep_mse1, mse_by_timestep_like


# ------------------------------
# Block-wise training with progressive validation
# ------------------------------

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_first_step_only: bool = True  # train on step B only (ignores multi-step head beyond step-1)


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def prepare_text_toks(model: RewardIncreasePredictor, prompts: List[str], device: torch.device):
    toks = model.tokenize(prompts, device=device)
    return toks["input_ids"], toks["attention_mask"]


def train_blocks(
    model: RewardIncreasePredictor,
    rewards: torch.Tensor,
    prompts: List[str],
    B: int,
    K: int,
    rollout_len: Optional[int] = None,
    tcfg: Optional[TrainConfig] = None,
    verbose: bool = True,
):
    """
    rewards: (N, P) tensor of reward *increases* for P=number of prompts (e.g., 100)
    prompts: list of length P with prompt strings (aligned with rewards' columns)
    B: block size. Train uses history of length B-1 to predict the B-th step.
    K: number of gradient descent steps per call.
    rollout_len: progressive validation length L_val. If None, uses min(model.cfg.horizon, N - B - 1).

    Behavior:
      - Each step samples a time index t in [B-1, N-2].
      - History = rewards[t-(B-1):t, :]  -> shape (B-1, P)
      - Target  = rewards[t, :]          -> shape (P,)
      - Forward on all P prompts in a single batch; loss on step-1 of head.
      - After K updates, run closed-loop validation starting at t (same sampled t) for rollout_len steps,
        predicting B+1 from B, B+2 from (B+1 with predicted), etc.
    """
    assert B - 1 == model.cfg.history_len, (
        f"Model history_len={model.cfg.history_len} must equal B-1; got B={B}")
    tcfg = tcfg or TrainConfig()
    device = torch.device(tcfg.device)
    model.to(device)

    N, P = rewards.shape
    assert len(prompts) == P, "prompts length must match rewards columns"

    input_ids, attention_mask = prepare_text_toks(model, prompts, device)

    # Optimizer on trainable params only
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    # Helper closures
    def batch_forward(history_bt: torch.Tensor) -> torch.Tensor:
        # history_bt: (P, B-1)
        pred = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            reward_history=history_bt,
        )  # (P, H)
        return pred

    # def sample_t() -> int:
    #     # ensure we have room for at least one future point for evaluation
    #     return torch.randint(low=B-1, high=N-1, size=(1,)).item()

    # === Training ===
    val_logs = []
    train_losses = {}
    for t in range(B-1,N):
        model.train()
        step_losses = []
        for step in range(1, K + 1):
            last_t = t
            # Build history (B-1) for all prompts, shape -> (P, B-1)
            hist = rewards[t-(B-1):t, :].T.contiguous().to(device)
            target_next = rewards[t, :].to(device)  # (P,)

            pred_full = batch_forward(hist)  # (P, H)
            pred_next = pred_full[:, 0] if tcfg.use_first_step_only else pred_full.sum(dim=1) * 0.0  # enforce usage
            loss = loss_fn(pred_next, target_next)
            step_losses.append(loss.detach().cpu())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()

            if verbose and (step % max(1, K // 10) == 0 or step == 1 or step == K):
                print({"step": step, "t": int(t), "train_mse": float(loss.detach().cpu())})
        
        train_losses[t] = step_losses
        # === Progressive validation ===
        model.eval()
        with torch.no_grad():
            t0 = last_t 
            L_val_max = N - (t0 + 1)  # steps available beyond t0
            if rollout_len is None:
                rollout_len = int(min(model.cfg.horizon, max(1, L_val_max)))
            else:
                rollout_len = int(min(rollout_len, max(1, L_val_max)))

            # Seed history with the *true* last B-1 steps ending at t0
            hist = rewards[t0-(B-1):t0, :].T.contiguous().to(device)  # (P, B-1)
            mse_per_step: List[float] = []

            for k in range(rollout_len):
                # Predict next step for all prompts
                pred_k_full = batch_forward(hist)   # (P, H)
                pred_k = pred_k_full[:, 0]          # next single-step prediction

                # Compare to ground truth at t0 + k
                gt_k = rewards[t0 + k, :].to(device)
                mse_k = F.mse_loss(pred_k, gt_k).item()
                mse_per_step.append(mse_k)

                # Autoregressive update: drop oldest, append predicted
                hist = torch.cat([hist[:, 1:], pred_k.unsqueeze(1)], dim=1)

            val_report = {
                "t0": int(t0),
                "rollout_len": int(rollout_len),
                "mse@k": [float(x) for x in mse_per_step],
                "mse_mean": float(sum(mse_per_step) / len(mse_per_step)),
            }
            val_logs.append(val_report)
            if verbose:
                print(val_report)

    return val_logs, train_losses

def train_prefix_progressive(
    model: RewardIncreasePredictorSeq,
    rewards: torch.Tensor,
    prompts: List[str],
    B: int,
    K: int,
    rollout_len: Optional[int] = None,
    tcfg: Optional[TrainConfig] = None,
    verbose: bool = True,
):
    """
    Progressive-history trainer.

    At update step s (1-indexed), predict next delta at timestep t = B + s - 1
    using the *entire* prefix history [0..t-1] for all prompts. Optionally perform
    multiple optimizer updates at the same t (per_t_updates>1). After K steps,
    run progressive rollout eval starting from the last t.

    Args
      model: RewardIncreasePredictorSeq
      rewards: (N, P) float tensor of reward *increases*
      prompts: list[str] of length P
      B: initial history length (first prediction target index is t=B)
      K: number of gradient steps (advances t by 1 per step)
      rollout_len: evaluation rollout horizon; defaults to min(model.cfg.horizon, N - t0)
      per_t_updates: number of optimizer steps per t
      eval_every_step: if True, compute rollout MSE curves for each visited t
    Returns dict with training losses and eval metrics.
    """
    tcfg = tcfg or TrainConfig()
    device = torch.device(tcfg.device)
    model.to(device)

    N, P = rewards.shape
    assert 1 <= B <= N - 1, "B must be in [1, N-1]"
    assert len(prompts) == P

    # Tokenize once for all prompts
    toks = model.tokenize(prompts, device=device)
    input_ids, attention_mask = toks["input_ids"], toks["attention_mask"]

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=tcfg.lr, weight_decay=tcfg.weight_decay)


    def forward_for_t(t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # history length = t (indices 0..t-1); target at index t
        hist = rewards[:t, :].T.contiguous().to(device)  # (P, t)
        mask = torch.ones(P, t, dtype=torch.bool, device=device)
        pred_full = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            reward_history=hist,
            reward_mask=mask,
        )  # (P, H)
        target_next = rewards[t, :].to(device)  # (P,)
        return pred_full[:, 0], target_next

    def eval_rollout_from_t(t0: int) -> List[float]:
        # Closed-loop rollout starting at t0, comparing to GT at t0, t0+1, ...
        L_val_max = N - t0
        if rollout_len is None:
            L = int(min(model.cfg.horizon, max(1, L_val_max)))
        else:
            L = int(min(rollout_len, max(1, L_val_max)))
        # Seed history with true prefix 0..t0-1
        hist = rewards[:t0, :].T.contiguous().to(device)
        mask = torch.ones(hist.size(0), hist.size(1), dtype=torch.bool, device=device)
        mses = []
        with torch.no_grad():
            for k in range(L):
                pred_full = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    reward_history=hist,
                    reward_mask=mask,
                )
                pred = pred_full[:, 0]
                gt = rewards[t0 + k, :].to(device)
                mses.append(F.mse_loss(pred, gt).item())
                # append prediction to history (autoregressive)
                hist = torch.cat([hist, pred.unsqueeze(1)], dim=1)
                mask = torch.cat([mask, torch.ones(mask.size(0), 1, dtype=torch.bool, device=device)], dim=1)
        return mses

    # === Training ===
    # Logs keyed by timestep t
    losses_by_timestep: dict[int, list] = {}

    val_logs = []
    
    for t in range(B-1,N):
        model.train()
        step_losses = []
        for step in range(1, K + 1):
            opt.zero_grad(set_to_none=True)
            pred_next, target_next = forward_for_t(t)
            loss = loss_fn(pred_next, target_next)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            step_losses.append(float(loss.detach().cpu()))
            if verbose and (step % max(1, K // 10) == 0 or step == 1 or step == K):
                print({"step": step, "t": int(t), "train_mse": loss})
        
        losses_by_timestep[t] = step_losses
        model.eval()
        with torch.no_grad():
            mses = eval_rollout_from_t(t)
        val_report = {
            "t0": int(t),
            "rollout_len": len(mses),
            "mse@k": [float(x) for x in mses],
            "mse_mean": float(sum(mses) / len(mses)),
        }
        val_logs.append(val_report)

    return val_logs, losses_by_timestep

def train_text_only(
    model: RewardIncreasePredictorTextOnly,
    rewards: torch.Tensor,
    prompts: List[str],
    K: int,
    rollout_len: Optional[int] = None,
    tcfg: Optional[TrainConfig] = None,
    verbose: bool = True,
):
    """
    rewards: (N, P) tensor of reward *increases* for P=number of prompts (e.g., 100)
    prompts: list of length P with prompt strings (aligned with rewards' columns)
    B: block size. Train uses history of length B-1 to predict the B-th step.
    K: number of gradient descent steps per call.
    rollout_len: progressive validation length L_val. If None, uses min(model.cfg.horizon, N - B - 1).

    Behavior:
      - Each step samples a time index t in [B-1, N-2].
      - History = rewards[t-(B-1):t, :]  -> shape (B-1, P)
      - Target  = rewards[t, :]          -> shape (P,)
      - Forward on all P prompts in a single batch; loss on step-1 of head.
      - After K updates, run closed-loop validation starting at t (same sampled t) for rollout_len steps,
        predicting B+1 from B, B+2 from (B+1 with predicted), etc.
    """

    tcfg = tcfg or TrainConfig()
    device = torch.device(tcfg.device)
    model.to(device)

    N, P = rewards.shape
    assert len(prompts) == P, "prompts length must match rewards columns"

    input_ids, attention_mask = prepare_text_toks(model, prompts, device)

    # Optimizer on trainable params only
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    # === Training ===
    val_logs = []
    train_losses = {}
    for t in range(N):
        model.train()
        step_losses = []
        for step in range(1, K + 1):
            last_t = t

            target_next = rewards[t, :].to(device)  # (P,)

            pred_full = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_next = pred_full[:, 0]
            loss = loss_fn(pred_next, target_next)
            step_losses.append(loss.detach().cpu())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()

            if verbose and (step % max(1, K // 10) == 0 or step == 1 or step == K):
                print({"step": step, "t": int(t), "train_mse": float(loss.detach().cpu())})
        
        train_losses[t] = step_losses
        # === Progressive validation ===
        model.eval()
        with torch.no_grad():
            t0 = last_t 
            L_val_max = N - (t0 + 1)  # steps available beyond t0
            if rollout_len is None:
                rollout_len = int(min(model.cfg.horizon, max(1, L_val_max)))
            else:
                rollout_len = int(min(rollout_len, max(1, L_val_max)))

            # Seed history with the *true* last B-1 steps ending at t0

            mse_per_step: List[float] = []

            for k in range(rollout_len):
                # Predict next step for all prompts
                pred_k_full = model(input_ids=input_ids, attention_mask=attention_mask)
                pred_k = pred_k_full[:, 0]          # next single-step prediction

                # Compare to ground truth at t0 + k
                gt_k = rewards[t0 + k, :].to(device)
                mse_k = F.mse_loss(pred_k, gt_k).item()
                mse_per_step.append(mse_k)

            val_report = {
                "t0": int(t0),
                "rollout_len": int(rollout_len),
                "mse@k": [float(x) for x in mse_per_step],
                "mse_mean": float(sum(mse_per_step) / len(mse_per_step)),
            }
            val_logs.append(val_report)
            if verbose:
                print(val_report)

    return val_logs, train_losses


def train_text_only_loo_cv(
    model: RewardIncreasePredictorTextOnly,
    rewards: torch.Tensor,
    prompts: List[str],
    K_max: int,
    rollout_len: Optional[int] = None,
    tcfg: Optional[TrainConfig] = None,
    verbose: bool = True,
):
    """
    Leave-one-out CV to choose the number of GD steps per timestep.

    Args
      model: RewardIncreasePredictorTextOnly
      rewards: (N, P) tensor with reward increases (rows: timesteps, cols: prompts)
      prompts: list[str] of length P, aligned to rewards' columns
      K_max: maximum number of GD steps to consider per timestep (the grid is {1..K_max})
      rollout_len: eval rollout horizon used after final per-t training
      tcfg: TrainConfig
      verbose: print per-step progress

    Returns:
      val_logs: list of dicts with rollout evaluation per timestep
      train_losses: dict[timestep -> list[float]] with training losses during the final
                    'best_step' updates on all prompts
      choice_log: dict with:
        - "best_step_by_t": dict[timestep -> int]
        - "mean_val_mse_by_t": dict[timestep -> List[float]]: mean LOO val MSE after step k (1..K_max)
        - "val_mse_by_t_and_fold": dict[timestep -> List[List[float]]], where outer index is fold j,
                                   inner list is val MSE after step k (1..K_max)
    """
    tcfg = tcfg or TrainConfig()
    device = torch.device(tcfg.device)
    model.to(device)

    N, P = rewards.shape
    assert len(prompts) == P, "prompts length must match rewards columns"

    # Tokenize once
    input_ids, attention_mask = prepare_text_toks(model, prompts, device)

    # Helper to build a fresh optimizer
    def make_opt(m: nn.Module):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()),
                                 lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    # Convenience forward producing per-prompt next-step prediction (P,)
    def forward_next() -> torch.Tensor:
        pred_full = model(input_ids=input_ids, attention_mask=attention_mask)  # (P, H)
        return pred_full[:, 0]  # (P,)

    # Logs
    val_logs = []
    train_losses: dict[int, List[float]] = {}
    best_step_by_t: dict[int, int] = {}
    mean_val_mse_by_t: dict[int, List[float]] = {}
    val_mse_by_t_and_fold: dict[int, List[List[float]]] = {}
    global_opt = make_opt(model)
    for t in range(N):
        # Targets for this timestep
        target_next = rewards[t, :].to(device)  # (P,)
        base_state = copy.deepcopy(model.state_dict())
        base_opt = copy.deepcopy(global_opt.state_dict())
        # === LOO: for each held-out prompt j, track val MSE after each step 1..K_max ===
        fold_val_curves: List[List[float]] = []

        for j in range(P):
            # Training indices = all except j
            train_idx = torch.tensor([i for i in range(P) if i != j], device=device, dtype=torch.long)

            # Reset model to base state for this fold
            model.load_state_dict(base_state)
            model.train()
            opt = make_opt(model)

            val_curve_j: List[float] = []

            for step in range(1, K_max + 1):
                # --- One GD step on the training subset (P-1 prompts) ---
                opt.zero_grad(set_to_none=True)
                pred_next_all = forward_next()             # (P,)
                loss = loss_fn(pred_next_all.index_select(0, train_idx),
                               target_next.index_select(0, train_idx))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()

                # --- Validation on held-out prompt j ---
                model.eval()
                with torch.no_grad():
                    pred_val_all = forward_next()
                    val_mse = F.mse_loss(pred_val_all[j], target_next[j])
                    val_curve_j.append(float(val_mse.detach().cpu()))
                model.train()

                if verbose and (step == 1 or step == K_max or step % max(1, K_max // 5) == 0):
                    print({"t": int(t), "fold_j": int(j), "step": int(step),
                           "train_mse_on_P-1": float(loss.detach().cpu()),
                           "val_mse_on_j": float(val_curve_j[-1])})

            fold_val_curves.append(val_curve_j)

        # Aggregate across folds: mean val MSE per step
        # shape: P x K_max -> K_max
        mean_by_step = [float(sum(fold_val_curves[j][k] for j in range(P)) / P) for k in range(K_max)]
        best_step = int(1 + min(range(K_max), key=lambda k: mean_by_step[k]))

        best_step_by_t[t] = best_step
        mean_val_mse_by_t[t] = mean_by_step
        val_mse_by_t_and_fold[t] = fold_val_curves

        if verbose:
            print({"t": int(t), "K_max": int(K_max), "mean_val_mse_by_step": mean_by_step, "best_step": int(best_step)})

        # === Final per-t training: restore base, train for 'best_step' using ALL prompts ===
        model.load_state_dict(base_state)
        global_opt.load_state_dict(base_opt)
        model.train()
        step_losses: List[float] = []
        for step in range(1, best_step + 1):
            global_opt.zero_grad(set_to_none=True)
            pred_next_all = forward_next()
            loss = loss_fn(pred_next_all, target_next)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            global_opt.step()
            step_losses.append(float(loss.detach().cpu()))
            if verbose and (step == 1 or step == best_step or step % max(1, best_step // 5) == 0):
                print({"t": int(t), "final_train_step": int(step), "train_mse_all_P": float(step_losses[-1])})
        train_losses[t] = step_losses

        # === Rollout evaluation from t (same as original, but without mutating rollout_len) ===
        model.eval()
        with torch.no_grad():
            t0 = t
            L_val_max = N - (t0 + 1)
            L = int(min(model.cfg.horizon, max(1, L_val_max))) if rollout_len is None \
                else int(min(rollout_len, max(1, L_val_max)))

            mse_per_step: List[float] = []
            for k in range(L):
                pred_k = forward_next()       # (P,)
                gt_k = rewards[t0 + k, :].to(device)
                mse_k = F.mse_loss(pred_k, gt_k).item()
                mse_per_step.append(mse_k)

            val_report = {
                "t0": int(t0),
                "rollout_len": int(L),
                "mse@k": [float(x) for x in mse_per_step],
                "mse_mean": float(sum(mse_per_step) / len(mse_per_step)),
                "best_step": int(best_step),
            }
            val_logs.append(val_report)
            if verbose:
                print(val_report)

    choice_log = {
        "best_step_by_t": best_step_by_t,
        "mean_val_mse_by_t": mean_val_mse_by_t,
        "val_mse_by_t_and_fold": val_mse_by_t_and_fold,
    }
    return val_logs, train_losses, choice_log
