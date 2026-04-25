"""MAGCAF: Modality-Adaptive Gated Cross-Attention Fusion.

Task-conditioned multi-source heterogeneous fusion of four frozen pretrained
sources, each producing complementary representations of the same DAiSEE
face-cropped clip:

    spatial      (B, T,   512)         InceptionResnetV1 / VGGFace2 face-identity
    videomae     (B,      768)         VideoMAE   K400 self-supervised
    timesformer  (B,      768)         TimeSformer K400 supervised
    landmark_seq (B, T, 478, 3)        MediaPipe FaceMesh 3D landmarks

For each affective task k in {Boredom, Engagement, Confusion, Frustration},
MAGCAF forms a task-conditioned query and gate from the joint context of the
four sources, computes softmax cross-attention over the four sources, and
emits a task-specific feature vector. Independent two-layer MLP heads map
each task feature to per-task logits, which are then coupled by a learnable
task-correlation matrix Omega regularised toward the empirical Pearson
correlation prior on the training labels. The four per-task cross-entropy
losses are aggregated with Kendall-Gal homoscedastic uncertainty weighting.

Forward signature:
    forward(spatial_seq, videomae_feat, timesformer_feat, landmark_seq)
        -> (logits (B, K=4, C=4), aux dict)

Output aux dict contents:
    attn        (B, K, n_src=4)  per-task source-attention probabilities
    gate        (B, K, 1)        per-task scalar gate
    omega       (K, K)           current Omega matrix (when not ablated)
    log_sigma2  (K,)             current uncertainty-weighting log-variances

Ablation flags via `ablate=...`:
    "no_magcaf" : replace the gated cross-attention by plain concat fusion.
    "no_omega"  : drop the task-correlation head, output uncoupled logits.
    "no_uw"     : drop the uncertainty weighting, use uniform 1/K loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import OmegaHead, UncertaintyWeighter


@dataclass
class MAGCAFv2Config:
    d_spatial: int = 512
    d_videomae: int = 768
    d_timesformer: int = 768
    d_landmark: int = 256
    d_model: int = 256
    dropout: float = 0.3
    num_tasks: int = 4
    num_classes: int = 4
    spatial_temporal: str = "mean"   # "mean" | "bilstm" | "attn"
    lstm_hidden: int = 256
    lstm_layers: int = 1
    use_landmark: bool = True
    num_landmarks: int = 478


class _MultiSourceFusion(nn.Module):
    """Per-task gated cross-attention over 3 or 4 projected source vectors."""
    def __init__(self, cfg: MAGCAFv2Config):
        super().__init__()
        D = cfg.d_model
        self.D = D
        self.K = cfg.num_tasks
        self.use_landmark = cfg.use_landmark
        self.n_src = 4 if cfg.use_landmark else 3

        # Per-source LayerNorm + projection to a common dim D.
        self.src_norm_s = nn.LayerNorm(cfg.d_spatial)
        self.src_norm_v = nn.LayerNorm(cfg.d_videomae)
        self.src_norm_t = nn.LayerNorm(cfg.d_timesformer)
        self.proj_s = nn.Linear(cfg.d_spatial,     D)
        self.proj_v = nn.Linear(cfg.d_videomae,    D)
        self.proj_t = nn.Linear(cfg.d_timesformer, D)
        if cfg.use_landmark:
            self.src_norm_l = nn.LayerNorm(cfg.d_landmark)
            self.proj_l = nn.Linear(cfg.d_landmark, D)

        concat_dim = self.n_src * D
        self.W_q = nn.Parameter(torch.empty(cfg.num_tasks, concat_dim, D))
        self.W_g = nn.Parameter(torch.empty(cfg.num_tasks, concat_dim, 1))
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_g)

        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(D)

    def forward(self, h_s: torch.Tensor, h_v: torch.Tensor, h_t: torch.Tensor,
                h_l: torch.Tensor | None = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hs = self.proj_s(self.src_norm_s(h_s))
        hv = self.proj_v(self.src_norm_v(h_v))
        ht = self.proj_t(self.src_norm_t(h_t))
        if self.use_landmark:
            assert h_l is not None, "use_landmark=True but landmark feature missing"
            hl = self.proj_l(self.src_norm_l(h_l))
            sources = torch.stack([hs, hv, ht, hl], dim=1)
            joint = torch.cat([hs, hv, ht, hl], dim=-1)
        else:
            sources = torch.stack([hs, hv, ht], dim=1)
            joint = torch.cat([hs, hv, ht], dim=-1)

        q = torch.einsum("bn,knd->bkd", joint, self.W_q)
        g = torch.einsum("bn,kne->bke", joint, self.W_g)
        g = torch.sigmoid(g)

        k = self.W_k(sources)
        v = self.W_v(sources)
        scale = 1.0 / math.sqrt(self.D)
        scores = torch.einsum("bkd,bsd->bks", q, k) * scale
        attn = F.softmax(scores, dim=-1)

        fused = torch.einsum("bks,bsd->bkd", attn, v)
        fused = g * fused
        fused = self.drop(self.norm(fused))
        return fused, {"attn": attn.detach(), "gate": g.detach()}


class _ConcatFusion(nn.Module):
    """Plain concat replacement for the ablation `ablate=no_magcaf`."""
    def __init__(self, cfg: MAGCAFv2Config):
        super().__init__()
        in_dim = cfg.d_spatial + cfg.d_videomae + cfg.d_timesformer
        if cfg.use_landmark:
            in_dim += cfg.d_landmark
        self.use_landmark = cfg.use_landmark
        self.K = cfg.num_tasks
        self.proj = nn.Linear(in_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h_s: torch.Tensor, h_v: torch.Tensor, h_t: torch.Tensor,
                h_l: torch.Tensor | None = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.use_landmark:
            assert h_l is not None, "landmark feature missing"
            x = torch.cat([h_s, h_v, h_t, h_l], dim=-1)
        else:
            x = torch.cat([h_s, h_v, h_t], dim=-1)
        x = self.drop(self.norm(self.proj(x)))
        fused = x.unsqueeze(1).expand(-1, self.K, -1)
        return fused, {"attn": None, "gate": None}


class _LandmarkEncoder(nn.Module):
    """MediaPipe FaceMesh per-frame 3D landmarks -> per-clip vector.

    Architecture: per-frame flatten + LayerNorm + Linear + GELU + Dropout
    -> single-layer BiLSTM -> additive attention pool over time -> Linear
    projection back to d_out. The landmark trajectory carries head-pose
    motion and expression-deformation cues that are orthogonal to the
    appearance signal of the three RGB-derived sources.
    """
    def __init__(self, num_landmarks: int = 478, hidden: int = 128,
                 d_out: int = 256, dropout: float = 0.3):
        super().__init__()
        in_dim = num_landmarks * 3
        self.norm_in = nn.LayerNorm(in_dim)
        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, d_out), nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(d_out, hidden, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn_w = nn.Linear(hidden * 2, hidden)
        self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.out_proj = nn.Linear(hidden * 2, d_out)
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, num_landmarks, 3)
        B, T, N, D = x.shape
        x = x.reshape(B, T, N * D)
        x = self.frame_proj(self.norm_in(x))
        h, _ = self.lstm(x)
        e = self.attn_v(torch.tanh(self.attn_w(h))).squeeze(-1)
        alpha = F.softmax(e, dim=-1).unsqueeze(-1)
        pooled = (alpha * h).sum(1)
        return self.out_proj(pooled)


class MAGCAFv2Net(nn.Module):
    VALID_ABLATES = (None, "no_magcaf", "no_omega", "no_uw")

    def __init__(self, cfg: MAGCAFv2Config | None = None,
                 omega_prior: torch.Tensor | None = None,
                 class_counts_per_task: torch.Tensor | None = None,
                 ablate: str | None = None, **_):
        super().__init__()
        assert ablate in self.VALID_ABLATES, f"unknown ablate={ablate}"
        self.cfg = cfg or MAGCAFv2Config()
        cfg = self.cfg
        self.ablate = ablate

        # Spatial temporal encoder over the per-frame IRv1 sequence.
        self.spatial_temporal = cfg.spatial_temporal
        if cfg.spatial_temporal == "bilstm":
            self.spatial_lstm = nn.LSTM(
                cfg.d_spatial, cfg.lstm_hidden, num_layers=cfg.lstm_layers,
                bidirectional=True, batch_first=True, dropout=0.0,
            )
            lstm_out = 2 * cfg.lstm_hidden
            self.spatial_attn_w = nn.Linear(lstm_out, cfg.d_model)
            self.spatial_attn_v = nn.Linear(cfg.d_model, 1, bias=False)
            self.spatial_out_dim = lstm_out
        elif cfg.spatial_temporal == "attn":
            self.spatial_attn_w = nn.Linear(cfg.d_spatial, cfg.d_model)
            self.spatial_attn_v = nn.Linear(cfg.d_model, 1, bias=False)
            self.spatial_out_dim = cfg.d_spatial
        else:     # "mean"
            self.spatial_out_dim = cfg.d_spatial

        # 4th-source landmark encoder.
        self.landmark_encoder = (
            _LandmarkEncoder(num_landmarks=cfg.num_landmarks,
                             d_out=cfg.d_landmark, dropout=cfg.dropout)
            if cfg.use_landmark else None
        )

        # Effective config that the fusion layer sees (spatial dim adjusted
        # to the temporal encoder's output dim).
        cfg_eff = MAGCAFv2Config(
            d_spatial=self.spatial_out_dim, d_videomae=cfg.d_videomae,
            d_timesformer=cfg.d_timesformer, d_landmark=cfg.d_landmark,
            d_model=cfg.d_model, dropout=cfg.dropout,
            num_tasks=cfg.num_tasks, num_classes=cfg.num_classes,
            use_landmark=cfg.use_landmark, num_landmarks=cfg.num_landmarks,
        )
        self.fusion = (_ConcatFusion(cfg_eff)
                       if (ablate and "no_magcaf" in ablate)
                       else _MultiSourceFusion(cfg_eff))

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.num_classes),
            ) for _ in range(cfg.num_tasks)
        ])

        # Single-source residual head on VideoMAE: a soft fallback that lets
        # the model recover the strongest single-source baseline if MAGCAF's
        # attention or Omega head collapses.
        self.residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(cfg.d_videomae),
                nn.Linear(cfg.d_videomae, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.num_classes),
            ) for _ in range(cfg.num_tasks)
        ])
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

        self.omega_head = (None if (ablate and "no_omega" in ablate)
                           else OmegaHead(cfg.num_tasks, cfg.num_classes,
                                          omega_prior=omega_prior))
        self.uw = (None if (ablate and "no_uw" in ablate)
                   else UncertaintyWeighter(cfg.num_tasks))

    def _encode_spatial(self, spatial_seq: torch.Tensor) -> torch.Tensor:
        """(B, T, d_s) -> (B, spatial_out_dim) via the chosen temporal encoder."""
        if self.spatial_temporal == "bilstm":
            h, _ = self.spatial_lstm(spatial_seq)
            e = self.spatial_attn_v(torch.tanh(self.spatial_attn_w(h))).squeeze(-1)
            alpha = F.softmax(e, dim=-1).unsqueeze(-1)
            return (alpha * h).sum(1)
        if self.spatial_temporal == "attn":
            e = self.spatial_attn_v(torch.tanh(self.spatial_attn_w(spatial_seq))).squeeze(-1)
            alpha = F.softmax(e, dim=-1).unsqueeze(-1)
            return (alpha * spatial_seq).sum(1)
        return spatial_seq.mean(dim=1)

    def forward(self, spatial_seq: torch.Tensor,
                videomae_feat: torch.Tensor,
                timesformer_feat: torch.Tensor,
                landmark_seq: torch.Tensor | None = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_s = self._encode_spatial(spatial_seq)
        h_l = (self.landmark_encoder(landmark_seq)
               if (self.cfg.use_landmark and landmark_seq is not None) else None)
        if self.cfg.use_landmark:
            assert h_l is not None, "use_landmark=True but landmark_seq missing"
        fused, aux = self.fusion(h_s, videomae_feat, timesformer_feat, h_l)

        magcaf_logits = torch.stack(
            [self.heads[k](fused[:, k]) for k in range(self.cfg.num_tasks)],
            dim=1,
        )
        residual_logits = torch.stack(
            [self.residual_heads[k](videomae_feat)
             for k in range(self.cfg.num_tasks)],
            dim=1,
        )
        task_logits = magcaf_logits + self.residual_scale * residual_logits
        aux["residual_scale"] = self.residual_scale.detach()

        if self.omega_head is not None:
            out = self.omega_head(task_logits)
            aux["omega"] = self.omega_head.omega.detach()
        else:
            out = task_logits
        if self.uw is not None:
            aux["log_sigma2"] = self.uw.log_sigma2.detach()
        aux["task_logits_uncoupled"] = task_logits.detach()
        return out, aux

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     per_task_criteria) -> Tuple[torch.Tensor, torch.Tensor]:
        K = logits.size(1)
        task_losses = torch.stack([
            per_task_criteria[k](logits[:, k], targets[:, k]) for k in range(K)
        ])
        total = (self.uw(task_losses) if self.uw is not None
                 else task_losses.mean())
        if self.omega_head is not None:
            total = total + self.omega_head.regularizer_loss()
        return total, task_losses.detach()
