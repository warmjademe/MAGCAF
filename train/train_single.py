"""Unified trainer for every (model, loss, seed[, ablate]) combination.

Model dispatch:
    feature_mode models  -> MAGCAFNet, FusionAttnLSTM
         forward(spatial_seq, flow_stacks) -> (B, K, C) logits
    raw_mode models      -> slowfast, timesformer, videomae, resnet_tcn
         forward(frames) -> (B, K, C) logits

Phase A one-liner:
    python -m train.train_single \
        --model magcaf --loss cb_focal --seed 42 \
        --run-dir runs/phaseA/magcaf__cb_focal__s42 --epochs 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.logging_utils import dump_json, setup_logger          # noqa: E402
from common.metrics import compute_all_metrics                    # noqa: E402
from common.protocol import (                                     # noqa: E402
    AFFECTIVE_TASKS, CFG, NUM_TASKS, set_seed, worker_init_fn,
)
from data.daisee_dataset import (                                 # noqa: E402
    DAiSEEDataset, class_counts_per_task, label_correlation, load_labels_df,
)
from losses.build import build_loss                               # noqa: E402
from models.build import build_model                              # noqa: E402


# All models train on cached frozen features (linear-probe-style protocol).
FEATURE_MODELS = {"magcaf", "ours", "magcaf_v2",
                  "timesformer", "videomae", "resnet_tcn", "lrcn"}
TRANSFORMER_CACHED = {"timesformer", "videomae"}

# Minimal dataset feature set per model (saves I/O)
FEATURES_WANTED = {
    "magcaf":      ("spatial", "videomae", "timesformer", "landmark_seq"),
    "ours":        ("spatial", "videomae", "timesformer", "landmark_seq"),
    "magcaf_v2":   ("spatial", "videomae", "timesformer", "landmark_seq"),
    "timesformer": ("timesformer",),
    "videomae":    ("videomae",),
    "resnet_tcn":  ("spatial",),
    "lrcn":        ("spatial",),
}


def build_dataloaders(mode: str, batch_size: int,
                      features_wanted: tuple[str, ...]):
    train_ds = DAiSEEDataset("Train", mode=mode, features_wanted=features_wanted,
                             augment=True)
    val_ds = DAiSEEDataset("Validation", mode=mode, features_wanted=features_wanted,
                           augment=False)
    test_ds = DAiSEEDataset("Test", mode=mode, features_wanted=features_wanted,
                            augment=False)
    loaders = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, num_workers=4, pin_memory=True,
            shuffle=(name == "train"), drop_last=(name == "train"),
            worker_init_fn=worker_init_fn,
        )
    return loaders, train_ds.df


# ---------------------------------------------------------------------------
def forward_model(model, batch, mode, device, model_name: str) -> torch.Tensor:
    """Dispatch on model identity so signatures stay tight."""
    if model_name in TRANSFORMER_CACHED:
        feat = batch[model_name].to(device, non_blocking=True)
        return model(feat)

    if model_name in ("resnet_tcn", "lrcn"):
        spatial = batch["spatial"].to(device, non_blocking=True)
        return model(spatial)

    # MAGCAF: 4-source heterogeneous fusion
    spatial = batch["spatial"].to(device, non_blocking=True)
    vmae = batch["videomae"].to(device, non_blocking=True)
    tsf = batch["timesformer"].to(device, non_blocking=True)
    landmark_seq = (batch["landmark_seq"].to(device, non_blocking=True)
                    if "landmark_seq" in batch else None)
    out = model(spatial, vmae, tsf, None, landmark_seq)
    if isinstance(out, tuple):
        return out[0]
    return out


def run_epoch(model, loader, per_task_criteria, optimizer, scaler, device,
              mode: str, _model_name: str, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    all_probs = {t: [] for t in AFFECTIVE_TASKS}
    all_trues = {t: [] for t in AFFECTIVE_TASKS}

    for batch in loader:
        n_batches += 1
        labels = batch["labels"].to(device, non_blocking=True)       # (B, 4)

        with autocast(dtype=torch.float16):
            logits = forward_model(model, batch, mode, device, _model_name)  # (B, K, C)

            if hasattr(model, "compute_loss"):
                total, _per_task = model.compute_loss(logits, labels, per_task_criteria)
            else:
                per_task = torch.stack([
                    per_task_criteria[k](logits[:, k], labels[:, k])
                    for k in range(NUM_TASKS)
                ])
                total = per_task.mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CFG.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(total.item())
        probs = torch.softmax(logits.float(), dim=-1)
        for k, t in enumerate(AFFECTIVE_TASKS):
            all_probs[t].append(probs[:, k].detach().cpu().numpy())
            all_trues[t].append(labels[:, k].cpu().numpy())

    probs_cat = {t: np.concatenate(all_probs[t], 0) for t in AFFECTIVE_TASKS}
    trues_cat = {t: np.concatenate(all_trues[t], 0) for t in AFFECTIVE_TASKS}
    metrics = compute_all_metrics(trues_cat, probs_cat)
    return total_loss / max(1, n_batches), metrics, probs_cat, trues_cat


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loss", default="cb_focal")
    ap.add_argument("--ablate", default=None,
                    help="MAGCAF ablate flag: no_magcaf | no_omega | no_uw | no_magcaf_no_omega")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-cfg", default="",
                    help="Comma-sep key=value overrides, e.g. 'd_model=512,spatial_temporal=bilstm'")
    ap.add_argument("--save-gates", action="store_true",
                    help="For magcaf_v2: dump per-sample per-task source-attention weights on the test set.")
    args = ap.parse_args()

    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    log = setup_logger(args.run_dir)
    log.info("Args: %s", vars(args))

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    model_name = args.model.lower()
    mode = "features" if model_name in FEATURE_MODELS else "raw"
    feats = FEATURES_WANTED.get(model_name, ("spatial", "flow"))
    log.info("Mode=%s  features_wanted=%s", mode, feats)

    loaders, train_df = build_dataloaders(mode, args.batch, feats)
    cc = class_counts_per_task(train_df)                              # (K, C)
    log.info("Class counts per task:\n%s", cc.numpy())

    per_task_criteria = nn.ModuleList([
        build_loss(args.loss, cc[k].tolist()) for k in range(NUM_TASKS)
    ]).to(device)

    build_kwargs = {}
    if args.model.lower() in ("magcaf", "ours", "magcaf_v2"):
        build_kwargs["omega_prior"] = label_correlation(train_df)
        build_kwargs["class_counts_per_task"] = cc
        build_kwargs["ablate"] = args.ablate

    model_cfg = {}
    if args.model_cfg:
        for kv in args.model_cfg.split(","):
            k, v = kv.split("=")
            v = v.strip()
            if v.lower() in ("true", "false"):
                v = (v.lower() == "true")
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            model_cfg[k.strip()] = v
    build_kwargs["model_cfg"] = model_cfg
    model = build_model(args.model, **build_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s  trainable_params=%d", type(model).__name__, n_params)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=CFG.min_lr,
    )
    scaler = GradScaler()

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_m, _, _ = run_epoch(
            model, loaders["train"], per_task_criteria,
            optimizer, scaler, device, mode, args.model.lower(), train=True,
        )
        scheduler.step()

        do_eval = epoch % CFG.eval_every_epochs == 0 or epoch == args.epochs
        if do_eval:
            with torch.no_grad():
                val_loss, val_m, _, _ = run_epoch(
                    model, loaders["val"], per_task_criteria,
                    optimizer, scaler, device, mode, args.model.lower(), train=False,
                )
            log.info(
                "ep=%d  lr=%.2e  tl=%.4f  vl=%.4f  va=%.4f  vf1=%.4f  vauc=%.4f  (%.1fs)",
                epoch, optimizer.param_groups[0]["lr"], tr_loss, val_loss,
                val_m.mean_accuracy, val_m.mean_macro_f1, val_m.mean_auc,
                time.time() - t0,
            )
            if val_m.mean_macro_f1 > best_val_f1:
                best_val_f1 = val_m.mean_macro_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= CFG.early_stop_patience:
                    log.info("Early-stop at epoch %d", epoch)
                    break
        else:
            log.info("ep=%d  lr=%.2e  tl=%.4f  (%.1fs)",
                     epoch, optimizer.param_groups[0]["lr"], tr_loss,
                     time.time() - t0)

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        test_loss, test_m, test_probs, test_trues = run_epoch(
            model, loaders["test"], per_task_criteria,
            optimizer, scaler, device, mode, args.model.lower(), train=False,
        )
    log.info("TEST: acc=%.4f  f1=%.4f  auc=%.4f",
             test_m.mean_accuracy, test_m.mean_macro_f1, test_m.mean_auc)

    dump_json(os.path.join(args.run_dir, "metrics.json"), {
        "args": vars(args),
        "best_val_macro_f1": best_val_f1,
        "test": test_m.to_dict(),
    })
    np.savez_compressed(
        os.path.join(args.run_dir, "predictions.npz"),
        **{f"prob_{t}": test_probs[t] for t in AFFECTIVE_TASKS},
        **{f"true_{t}": test_trues[t] for t in AFFECTIVE_TASKS},
    )
    log.info("Dumped metrics + predictions to %s", args.run_dir)

    # Optional: dump per-task source-attention gates on the test set for RQ3.
    if args.save_gates and args.model.lower() in ("magcaf", "magcaf_v2", "ours"):
        log.info("Extracting per-task source attention gates on test set...")
        model.eval()
        all_attn = []           # list of (B, K, n_sources)
        all_gate = []           # list of (B, K, 1)
        with torch.no_grad():
            for batch in loaders["test"]:
                spatial = batch["spatial"].to(device, non_blocking=True)
                vmae = batch["videomae"].to(device, non_blocking=True)
                tsf = batch["timesformer"].to(device, non_blocking=True)
                landmark_seq = (batch["landmark_seq"].to(device, non_blocking=True)
                                if "landmark_seq" in batch else None)
                _, aux = model(spatial, vmae, tsf, None, landmark_seq)
                if aux.get("attn") is not None:
                    all_attn.append(aux["attn"].cpu().numpy())
                if aux.get("gate") is not None:
                    all_gate.append(aux["gate"].cpu().numpy())
        if all_attn:
            attn = np.concatenate(all_attn, 0)      # (N, K, n_src)
            np.save(os.path.join(args.run_dir, "gate_attn.npy"), attn)
            log.info("Gate attention: %s  mean per-task source weights:\n%s",
                     attn.shape, attn.mean(0))
        if all_gate:
            gate = np.concatenate(all_gate, 0)      # (N, K, 1)
            np.save(os.path.join(args.run_dir, "gate_scalar.npy"), gate)


if __name__ == "__main__":
    main()
