# MAGCAF: Modality-Adaptive Gated Cross-Attention Fusion

Reference implementation of the MAGCAF model for fine-grained student-engagement
recognition on the [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)
benchmark.

MAGCAF integrates four **frozen** pretrained sources of complementary
representations of the same RGB clip and learns a **task-conditioned gated
cross-attention** that mixes these sources differently for each affective
dimension. A learnable task-correlation matrix and a Kendall–Gal
uncertainty-weighted multi-task loss complete the framework:

| Source | Encoder | Pretraining | Output dim |
|---|---|---|---|
| Face-domain identity | InceptionResnetV1 | VGGFace2 | 512 |
| Self-supervised video | VideoMAE-base | Kinetics-400 SSL | 768 |
| Supervised spatiotemporal | TimeSformer-base | Kinetics-400 supervised | 768 |
| 3D facial geometry | MediaPipe FaceMesh | — (geometric, structured) | 256 |

## Repository layout

```
source_codes/
├── common/
│   ├── protocol.py          # CFG: split / seed / optim / batch / epochs
│   ├── metrics.py           # per-task F1/AUC/Kappa + McNemar paired test
│   └── logging_utils.py
├── data/
│   ├── preprocess_clip.py            # MTCNN face detect + crop, runs once
│   ├── face_pipeline.py              # facenet-pytorch MTCNN + 5-point align
│   ├── extract_features.py           # cache InceptionResnetV1 face features
│   ├── extract_transformer_features.py # cache VideoMAE + TimeSformer
│   ├── extract_landmark_features.py  # cache MediaPipe FaceMesh landmarks
│   └── daisee_dataset.py             # torch.Dataset with feature dispatch
├── models/
│   ├── _common.py            # MultiTaskHead, OmegaHead, UncertaintyWeighter
│   ├── magcaf_v2.py          # MAGCAFv2Net: the model of record
│   ├── baseline_engagement.py # ResNet-TCN (Abedi 2021), LRCN (Donahue 2015)
│   ├── baseline_transformer.py# TimeSformer, VideoMAE (frozen + adapter)
│   ├── baseline_vibednet.py  # ViBED-Net (Gothwal 2025), dual-stream EffNetV2-S + LSTM
│   └── build.py              # name -> nn.Module factory
├── losses/                  # ce | weighted_ce | focal | cb_focal | ldam
├── train/train_single.py    # unified trainer, McNemar-ready predictions
├── experiments/
│   ├── run_all.sh           # phase orchestration
│   ├── run_vibednet.sh      # ViBED-Net 3-seed under unified protocol
│   ├── aggregate_tables.py  # produces 3-seed mean±std LaTeX tables
│   └── ensemble.py          # 3-seed probability ensemble + per-task breakdown
└── environment.yml          # conda env "magcaf"
```

## Quick start

```bash
# 1. environment
conda env create -f environment.yml
conda activate magcaf

# 2. download DAiSEE (videos + labels) into $DAISEE_ROOT
#    https://people.iith.ac.in/vineethnb/resources/daisee/index.html
export DAISEE_ROOT=/path/to/DAiSEE

# 3. preprocess (once, ~3-4h on a single 4090)
python -m data.preprocess_clip      --face-cache $DAISEE_ROOT/face_cache
python -m data.extract_features              --face-cache $DAISEE_ROOT/face_cache
python -m data.extract_transformer_features  --face-cache $DAISEE_ROOT/face_cache
python -m data.extract_landmark_features     --face-cache $DAISEE_ROOT/face_cache
# (optional) ViBED-Net dual-stream EfficientNetV2-S features
python -m data.extract_vibednet_features \
    --face-cache $DAISEE_ROOT/face_cache \
    --avi-root   $DAISEE_ROOT/DAiSEE/DataSet \
    --weights    /path/to/efficientnet_v2_s-dd5fe13b.pth

# 4. train MAGCAF (4-source) and the five baselines, 3 seeds each
bash experiments/run_all.sh
bash experiments/run_vibednet.sh   # adds the ViBED-Net 2025 baseline

# 5. aggregate LaTeX-ready tables
python experiments/aggregate_tables.py --runs runs/pilot --out tables/
```

## Reproducing the numbers

The reference 3-seed-mean numbers reported in the paper:

| Model | Avg Acc | Macro-F1 | macro-AUC |
|---|---|---|---|
| LRCN (Gupta 2016 / Donahue 2015) | 58.20 ± 0.82 | 0.258 | 0.556 |
| ResNet-TCN (Abedi & Khan 2021) | 58.14 ± 2.10 | 0.261 | 0.564 |
| TimeSformer (Bertasius 2021) | 60.85 ± 0.38 | 0.273 | 0.642 |
| VideoMAE (Tong 2022) | 60.84 ± 0.57 | 0.269 | 0.633 |
| ViBED-Net (Gothwal 2025) | 57.20 ± 1.56 | 0.272 ± 0.009 | 0.545 ± 0.012 |
| **MAGCAF (ours)** | **64.09 ± 0.39** | **0.286 ± 0.006** | **0.658 ± 0.007** |

All numbers under the unified protocol: T=16 frames, 20 epochs, AdamW lr=1e-4
weight-decay=1e-4 + cosine schedule, batch=16, AMP fp16, multi-task CE, early
stop on val Macro-F1, seeds {42, 123, 2024}.

**Note on ViBED-Net.** The original ViBED-Net paper reports 73.43% on a
single-task Engagement-only protocol with T=60 frames, 40 epochs, aggressive
minority-class augmentation (salt-and-pepper, elastic), and single-run
reporting. The 57.20% figure above is what the same dual-stream
EfficientNetV2-S + LSTM architecture produces under our unified four-task
multi-task protocol with T=16, 20 epochs, no aggressive augmentation, and
3-seed mean ± std reporting. The drop is the expected cost of protocol
control; under the unified protocol ViBED-Net (2025) sits below the
2021/2022 video Transformers, suggesting that on this benchmark aggregate
performance depends jointly on the choice of backbone and on training-protocol
details rather than on backbone recency alone.

Component ablation (within MAGCAF, 3-seed mean):

| Variant | Avg Acc | Macro-F1 | macro-AUC |
|---|---|---|---|
| MAGCAF (full) | **64.09 ± 0.39** | **0.286 ± 0.006** | **0.658 ± 0.007** |
| − M1 (gated → concat fusion) | 60.44 ± 0.64 | 0.277 ± 0.006 | 0.649 ± 0.003 |
| − M2 (Ω task-correlation head) | 58.87 ± 0.38 | 0.285 ± 0.011 | 0.620 ± 0.012 |
| − M3 (Kendall–Gal uncertainty weighting) | 59.66 ± 0.80 | 0.281 ± 0.005 | 0.631 ± 0.016 |

## Citation

If you use this code, please cite our paper.

## Licence

MIT.
