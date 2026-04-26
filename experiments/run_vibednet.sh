#!/usr/bin/env bash
# Train ViBED-Net under our unified protocol: T=16, 20 epoch, plain CE, 3 seeds.
set -e
cd /home/qyb/TongBu/discover_artifical_intelligence
source /home/qyb/miniconda3/etc/profile.d/conda.sh
conda activate discover_artifical_intelligence

LOSS=ce
SEEDS="42 123 2024"
ROOT=runs/pilot

for s in $SEEDS; do
    OUT="$ROOT/vibednet__${LOSS}__s${s}"
    mkdir -p "$OUT"
    echo "=== ViBED-Net seed=$s -> $OUT ==="
    python -m train.train_single \
        --model vibednet --loss "$LOSS" --seed "$s" \
        --run-dir "$OUT" --epochs 20 \
        2>&1 | tee "$OUT/train.log"
done

echo "=== aggregate ==="
python - <<'PYEOF'
import json, glob, numpy as np
runs = sorted(glob.glob('runs/pilot/vibednet__ce__s*/metrics.json'))
print(f'collected {len(runs)} runs')
acc, f1, auc = [], [], []
per_task = {t: [] for t in ('boredom','engagement','confusion','frustration')}
for r in runs:
    m = json.load(open(r))['test']
    acc.append(m['mean_accuracy']); f1.append(m['mean_macro_f1']); auc.append(m['mean_auc'])
    for t_entry in m['per_task']:
        per_task[t_entry['task']].append(t_entry['accuracy'])
print(f"AvgAcc  : {np.mean(acc)*100:.2f} ± {np.std(acc)*100:.2f}")
print(f"Macro-F1: {np.mean(f1):.3f} ± {np.std(f1):.3f}")
print(f"Macro-AUC: {np.mean(auc):.3f} ± {np.std(auc):.3f}")
for t in ('boredom','engagement','confusion','frustration'):
    print(f"  {t:>12s}: {np.mean(per_task[t])*100:.2f} ± {np.std(per_task[t])*100:.2f}")
PYEOF
