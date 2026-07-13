"""Build the shadow label subset for the frame-sampling check (paper Table 2).

Creates {OUT}/Labels/{Train,Validation,Test}Labels.csv with a fixed random
third of Train/Validation (seed 42) and the full Test split, plus the flat
clip list consumed by preprocess_uniform16.py.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("DAISEE_ROOT", "/path/to/DAiSEE")
OUT = os.environ.get("SAMPLING_CHECK_ROOT", os.path.join(os.path.dirname(ROOT), "uni16")) + "/shadow_root/Labels"
os.makedirs(OUT, exist_ok=True)
clips = []
for split, frac in [("Train", 1 / 3), ("Validation", 1 / 3), ("Test", 1.0)]:
    df = pd.read_csv(f"{ROOT}/Labels/{split}Labels.csv")
    df.columns = [c.strip() for c in df.columns]
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=42).sort_index()
    df.to_csv(f"{OUT}/{split}Labels.csv", index=False)
    clips += [os.path.splitext(str(x).strip())[0] for x in df["ClipID"]]
    print(split, len(df))
open(os.path.join(os.path.dirname(OUT), "clips.txt"), "w").write("\n".join(clips))
print("total clips:", len(clips))
