import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)

ROOT = Path("COVIDGR_1.0")
OUT = Path("data")
train_ratio = 0.8

for cls in ["N", "P"]:
    src = ROOT / cls
    files = list(src.glob("*"))
    label = "Normal" if cls == "N" else "COVID"
    train_files, val_files = train_test_split(files, train_size=train_ratio, random_state=42)
    for split, files_split in [("train", train_files), ("val", val_files)]:
        dst = OUT / split / label
        dst.mkdir(parents=True, exist_ok=True)
        for f in files_split:
            shutil.copy2(f, dst / f.name)

print("Dataset prepared:")
print(f"  train/Normal: {(OUT/'train'/'Normal').exists()}  val/Normal: {(OUT/'val'/'Normal').exists()}")
print(f"  train/COVID: {(OUT/'train'/'COVID').exists()}  val/COVID: {(OUT/'val'/'COVID').exists()}")
