"""
GazeCapture dataset utilities.
Raw images are NEVER fully loaded into RAM:
  - GazeCaptureRawIndex        : metadata index only (no images in memory)
  - GazeCaptureFeatureDataset  : loads pre-extracted feature cache via mmap

Dataset format reference (from GazeCapture docs):
  faceGrid.json  IsValid = 1 iff Apple detected BOTH face AND eyes (most reliable validity flag)
  dotInfo.json   XPts/YPts = dot center in screen points from top-left
                 XCam/YCam = dot center in cm from camera center (prediction space)
  screen.json    W/H in points (may differ from pixels due to Retina scaling)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# split name → integer code stored in cache npz
SPLIT_CODE     = {'train': 0, 'val': 1, 'test': 2}
SPLIT_CODE_INV = {v: k for k, v in SPLIT_CODE.items()}


class GazeCaptureRawIndex:
    """
    Scans archive directory and builds a lightweight index of every valid frame.
    NO images are loaded — only paths and label scalars are kept in memory.

    Validity criterion: faceGrid.json IsValid == 1
    (requires Apple detection of BOTH face and eyes)

    Each record: (img_path, x_norm, y_norm, x_cm, y_cm, split_code)
      x_norm = XPts / W  (normalized horizontal, [0,1])
      y_norm = YPts / H  (normalized vertical,   [0,1])
      x_cm, y_cm = XCam, YCam (cm from camera center)
    """

    def __init__(self, archive_dir: str):
        self.archive_dir = Path(archive_dir)
        self.records: list = []
        self._build()

    def _build(self):
        subjects = sorted(os.listdir(self.archive_dir))
        print(f"[Index] Scanning {len(subjects)} subject folders...")
        skipped = 0
        for subj in subjects:
            subj_dir = self.archive_dir / subj / subj
            if not subj_dir.is_dir():
                continue
            try:
                with open(subj_dir / 'info.json') as f:
                    info = json.load(f)
                split = info.get('Dataset', '')
                if split not in SPLIT_CODE:
                    continue
                code = SPLIT_CODE[split]

                with open(subj_dir / 'frames.json') as f:
                    frame_names = json.load(f)
                with open(subj_dir / 'dotInfo.json') as f:
                    dot = json.load(f)
                with open(subj_dir / 'screen.json') as f:
                    screen = json.load(f)
                with open(subj_dir / 'faceGrid.json') as f:
                    grid = json.load(f)
            except Exception as e:
                skipped += 1
                continue

            orientations = screen.get('Orientation', None)  # per-frame orientation (may be absent)
            n = len(frame_names)
            for i in range(n):
                # Use faceGrid IsValid: requires BOTH face AND eye detections
                if not grid['IsValid'][i]:
                    continue
                img_path = subj_dir / 'frames' / frame_names[i]
                if not img_path.exists():
                    continue
                W = screen['W'][i]
                H = screen['H'][i]
                if W == 0 or H == 0:
                    continue
                ori = int(orientations[i]) if orientations is not None else 1

                x_raw = float(dot['XPts'][i]) / W   # [0,1] in device screen coords
                y_raw = float(dot['YPts'][i]) / H

                # Normalize all frames to portrait coordinate space.
                # iOS orientation codes:
                #   1 = Portrait (home bottom)              → no rotation
                #   2 = Portrait upside-down (home top)     → 180°
                #   3 = LandscapeLeft  (device CCW, image needs 90° CW)
                #   4 = LandscapeRight (device CW,  image needs 90° CCW)
                if ori == 1:
                    x_norm, y_norm = x_raw, y_raw
                elif ori == 2:
                    x_norm, y_norm = 1.0 - x_raw, 1.0 - y_raw
                elif ori == 3:
                    x_norm, y_norm = 1.0 - y_raw, x_raw
                else:   # ori == 4
                    x_norm, y_norm = y_raw, 1.0 - x_raw

                x_cm = float(dot['XCam'][i])
                y_cm = float(dot['YCam'][i])
                # record tuple: (img_path, x_norm, y_norm, x_cm, y_cm, split_code, orientation)
                self.records.append((str(img_path), x_norm, y_norm, x_cm, y_cm, code, ori))

        print(f"[Index] {len(self.records)} valid frames  ({skipped} subjects skipped)")

    def __len__(self):
        return len(self.records)

    def by_split(self, split: str) -> list:
        code = SPLIT_CODE[split]
        return [r for r in self.records if r[5] == code]

    def orientation_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(r[6] for r in self.records))


class GazeCaptureFeatureDataset(Dataset):
    """
    PyTorch Dataset over a pre-extracted feature cache (.npz).
    numpy mmap_mode='r' — only requested rows are paged into RAM.

    Cache schema (written by gazeCapture_validate.py):
        X          (N, 7)  float32  MediaPipe 7D features
        y_norm     (N, 2)  float32  [x_norm, y_norm] in [0,1]
        y_cm       (N, 2)  float32  [x_cm, y_cm] in cm
        split_code (N,)    int32    0=train 1=val 2=test
    """

    def __init__(self, cache_path: str, split: str = 'train', target: str = 'norm'):
        """
        Args:
            cache_path : path to .npz cache
            split      : 'train' | 'val' | 'test'
            target     : 'norm' → y_norm  |  'cm' → y_cm
        """
        cache = np.load(cache_path, mmap_mode='r')
        mask = cache['split_code'] == SPLIT_CODE[split]
        self.X = np.array(cache['X'][mask],     dtype=np.float32)
        self.y = np.array(
            cache['y_norm'][mask] if target == 'norm' else cache['y_cm'][mask],
            dtype=np.float32,
        )
        print(f"[Dataset] split={split}  n={len(self.X)}  target={target}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
