# logs/ — the sessions Tgaze was evaluated on

These are the **real webcam sessions** behind the live-accuracy numbers in the main README.
They are published so that "we measured 1.3–2.1 cm on a live webcam" is not something you
have to take on faith — you can see the actual per-frame record.

## What a file contains

One row per camera frame:

| column | meaning |
|---|---|
| `time_s` | seconds since the session started |
| `gaze_x`, `gaze_y` | the estimate Tgaze showed, screen-normalized `[0,1]` (after smoothing) |
| `raw_x`, `raw_y` | the regression output before the One-Euro filter |
| `pitch_deg`, `yaw_deg` | head pose from solvePnP |
| `ear`, `blink` | eye aspect ratio and blink flag |
| `face_detected`, `calibrated` | pipeline state |
| `calib_point_idx`, `calib_target_x/y` | during calibration: which point, and where it was (this is the ground truth) |
| `X_feat`, `Y_feat` | legacy debug features (superseded — see Insight 1 in the main README) |
| `loo_euc_cm`, `loo_euc_x_cm`, `loo_euc_y_cm` | leave-one-point-out error shown on the HUD |

**No images and no face landmarks are in these files.** They are per-frame numbers:
gaze coordinates, head angles, and calibration state.

## What is *not* here, and why

The experiments in `experiments/` do not actually read these CSVs — they read
`logs/*_landmarks.bin`, which stores all 478 MediaPipe landmarks per frame so that any
future feature can be recomputed from past recordings. Those files total **~900 MB**, which
is too large to put in a git repository, so they are not published. The CSVs are the
human-readable companion (matched by frame order), not a drop-in replacement.

So: these files let you **inspect and audit** the sessions. They are not sufficient to
re-run `exp35` / `exp66` / `exp67` end-to-end.

## Honest limits of this data

- It is **one person** (the author) plus one friend session — not a dataset, not a user study.
- Sessions vary in length, lighting, distance and head motion; they were recorded during
  development, not under a controlled protocol.
- Ground truth exists only where `calib_target_x/y` is filled in (calibration points and
  click-taps). Everything else is unlabeled usage.
- `session_20260801_181937.csv.gz` is gzip-compressed because it is a very long session
  (148 MB uncompressed). `gzip -d` it to read.

For the objective, multi-person numbers, see the MPIIFaceGaze results in
[`experiments/REPORT8_metrics.md`](../experiments/REPORT8_metrics.md).
