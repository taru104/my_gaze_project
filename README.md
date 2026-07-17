# Tgaze

> **T**rue **Gaze** — accurate eye tracking from a single webcam.

Real-time gaze estimation using nothing but your laptop camera.
**~1.4 cm accuracy at screen center** after a quick 9-point calibration — no infrared, no headset, no special hardware.

<!-- TODO: demo.gif here — the single most important thing for adoption -->
<p align="center"><i>(demo GIF coming soon)</i></p>

---

## Why another gaze tracker?

Browser/webcam gaze libraries (e.g. Tgazer.js) are easy to run but **coarse** (several cm to 10 cm+ error) and **fragile to head movement**. Tgaze aims for **near-hardware accuracy from a commodity webcam**:

- 🎯 **~1.4 cm at center** with a 9-point calibration (single user, measured by leave-one-point-out)
- 📏 **Distance-invariant** — iris position is normalized by the inter-eye-corner distance, so moving closer/farther doesn't break it
- 🔄 **Head-pose aware** — pitch / yaw / distance are first-class features, not an afterthought
- 🪶 **Lightweight** — MediaPipe FaceLandmarker + scikit-learn, real time on CPU
- 🔒 **Private by design** — everything runs locally, nothing leaves your machine

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Then:
- **`C`** — calibrate (look at each of the 9 dots, keep your head still)
- **`M`** — multi-pose calibrate (look at each dot while slowly rotating your head — improves accuracy across head angles)
- **`R`** — reset · **`Q`** — quit

The top-left preview shows a live **head-pose arrow** so you can see the tracked pose.

## How it works

```
webcam frame
  └─ MediaPipe FaceLandmarker (478 landmarks + iris)
       └─ 7D geometric feature:
            [ left iris (x,y), right iris (x,y), pitch, yaw, distance ]
            iris positions normalized to each eye's corners  → distance-invariant
       └─ H1 calibration:
            base    = 2nd-order polynomial  (iris → screen point)
            correct = 2nd-order polynomial  (head pose → residual)
       └─ One-Euro filter → smooth on-screen gaze point
```

The key idea: **the iris carries the gaze signal; head pose is a correction term.**
An earlier version normalized the iris against the *image center*, which turned the feature into a face-position sensor (0.98 correlation with head location!). Normalizing against the **eye corners** fixed it and cut center error from ~9 cm to ~1.4 cm.

## Accuracy (single user, leave-one-point-out)

| Condition | Median error |
|---|---|
| Center, head still (`C` calibration) | **~1.4 cm** |
| All head poses (`M` multi-pose calibration) | ~4.5 cm |
| Wide angle (>30° yaw) | ~8 cm *(work in progress)* |

> **Honest status:** center accuracy is excellent and stable. Wide head angles are the current frontier — the *direction* you look is captured well, but absolute error grows. This is a **data** problem (see roadmap), not a model problem.

## Roadmap

- [ ] **Generalization — calibration-light for anyone.** Pretrain a base model on public datasets (GazeCapture / ETH-XGaze) so a new user works out-of-the-box, then fine-tune with a short personal calibration. *Balance: anyone gets decent accuracy; you get 1.4 cm.*
- [ ] **Wide-angle robustness** — more multi-pose data + ETH-XGaze (±80° head poses).
- [ ] **Synthetic data pipeline** — Blender-rendered faces with ground-truth gaze (commercial-friendly, unlimited poses).
- [ ] **pip package** + simple Python API.
- [ ] Demo GIF & hosted playground.

## Project layout

| Path | What |
|---|---|
| `main.py` | Real-time app (calibration UI, live tracking) |
| `features.py` | 7D geometric feature extraction |
| `rich16d.py` | Shared feature definition (single source of truth) |
| `calibration.py` | `H1Calibration` (polynomial + head-pose correction) |
| `estimator.py` | End-to-end pipeline (features → calibration → filter) |
| `benchmarks/` | Research & evaluation scripts |

## License

© taru104. **Noncommercial use only.**
Free for personal, research, and educational use **with attribution** — if you use or build on this project, you must credit **taru104 / Tgaze**.
**Commercial use requires prior written permission from the author.**
(A commercial-friendly track, using models trained on synthetic / self-collected data, is planned — see roadmap.)

## Acknowledgments

- Built on [MediaPipe](https://developers.google.com/mediapipe) FaceLandmarker.
- **Inspired by the [EyeTrax](https://github.com/ck-zhang/EyeTrax) and eyetracker open-source projects** — this work was motivated by them.
- Research and evaluation used the **GazeCapture** dataset (academic/research license) and drew on **ETH-XGaze**. These research datasets are **not** used in any commercial build.
