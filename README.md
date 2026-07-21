<h1 align="center">Tgaze</h1>

<p align="center">
  <b>True Gaze</b> — accurate, calibration-light eye tracking from a single webcam.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/backbone-MediaPipe-orange.svg" alt="mediapipe">
  <img src="https://img.shields.io/badge/runtime-CPU%20realtime-green.svg" alt="cpu">
  <img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="license">
</p>

<p align="center"><i>🎥 demo GIF coming soon</i></p>

---

## TL;DR

Tgaze estimates **where you look on screen** using only a laptop webcam — no infrared, no headset, no GPU.

- 🎯 **1.4 cm** at screen center *(personal 9-point calibration)*
- 🌍 **5.3 cm** for a **completely unseen person, zero calibration** *(person-independent on MPIIFaceGaze, 15 subjects)*
- 🪶 **CPU, real-time** — MediaPipe FaceLandmarker + scikit-learn
- 🔒 **Private by design** — everything runs locally

Existing webcam gaze libraries (e.g. WebGazer.js) are easy but coarse (several cm–10 cm+) and fragile to head motion. Tgaze targets **near-hardware accuracy from commodity hardware**, and treats **generalization (works for anyone)** as a first-class goal — not an afterthought.

## 📊 Key results

**Personal accuracy** — single user, leave-one-point-out:

| Condition | Median error |
|---|---:|
| Center, still head | **1.4 cm** |
| All head poses (multi-pose calibration) | 4.5 cm |

**Generalization** — MPIIFaceGaze, 15 subjects, person-independent:

| Setting | Median error |
|---|---:|
| Unseen person, **zero calibration** | 5.3 cm |
| Unseen person, 50-point affine adaptation | 4.6 cm |
| Personal upper bound (same person) | 4.0 cm |

> The gap between "unseen person" (5.3) and "same person" (4.0) is only **~1.3 cm** — a model trained on *other people* already works well on *you*. That's the core evidence that True Gaze can be **universal**.

## 🔍 How it works — and the one insight that unlocked it

```
webcam frame
  └─ MediaPipe FaceLandmarker  (478 landmarks + iris)
       └─ 7D geometric feature:
            [ L-iris(x,y), R-iris(x,y), pitch, yaw, distance ]
            iris normalized to each eye's corners  →  distance-invariant
       └─ H1 calibration:
            base    = 2nd-order polynomial   (iris → screen point)
            correct = 2nd-order polynomial   (head pose → residual)
       └─ One-Euro filter  →  smooth on-screen gaze point
```

**The bug that unlocked everything.** An early version normalized the iris against the **image center**, which quietly turned the feature into a *face-position sensor* — its correlation with head location was **0.98**, while its correlation with actual gaze was only 0.48. Switching to **eye-corner normalization** made the feature distance- and translation-invariant and cut center error from **~9 cm → 1.4 cm**.

The principle: **the iris carries the gaze signal; head pose is only a correction term.**

## 🚀 Quick start

```bash
pip install -r requirements.txt
python main.py
```

| Key | Action |
|---|---|
| `C` | Calibrate (look at each of 9 dots, keep head still) |
| `M` | Multi-pose calibrate (look at each dot while slowly rotating your head) |
| `R` / `Q` | Reset / Quit |

The top-left preview shows a live **head-pose arrow** so you can see the tracked pose in real time.

## 🧪 Method & reproduction (research notes)

- **Feature (`features.py`, `rich16d.py`)** — 7D: both-eye iris positions normalized to eye corners (distance/translation invariant) + head pose (pitch/yaw) + inter-eye distance.
- **Calibration (`calibration.py`, `H1Calibration`)** — a classic iris→screen 2nd-order polynomial, plus a 2nd-order head-pose correction on the residual. No deep net; runs instantly on CPU.
- **Generalization (`benchmarks/`)** — trained a person-independent model on **MPIIFaceGaze** (15 subjects) with the *same* 7D features, evaluated leave-one-person-out. Transfers to a new camera/person with a light affine adaptation (absolute screen geometry differs per setup; the gaze *structure* transfers).
- All numbers above are reproducible from the scripts in `benchmarks/`.

## 🗺️ Roadmap

- [ ] **Calibration-light onboarding** — ship "generic base + few-tap affine" so a new user works in seconds.
- [ ] **Wide-angle robustness** — ETH-XGaze (±80° head poses) to strengthen extreme yaw.
- [ ] **Synthetic data pipeline** — Blender-rendered faces with ground-truth gaze (commercial-friendly, unlimited poses).
- [ ] pip package + demo GIF & hosted playground.

## 📚 Citation

```bibtex
@software{tgaze,
  title  = {Tgaze: True Gaze — calibration-light webcam eye tracking},
  author = {taru104},
  year   = {2026},
  url    = {https://github.com/taru104/my_gaze_project}
}
```

## License

© taru104. **Noncommercial use only** (CC BY-NC-SA 4.0). Free for personal, research, and educational use **with attribution** (credit **taru104 / Tgaze**). **Commercial use requires prior written permission.** A commercial-friendly track using synthetic / self-collected data is planned.

## Acknowledgments

- Built on [MediaPipe](https://developers.google.com/mediapipe) FaceLandmarker.
- Inspired by the [EyeTrax](https://github.com/ck-zhang/eyetrax) and [GazeTracking](https://github.com/antoinelame/GazeTracking) open-source projects.
- Research used **MPIIFaceGaze** (Zhang et al., CVPRW 2017; research license) and draws on **GazeCapture** and **ETH-XGaze**. These research datasets are **not** used in any commercial build.
