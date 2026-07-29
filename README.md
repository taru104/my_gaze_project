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

Most people who want gaze tracking want exactly this: **the point on the screen you're looking at.** The few open webcam options (WebGazer, EyeTrax) give that, but coarsely (several cm and up) and they jump around under head motion. Tgaze targets **near-hardware accuracy and *usable stability* from commodity hardware**, with **generalization (works for anyone)** treated as a first-class goal.

- 🎯 **~1.3–1.6 cm** at screen center, live webcam *(personal 9-point calibration)*
- 👥 **~2 cm** for a **friend who had never used it before** *(live, 9-point)*
- 🪶 **CPU, real-time** — MediaPipe FaceLandmarker + scikit-learn (no deep net at runtime)
- 🧊 **Stable, not just accurate** — the default model fuses eye geometry with a compressed eye-image patch, which roughly **halves wild off-screen jumps** vs geometry alone
- 🔒 **Private by design** — everything runs locally; nothing is uploaded

## 📊 Key results

**Personal accuracy** — live webcam, leave-one-point-out (the on-screen HUD number):

| Model | Median error | Off-screen jump rate |
|---|---:|---:|
| Geometry only (16-D) | **1.6 cm** | ~12 % |
| **Geometry + eye-image (default)** | 1.3 – 2.1 cm | **~4–6 %** |

> The two models trade places depending on what you measure. Geometry alone often wins the *point-accuracy* number by a hair, but the eye-image model is far **steadier in real use** — it flings the cursor off-screen about **half as often**. In side-by-side testing the eye-image version simply *felt* better to use, and the logs confirmed it (details below). **That's why it is the default.**

**Generalization** — MPIIFaceGaze, 15 subjects, offline, honest splits:

| Setting | Median error |
|---|---:|
| Unseen person, **zero calibration** | ~6 cm |
| Same person, calibrated (geometry) | ~4.0 cm |
| Same person, calibrated (**+ eye-image**) | **~4.5 cm***  |

<sub>*Verified with strict time-based / block splits and leave-one-person-out — the eye-image gain survives the hard evaluations, so it is real signal, not train/test leakage.</sub>

## 🔍 How it works — and two insights that shaped it

```
webcam frame
  └─ MediaPipe FaceLandmarker (478 landmarks + iris)
       ├─ 16-D geometric feature
       │    iris(x,y) L/R, head pitch/yaw/roll, inter-eye distance,
       │    eye-openness, iris vertical position, iris size, iris aspect
       │    (iris normalized to each eye's corners → distance/translation invariant)
       └─ (default) eye-image patch
            each eye warped to a canonical 48×32 crop by its corners,
            CLAHE contrast-normalized, compressed to 16 PCA components
       └─ Huber regression (robust) → on-screen point
       └─ One-Euro filter → smooth gaze cursor
```

**Insight 1 — the bug that unlocked accuracy.** An early feature normalized the iris against the **image center**, quietly turning it into a *face-position sensor*: its correlation with head location was **0.98**, with actual gaze only **0.48**. Switching to **eye-corner normalization** made it distance- and translation-invariant and cut center error from **~9 cm → ~1.4 cm**. *The iris carries the gaze signal; head pose is only a correction term.*

**Insight 2 — the metric and the feel disagreed, and the feel was right.** Adding the eye-image patch *slightly hurt* the point-accuracy number in some sessions, so by the number alone you'd reject it. But in hands-on use it was noticeably more stable, so I logged both versions and measured the real culprit — **wild off-screen jumps** — and found the eye-image model cut them roughly in half. A single offline metric can miss what actually makes a tool usable; the eye-image model shipped as the default because it *behaves* better, and the logs backed that up. *(A new user — a friend testing it cold — reproduced the same preference and ~2 cm accuracy.)*

## 🚀 Quick start

```bash
pip install -r requirements.txt
python main.py            # default: geometry + eye-image (steadier)
python main_16d.py        # geometry only (the leaner baseline)
```

| Key | Action |
|---|---|
| `C` | Calibrate — follow the dot through **9 points, clockwise** (easy to track on first try) |
| `M` | Multi-pose calibrate (look at each dot while slowly rotating your head) |
| `R` / `Q` | Reset / Quit |

The top-left preview shows a live **head-pose arrow**; the HUD shows a leave-one-out **cm error** so you can compare setups. Switch the default in `config.py` (`USE_APPEARANCE`).

## 🧪 Method & reproduction (research notes)

- **Features** (`features.py`, `rich16d.py`) — 16-D geometry: both-eye iris positions normalized to eye corners + head pose + eye-openness / iris size / iris aspect. `appearance.py` adds the canonical eye-image patch → PCA-16.
- **Calibration** (`calibration.py`, `appearance.py`) — robust Huber regression on the fused feature. No deep net; fits instantly on CPU. Optional tap-based dynamic correction.
- **Why not more dimensions?** Extensive ablations (`experiments/`) show that *adding hand-crafted features to the 16-D vector saturates*, and that raising the eye-image resolution helps **only** when tightly PCA-compressed — more raw dimensions overfit on a sparse calibration. Overfitting was the constant adversary, so every gain is checked person-independently.
- **Generalization** (`benchmarks/`) — person-independent evaluation on **MPIIFaceGaze** (15 subjects) with the same features; the gaze structure transfers to a new camera/person (absolute screen geometry is handled by the light per-user calibration).
- Numbers above come from live sessions and the scripts in `benchmarks/` + `experiments/`.

## 🗺️ Roadmap

- [x] Real-webcam validation on a second person (cold first-time user).
- [ ] **Denser / continuous calibration** — give the eye-image model enough gaze coverage to also win the point-accuracy number, not just stability.
- [ ] **Save eye-crops during use** (privacy-scoped) so the eye-image model can be tuned on real sessions, not just MPIIFaceGaze.
- [ ] **Calibration-light onboarding** — generic base + a few taps so a new user works in seconds.
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
- Compared against the [EyeTrax](https://github.com/ck-zhang/eyetrax) and [GazeTracking](https://github.com/antoinelame/GazeTracking) open-source projects.
- Research used **MPIIFaceGaze** (Zhang et al., CVPRW 2017; research license) and draws on **GazeCapture** and **ETH-XGaze**. These research datasets are **not** used in any commercial build.
