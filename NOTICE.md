# NOTICE — Tgaze licensing in plain words

Tgaze © 2026 taru104. **All rights reserved.**
The binding terms are in [LICENSE](LICENSE). This file is the plain-language summary;
where the two differ, LICENSE governs.

## Short version

**The code is published to be read, not to be used freely.**
Tgaze is *source-available*, not open source.

| | |
|---|---|
| Reading / reviewing / studying the code | ✅ no permission needed |
| Running, copying, modifying, building on it | ❌ **ask first** |
| Personal, educational, academic, research use | ❌ **ask first** — no category is pre-authorised |
| Commercial use | ❌ **not licensed at all**, and not available on request |

## Asking for permission

Open an issue: https://github.com/taru104/tgaze/issues

Say who you are and what you want to do with it. If it is noncommercial and reasonable,
expect a yes — the point of asking is that I want to know where Tgaze is being used, not to
make it hard. Permission is given in writing, for the purpose you describe.

**Do not send commercial licensing requests.** Commercial use is not offered.

## If you get permission, credit is mandatory

Any resulting software, paper, poster, demo, article, video or talk must visibly say:

> This work uses **Tgaze** by taru104 — https://github.com/taru104/tgaze
> Used with permission. © 2026 taru104.

For academic writing:

```bibtex
@software{tgaze,
  title  = {Tgaze: True Gaze — calibration-light webcam eye tracking},
  author = {taru104},
  year   = {2026},
  url    = {https://github.com/taru104/tgaze}
}
```

Presenting results obtained with Tgaze without naming Tgaze is a breach of the licence.
If you modified it, say what you changed.

## Third-party components are not covered by this

The licence covers only the parts written by taru104. These keep their own licences:

- **MediaPipe** and the bundled `face_landmarker.task` model — © Google LLC, Apache 2.0.
- **Python dependencies** in `pyproject.toml` / `requirements.txt` — each under its own licence.
- **MPIIFaceGaze** (Zhang et al., CVPRW 2017), **GazeCapture**, **ETH-XGaze** — research
  datasets used for *evaluation only*, under their own terms. None are redistributed here.

## A note on earlier versions

Earlier revisions of this repository were published under CC BY-NC-SA 4.0. Creative Commons
licences are irrevocable, so rights validly obtained under it — **for the versions
distributed at that time** — still stand. The terms above govern this and later revisions.

## No warranty

Tgaze is provided "as is", with no warranty of any kind. It is a research project, not a
medical or safety-critical device. Do not use it anywhere a wrong gaze estimate can hurt
someone.
