"""Tgaze quickstart — calibrate once, then draw the gaze point live.

    python examples/quickstart.py

Look at each dot as it appears (9 of them), then a window shows where Tgaze
thinks you are looking. Press `s` to save the calibration so you can skip it
next time, `q` to quit.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tgaze import GazeTracker

MODEL = Path(__file__).resolve().parent.parent / "cache" / "my_calibration.tgaze"


def main() -> None:
    tracker = GazeTracker()                 # 16-D geometry + eye-image patch (default)

    if MODEL.exists():
        tracker.load(MODEL)
        print(f"loaded calibration from {MODEL}")
    else:
        print("starting 9-point calibration — just look at each dot")
        err = tracker.calibrate()
        print(f"calibration done. leave-one-point-out error = {err:.4f} (screen widths)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open the webcam")

    win = "Tgaze — press s to save, q to quit"
    cv2.namedWindow(win)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gaze = tracker.predict(frame)

            h, w = frame.shape[:2]
            view = cv2.flip(frame, 1)       # mirror so it reads like a selfie
            if gaze.face_detected and np.isfinite(gaze.x):
                gx, gy = gaze.as_pixels(w, h)
                cv2.circle(view, (w - gx, gy), 18, (0, 220, 255), 3)
                cv2.putText(view, f"gaze = ({gaze.x:.2f}, {gaze.y:.2f})", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                if not gaze.head_pose_ok:
                    cv2.putText(view, "head far from calibration pose", (12, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            else:
                cv2.putText(view, "no face", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and tracker.is_calibrated:
                MODEL.parent.mkdir(parents=True, exist_ok=True)
                tracker.save(MODEL)
                print(f"saved calibration to {MODEL}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
