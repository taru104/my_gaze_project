"""Tgaze — accurate, calibration-light webcam eye tracking.

Five lines to a gaze point on screen:

    import cv2
    from tgaze import GazeTracker

    tracker = GazeTracker()
    tracker.calibrate()                     # fullscreen 9-point calibration
    x, y = tracker.predict(frame)           # normalized screen coords in [0, 1]

Everything runs locally on the CPU (MediaPipe + scikit-learn). No GPU, no upload.
"""
from .tracker import GazeTracker, GazePoint

__all__ = ["GazeTracker", "GazePoint"]
__version__ = "0.1.0"
