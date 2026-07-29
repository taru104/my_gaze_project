"""画像クリップ版を明示起動（config を問わず必ず 16D幾何 + 目パッチPCA16）。
既定(config.USE_APPEARANCE=True)なら `python main.py` でも同じ版が起動する。
"""
import sys

from main import GazeApp

if __name__ == '__main__':
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        GazeApp(cam_id=cam_id, win_w=1280, win_h=720, use_appearance=True).run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
