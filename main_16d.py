"""16D幾何のみ版を明示起動（画像クリップを使わない従来版）。config.USE_APPEARANCE に関係なく必ず16D。
点精度LOOはやや上だが実機では暴走(画面外ジャンプ)しやすい。比較・従来動作の確認用。
"""
import sys

from main import GazeApp

if __name__ == '__main__':
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        GazeApp(cam_id=cam_id, win_w=1280, win_h=720, use_appearance=False).run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
