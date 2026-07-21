"""
生ランドマークロガー — 将来どんな特徴次元にも対応できるよう、
MediaPipeの全478ランドマーク(x,y,z)を1フレームずつ丸ごと保存する。

なぜ生で保存するか:
  加工後の値(pitch/X_feat等)だけ保存すると、後から新しい特徴(虹彩楕円/新しい眼形状/
  N次元拡張)を作りたくても再計算できず詰む(2026-07-16に実際に詰まった)。
  生ランドマークさえ残せば、後からオフラインで extract_rich 相当を再実行して
  任意次元の特徴を再構成できる。

形式(クラッシュ耐性のため各フレームflushで追記する固定長バイナリ):
  logs/session_<id>_landmarks.bin       … float32 の固定長レコード列
  logs/session_<id>_landmarks.meta.json … 形式の自己記述(下記loaderが読む)

1レコード = HEADER(9 float32) + landmarks(N_LM*3 float32):
  [0] frame_idx     … CSVログの行と対応づけるグローバル通し番号
  [1] time_s        … セッション開始からの秒
  [2] img_w         … このフレームの幅(px)。正規化座標→px変換に必須
  [3] img_h         … 高さ(px)
  [4] has_target    … 1.0=キャリブ等で正解ターゲットあり / 0.0=なし
  [5] target_x      … 正解の視線ターゲット(正規化[0,1])。無ければNaN
  [6] target_y
  [7] gaze_x        … その時の推定視線(正規化[0,1])。無ければNaN(参考用)
  [8] gaze_y
  [9:] landmarks    … lm0.x,lm0.y,lm0.z, lm1.x,... (MediaPipe正規化座標。x*img_wでpx)

再構成の仕方(オフライン):
  from raw_landmark_logger import load_raw_landmarks
  d = load_raw_landmarks("logs/session_<id>_landmarks")
  d['landmarks'][k]  # (478,3) k番目フレームの生ランドマーク → 任意の特徴を再計算
"""
import json
import struct
from pathlib import Path
import numpy as np

N_LM = 478          # MediaPipe FaceLandmarker(refine=True)の総点数(虹彩含む)
N_COORD = 3         # x, y, z
HEADER = 9          # frame_idx,time_s,img_w,img_h,has_target,tx,ty,gaze_x,gaze_y
RECORD_FLOATS = HEADER + N_LM * N_COORD          # = 1443
RECORD_BYTES = RECORD_FLOATS * 4                 # float32


class RawLandmarkLogger:
    """全ランドマークをフレーム毎に追記保存する。使い捨てず必ず close() する。"""

    def __init__(self, base_path: Path):
        """base_path: 拡張子なしのパス。 .bin と .meta.json を作る。"""
        base_path = Path(base_path)
        self.bin_path  = base_path.with_suffix(".bin")
        self.meta_path = Path(str(base_path) + ".meta.json")
        self._f = open(self.bin_path, "wb")
        self._n = 0
        meta = {
            "format": "fixed-length float32 records, little-endian",
            "n_landmarks": N_LM, "n_coord": N_COORD,
            "header_fields": ["frame_idx", "time_s", "img_w", "img_h",
                              "has_target", "target_x", "target_y", "gaze_x", "gaze_y"],
            "record_floats": RECORD_FLOATS, "record_bytes": RECORD_BYTES,
            "landmark_layout": "lm0.x,lm0.y,lm0.z,lm1.x,... (MediaPipe正規化座標)",
            "note": "px変換は x*img_w, y*img_h。zは相対深度。詳細は raw_landmark_logger.py。",
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    def log(self, frame_idx, time_s, img_w, img_h, landmarks,
            target=None, gaze=None):
        """1フレーム分を追記。landmarks=MediaPipeのlms(len>=478, .x/.y/.z)。"""
        if landmarks is None or len(landmarks) < N_LM:
            return  # 顔未検出フレームは生ログをスキップ(frame_idxで欠番として追える)
        buf = np.empty(RECORD_FLOATS, dtype=np.float32)
        has_t = 1.0 if target is not None else 0.0
        tx, ty = (float(target[0]), float(target[1])) if target is not None else (np.nan, np.nan)
        gx, gy = (float(gaze[0]),   float(gaze[1]))   if gaze   is not None else (np.nan, np.nan)
        buf[:HEADER] = [frame_idx, time_s, img_w, img_h, has_t, tx, ty, gx, gy]
        # ランドマーク(x,y,z)を平坦化して詰める
        lm = np.empty((N_LM, N_COORD), dtype=np.float32)
        for i in range(N_LM):
            p = landmarks[i]
            lm[i, 0] = p.x; lm[i, 1] = p.y; lm[i, 2] = p.z
        buf[HEADER:] = lm.ravel()
        self._f.write(buf.tobytes())
        self._n += 1
        if self._n % 30 == 0:
            self._f.flush()   # 30フレーム(約1秒)毎にflush。毎フレームのディスク同期を避けfps確保

    @property
    def n_written(self) -> int:
        return self._n

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def load_raw_landmarks(base_path) -> dict:
    """保存した生ランドマークを読み込む。返り値の 'landmarks' は (N,478,3)。"""
    base_path = str(base_path)
    bin_path = base_path if base_path.endswith(".bin") else base_path + ".bin"
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % RECORD_FLOATS != 0:
        raw = raw[: raw.size - (raw.size % RECORD_FLOATS)]  # 途中で切れた末尾を捨てる
    recs = raw.reshape(-1, RECORD_FLOATS)
    hdr = recs[:, :HEADER]
    lms = recs[:, HEADER:].reshape(-1, N_LM, N_COORD)
    return {
        "frame_idx": hdr[:, 0].astype(np.int64),
        "time_s":    hdr[:, 1],
        "img_w":     hdr[:, 2], "img_h": hdr[:, 3],
        "has_target": hdr[:, 4].astype(bool),
        "target":    hdr[:, 5:7],
        "gaze":      hdr[:, 7:9],
        "landmarks": lms,
        "n": len(recs),
    }
