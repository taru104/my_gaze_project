"""
豊富特徴(14D)の抽出。先頭7次元は既存7Dと同一, 末尾7次元が追加特徴。

追加特徴の狙い(横向き精度向上):
  [7]  roll          頭部ロール(rad)。斜め顔での眼配置補正。
  [8]  L_EAR         左眼アスペクト比(縦/横)=まぶた開き。垂直視線の見え方を規定。
  [9]  R_EAR         右眼開き。
  [10] L_iris_vert   左虹彩の眼内垂直位置(まぶた基準,眼高で正規化)。
                     現行7Dは横軸のみで正規化しており垂直視線が弱い。それを補う。
  [11] R_iris_vert   右虹彩の垂直位置。
  [12] L_iris_diam   左虹彩径/眼幅。per-eye深度・スケール手掛かり。
  [13] R_iris_diam   右虹彩径/眼幅。

全317k(train+val+test)フレームを抽出し cache/rich_features_cache.npz に保存。
X[:, :7] は既存7Dと厳密一致 → 7D vs rich を同一フレームで比較できる。

Usage:
    .venv/Scripts/python.exe benchmarks/extract_rich_features.py --test    # 少数検証
    .venv/Scripts/python.exe benchmarks/extract_rich_features.py           # フル抽出
"""
import sys, io, os, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, str(Path(__file__).parent))
from gazeCapture_dataset import GazeCaptureRawIndex

ROOT_DIR    = Path(__file__).parent.parent
ARCHIVE_DIR = ROOT_DIR / 'archive'
MODEL_PATH  = ROOT_DIR / 'face_landmarker.task'
OUT_PATH    = ROOT_DIR / 'cache' / 'rich_features_cache.npz'
CK_PATH     = ROOT_DIR / 'cache' / 'rich_features_checkpoint.npz'
CK_EVERY    = 20_000

# ─── landmark indices ─────────────────────────────────────────────────────
_FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0],
], dtype=np.float64)
_FACE_2D_IDX = [1, 152, 33, 263, 61, 291]
_LEFT_IRIS   = [468, 469, 470, 471, 472]
_RIGHT_IRIS  = [473, 474, 475, 476, 477]
_L_IN, _L_OUT = 133, 33
_R_IN, _R_OUT = 362, 263
_L_UP, _L_LO = 159, 145   # 左眼 上/下まぶた
_R_UP, _R_LO = 386, 374   # 右眼 上/下まぶた
_DIST = np.zeros((4, 1), dtype=np.float64)


def _geo_normalize(pupil, inner, outer):
    vec = outer - inner
    length = np.linalg.norm(vec) + 1e-8
    center = (inner + outer) / 2.0
    rel = pupil - center
    ang = np.arctan2(vec[1], vec[0])
    ca, sa = np.cos(-ang), np.sin(-ang)
    rot = np.array([[ca, -sa], [sa, ca]])
    return (rot @ rel) / length


def _rotate_to_portrait(img, ori):
    if ori == 1: return img
    if ori == 2: return cv2.rotate(img, cv2.ROTATE_180)
    if ori == 3: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def extract_rich(bgr, landmarker, orientation=1):
    """(14,) float32 or None"""
    bgr = _rotate_to_portrait(bgr, orientation)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    if len(lms) < 478:
        return None

    def P(i):
        return np.array([lms[i].x * w, lms[i].y * h])

    L_px = np.mean([P(i) for i in _LEFT_IRIS], axis=0)
    R_px = np.mean([P(i) for i in _RIGHT_IRIS], axis=0)

    # 頭部姿勢
    f = float(w)
    cam = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    face_2d = np.array([[lms[i].x * w, lms[i].y * h] for i in _FACE_2D_IDX], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) * np.pi / 180.0
    yaw   = float(angles[1]) * np.pi / 180.0
    roll  = float(angles[2]) * np.pi / 180.0
    if pitch > np.pi/2: pitch = np.pi - pitch
    elif pitch < -np.pi/2: pitch = -np.pi - pitch

    # 既存7D
    l_in, l_out = P(_L_IN), P(_L_OUT)
    r_in, r_out = P(_R_IN), P(_R_OUT)
    L_n = _geo_normalize(L_px, l_in, l_out)
    R_n = _geo_normalize(R_px, r_in, r_out)
    dist = float(np.linalg.norm(L_px - R_px) / w)

    # ── 追加特徴 ──
    l_up, l_lo = P(_L_UP), P(_L_LO)
    r_up, r_lo = P(_R_UP), P(_R_LO)
    l_width = np.linalg.norm(l_out - l_in) + 1e-8
    r_width = np.linalg.norm(r_out - r_in) + 1e-8
    l_height = np.linalg.norm(l_up - l_lo)
    r_height = np.linalg.norm(r_up - r_lo)
    L_EAR = l_height / l_width
    R_EAR = r_height / r_width
    # 虹彩の眼内垂直位置(まぶた中心基準, 眼高正規化)
    l_eye_cy = (l_up[1] + l_lo[1]) / 2.0
    r_eye_cy = (r_up[1] + r_lo[1]) / 2.0
    L_iris_vert = (L_px[1] - l_eye_cy) / (l_height + 1e-8)
    R_iris_vert = (R_px[1] - r_eye_cy) / (r_height + 1e-8)
    # 虹彩径 / 眼幅
    l_diam = 2.0 * np.mean([np.linalg.norm(P(i) - L_px) for i in _LEFT_IRIS[1:]])
    r_diam = 2.0 * np.mean([np.linalg.norm(P(i) - R_px) for i in _RIGHT_IRIS[1:]])
    L_iris_diam = l_diam / l_width
    R_iris_diam = r_diam / r_width

    # 虹彩アスペクト比 (NotebookLM Q2: 楕円の短長軸比。横向きで虹彩が扁平化=前縮み)
    # エッジ4点(469-472等)の対向距離2軸から短軸/長軸を計算。横向きほど<1になる。
    def iris_aspect(idx, center):
        ax1 = np.linalg.norm(P(idx[1]) - P(idx[3]))   # 一方の対向軸
        ax2 = np.linalg.norm(P(idx[2]) - P(idx[4]))   # 直交する対向軸
        lo, hi = min(ax1, ax2), max(ax1, ax2)
        return lo / (hi + 1e-8)
    L_aspect = iris_aspect(_LEFT_IRIS, L_px)
    R_aspect = iris_aspect(_RIGHT_IRIS, R_px)

    return np.array([L_n[0], L_n[1], R_n[0], R_n[1], pitch, yaw, dist,
                     roll, L_EAR, R_EAR, L_iris_vert, R_iris_vert,
                     L_iris_diam, R_iris_diam, L_aspect, R_aspect], dtype=np.float32)


def make_landmarker():
    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    return mp_vision.FaceLandmarker.create_from_options(options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="少数フレームで検証のみ")
    args = ap.parse_args()

    index = GazeCaptureRawIndex(str(ARCHIVE_DIR))
    records = index.records
    n_total = len(records)
    lm = make_landmarker()

    if args.test:
        print(f"[Test] 先頭300レコードで検証")
        feats = []
        for rec in records[:300]:
            img = cv2.imread(rec[0])
            if img is None: continue
            fr = extract_rich(img, lm, rec[6] if len(rec) > 6 else 1)
            if fr is not None: feats.append(fr)
        lm.close()
        F = np.array(feats)
        print(f"  抽出成功 {len(F)}/300")
        names = ['Lx','Ly','Rx','Ry','Pitch','Yaw','dist',
                 'roll','L_EAR','R_EAR','L_ivert','R_ivert','L_idiam','R_idiam',
                 'L_aspect','R_aspect']
        for i, nm in enumerate(names):
            c = F[:, i]
            print(f"    [{i:>2}] {nm:<8}: mean={c.mean():+.3f} std={c.std():.3f} "
                  f"range=[{c.min():+.2f},{c.max():+.2f}]")
        # NaN/Inf チェック
        bad = np.sum(~np.isfinite(F))
        print(f"  NaN/Inf: {bad} (0であるべき)")
        return

    # フル抽出
    X_list, yn_list, yc_list, sc_list = [], [], [], []
    start = 0
    if CK_PATH.exists():
        ck = np.load(str(CK_PATH))
        X_list = list(ck['X']); yn_list = list(ck['y_norm'])
        yc_list = list(ck['y_cm']); sc_list = list(ck['split_code'])
        start = int(len(X_list) / 0.93)
        print(f"[Resume] {len(X_list)} frames, start~{start}")
    else:
        print(f"[Fresh] {n_total} frames to process")

    n_ok = len(X_list); n_fail = 0; t0 = time.time(); next_ck = n_ok + CK_EVERY
    for i, rec in enumerate(records[start:], start=start):
        if i > start and (i - start) % 2000 == 0:
            el = time.time() - t0; fps = max((i-start)/el, 0.01)
            print(f"  [{100*i/n_total:5.1f}%] {i}/{n_total} ok={n_ok} fail={n_fail} "
                  f"{fps:.1f}fps ETA {(n_total-i)/fps/60:.1f}min")
        img = cv2.imread(rec[0])
        if img is None: n_fail += 1; continue
        fr = extract_rich(img, lm, rec[6] if len(rec) > 6 else 1)
        if fr is None: n_fail += 1; continue
        X_list.append(fr); yn_list.append([rec[1], rec[2]])
        yc_list.append([rec[3], rec[4]]); sc_list.append(rec[5]); n_ok += 1
        if n_ok >= next_ck:
            np.savez_compressed(str(CK_PATH), X=np.array(X_list, np.float32),
                y_norm=np.array(yn_list, np.float32), y_cm=np.array(yc_list, np.float32),
                split_code=np.array(sc_list, np.int32))
            print(f"  [CK] {n_ok} saved"); next_ck = n_ok + CK_EVERY
    lm.close()
    np.savez_compressed(str(OUT_PATH), X=np.array(X_list, np.float32),
        y_norm=np.array(yn_list, np.float32), y_cm=np.array(yc_list, np.float32),
        split_code=np.array(sc_list, np.int32))
    print(f"\n[Done] {time.time()-t0:.0f}s ok={n_ok} fail={n_fail} → {OUT_PATH} "
          f"({os.path.getsize(OUT_PATH)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
