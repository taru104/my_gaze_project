"""Step1: MPII目領域からリンバス(虹彩/白目境界)に楕円フィット→ arm A/B/C 用の特徴を抽出。
放射状レイ+サブピクセル勾配ピーク+当たり判定+RANSAC的トリム+fitEllipse。失敗はMediaPipeにフォールバック(率を記録)。
rich16d/main等は触らず、_geo_normalizeはimportのみ。 出力: cache/mpii_limbus{tag}.npz
Usage:
  .venv/Scripts/python.exe experiments/exp62_limbus_extract.py --test
  .venv/Scripts/python.exe experiments/exp62_limbus_extract.py --limit 300           # scale1.0
  .venv/Scripts/python.exe experiments/exp62_limbus_extract.py --limit 300 --scale 0.5 --tag _s050
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from rich16d import rich_16d_from_lms, lms_to_array, _geo_normalize
import scipy.io as sio

MPII = ROOT / "MPIIFaceGaze"; MODEL = ROOT / "face_landmarker.task"
L_IRIS=[468,469,470,471,472]; R_IRIS=[473,474,475,476,477]
L_IN,L_OUT=133,33; R_IN,R_OUT=362,263
UP=4  # 拡大率
# レイの弧: 水平±55°(左右)。±40°だと縦(短軸)が拘束不足で楕円が潰れる。±55はまつ毛回避と縦拘束の折衷。
ANGS = np.radians(np.array(list(range(-55,56,3))+list(range(125,236,3)), float))

def make_lm():
    return mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=mp_vision.RunningMode.IMAGE, num_faces=1))
def screen(pid):
    ss=sio.loadmat(str(MPII/pid/"Calibration"/"screenSize.mat")); return float(ss["width_pixel"][0][0]),float(ss["height_pixel"][0][0])

def _sample(img,x,y):
    x0,y0=int(np.floor(x)),int(np.floor(y))
    if x0<0 or y0<0 or x0+1>=img.shape[1] or y0+1>=img.shape[0]: return np.nan
    dx,dy=x-x0,y-y0
    return float(img[y0,x0]*(1-dx)*(1-dy)+img[y0,x0+1]*dx*(1-dy)+img[y0+1,x0]*(1-dx)*dy+img[y0+1,x0+1]*dx*dy)

def fit_limbus(gray, iris_c, inner, outer):
    """成功: (center_orig(2,), axis_ratio, angle_rad, n_pts)。失敗: None。"""
    eye_w = np.linalg.norm(outer-inner)
    if eye_w < 6: return None
    S = int(round(1.6*eye_w)); out = S*UP
    if out < 24 or out > 2000: return None
    ang_deg = np.degrees(np.arctan2((outer-inner)[1],(outer-inner)[0]))
    M = cv2.getRotationMatrix2D((float(iris_c[0]),float(iris_c[1])), ang_deg, float(UP))
    M[0,2]+=out/2-iris_c[0]; M[1,2]+=out/2-iris_c[1]
    crop = cv2.warpAffine(gray, M, (out,out), flags=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(crop,(5,5),0).astype(np.float32)
    _, scl = cv2.threshold(blur.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 白目=明
    sat = (blur>245).astype(np.uint8)
    cx=cy=out/2.0
    rmin=int(0.12*eye_w*UP); rmax=int(0.42*eye_w*UP)
    if rmax-rmin<5: return None
    pts=[]
    for a in ANGS:
        dxr,dyr=np.cos(a),np.sin(a)
        prof=np.array([_sample(blur,cx+r*dxr,cy+r*dyr) for r in range(rmin,rmax)])
        if np.isnan(prof).sum()>len(prof)*0.3: continue
        prof=np.where(np.isnan(prof), np.nanmean(prof), prof)
        g=np.gradient(prof)
        ip=int(np.argmax(g))
        if g[ip]<=0 or ip<1 or ip>=len(g)-1: continue
        # サブピクセル(放物線頂点)
        d=(g[ip-1]-g[ip+1]); den=(g[ip-1]-2*g[ip]+g[ip+1])
        off=0.5*d/den if abs(den)>1e-6 else 0.0
        off=float(np.clip(off,-1,1))
        rp=rmin+ip+off
        # 暗→明チェック(内側<外側)
        if not (np.nanmean(prof[max(0,ip-4):ip]) < np.nanmean(prof[ip+1:ip+5])): continue
        px,py=cx+rp*dxr,cy+rp*dyr
        ox,oy=int(px+3*dxr),int(py+3*dyr); ix,iy=int(px-3*dxr),int(py-3*dyr)
        if not(0<=ox<out and 0<=oy<out and 0<=ix<out and 0<=iy<out): continue
        # 白目Otsuゲートは撤去(クロップに肌が入りOtsuの明=肌になり誤爆・正常レイを殺すため)。
        # 暗→明・距離・飽和・トリムで外れ値は十分弾ける。
        if sat[int(py),int(px)]!=0: continue        # 飽和(反射)でない
        rorig=rp/UP
        if not (0.2*eye_w <= rorig <= 0.35*eye_w): continue   # 虹彩半径として妥当
        pts.append([px,py])
    if len(pts)<8: return None
    P=np.array(pts,np.float32)
    rr=np.hypot(P[:,0]-cx,P[:,1]-cy); mr=np.median(rr)   # 半径コンセンサス(巨大外れ値=誤検出を除去)
    P=P[np.abs(rr-mr)<0.22*mr]
    if len(P)<8: return None
    # トリム(残差>1.5*中央値を2回除去)して fitEllipse
    for _ in range(2):
        if len(P)<5: return None
        e=cv2.fitEllipse(P)
        (ecx,ecy),(MA,ma),th=e
        a2=max(MA,ma)/2.0; b2=min(MA,ma)/2.0
        if a2<1: return None
        c,s=np.cos(np.radians(th)),np.sin(np.radians(th))
        d=P-np.array([ecx,ecy]); u=(d[:,0]*c+d[:,1]*s)/a2; v=(-d[:,0]*s+d[:,1]*c)/b2
        res=np.abs(np.hypot(u,v)-1.0)
        keep=res<max(0.25, 1.5*np.median(res))
        if keep.sum()<8 or keep.all(): break
        P=P[keep]
    e=cv2.fitEllipse(P); (ecx,ecy),(MA,ma),th=e
    major=max(MA,ma); minor=min(MA,ma); ratio=minor/(major+1e-8)
    # 妥当性
    if not (0.5<=ratio<=1.0): return None
    if not (0.4*eye_w <= (major/UP) <= 0.6*eye_w): return None
    if not (0<=ecx<out and 0<=ecy<out): return None
    if len(P)<8: return None
    Minv=cv2.invertAffineTransform(M)
    co=Minv@np.array([ecx,ecy,1.0]); center_orig=np.array([co[0],co[1]])
    ang_e=np.radians(th)   # fitEllipseの傾き(0-180)。sin/cosは2倍角で周期解消
    return center_orig, float(ratio), float(ang_e), int(len(P))

def main():
    test="--test" in sys.argv
    limit=20 if test else 300
    if "--limit" in sys.argv: limit=int(sys.argv[sys.argv.index("--limit")+1])
    scale=1.0
    if "--scale" in sys.argv: scale=float(sys.argv[sys.argv.index("--scale")+1])
    tag=""
    if "--tag" in sys.argv: tag=sys.argv[sys.argv.index("--tag")+1]
    lm=make_lm()
    rows=dict(X16=[],A=[],B=[],fbL=[],fbR=[],iris=[],y=[],pid=[])
    parts=["p00"] if test else [f"p{i:02d}" for i in range(15)]
    t0=time.time(); nimg=0
    for pid in parts:
        try: wpx,hpx=screen(pid)
        except Exception as e: print(f"{pid}:{e}"); continue
        lines=open(MPII/pid/f"{pid}.txt").read().strip().splitlines()[:limit]
        okc=0
        for ln in lines:
            f=ln.split(); img=cv2.imread(str(MPII/pid/f[0]))
            if img is None: continue
            if scale!=1.0: img=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            h,w=img.shape[:2]
            res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
            if not res.face_landmarks: continue
            arr=lms_to_array(res.face_landmarks[0])
            X=rich_16d_from_lms(arr,w,h)
            if X is None or not np.isfinite(X).all(): continue
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            def P(i): return np.array([arr[i,0]*w, arr[i,1]*h])
            Lc=np.mean([P(i) for i in L_IRIS],axis=0); Rc=np.mean([P(i) for i in R_IRIS],axis=0)
            # arm A/B の初期値=MediaPipe(=baseline)。楕円成功時のみ差し替え。
            A=[X[0],X[1],X[2],X[3]]        # Lx,Ly,Rx,Ry
            B=[X[14],X[15],0.0,0.0,0.0,0.0] # Lasp,Rasp,Lsin2,Lcos2,Rsin2,Rcos2
            fb=[1,1]
            for e_i,(IR,IN,OUT,cc) in enumerate([(L_IRIS,L_IN,L_OUT,Lc),(R_IRIS,R_IN,R_OUT,Rc)]):
                r=fit_limbus(gray, cc, P(IN), P(OUT))
                if r is None: continue
                center,ratio,ang,npts=r
                Ln=_geo_normalize(center, P(IN), P(OUT))
                if e_i==0: A[0],A[1]=float(Ln[0]),float(Ln[1]); B[0]=ratio; B[2]=np.sin(2*ang); B[3]=np.cos(2*ang)
                else:      A[2],A[3]=float(Ln[0]),float(Ln[1]); B[1]=ratio; B[4]=np.sin(2*ang); B[5]=np.cos(2*ang)
                fb[e_i]=0
            l_diam=2.0*np.mean([np.linalg.norm(P(i)-Lc) for i in L_IRIS[1:]])
            rows["X16"].append(X); rows["A"].append(A); rows["B"].append(B)
            rows["fbL"].append(fb[0]); rows["fbR"].append(fb[1]); rows["iris"].append(l_diam)
            rows["y"].append([float(f[1])/wpx, float(f[2])/hpx]); rows["pid"].append(pid); okc+=1; nimg+=1
        fbrate=100*(np.mean(rows["fbL"][-okc:])+np.mean(rows["fbR"][-okc:]))/2 if okc else 0
        print(f"{pid}: {okc}枚  FB率={fbrate:.0f}%  ({nimg/(time.time()-t0+1e-9):.1f}img/s)",flush=True)
    out=ROOT/"cache"/(f"mpii_limbus{'_test' if test else tag}.npz")
    np.savez_compressed(str(out), X16=np.array(rows["X16"],np.float32), A=np.array(rows["A"],np.float32),
        B=np.array(rows["B"],np.float32), fbL=np.array(rows["fbL"]), fbR=np.array(rows["fbR"]),
        iris=np.array(rows["iris"],np.float32), y=np.array(rows["y"],np.float32), pid=np.array(rows["pid"]))
    fball=100*(np.array(rows["fbL"]).mean()+np.array(rows["fbR"]).mean())/2
    print(f"[Done] {nimg}枚 全体FB率={fball:.1f}% iris中央値={np.median(rows['iris']):.1f}px → {out}",flush=True)

if __name__=="__main__": main()
