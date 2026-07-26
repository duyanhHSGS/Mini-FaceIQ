import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code", "scut"))
import torch, torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from net import Net
from data import mat_process
ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "code", "scut", "checkpoint"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "code", "scut", "scut"), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_AUX_INPUT_SEED = 0
import net as _n
import Mv2_chaatt as _ch
import Mv2attn as _at
def _p1():
    m=_ch.MV2_cattn(); m=nn.DataParallel(m)
    m.module.conv=nn.Conv2d(7,32,3,2,1,bias=False); m.module.fc=nn.Linear(7,7)
    m.module.classifier._modules["1"]=nn.Linear(1280,1280); return m.to(device)
def _p2():
    m=_at.MV2attn(); m=nn.DataParallel(m)
    m.module.attention.L3=nn.Linear(1280,1280); return m.to(device)
_n.parsing_net1=_p1; _n.image_net1=_p2

def make_aux_input():
    """Return the deterministic auxiliary vector expected by the SCUT model."""
    generator = torch.Generator(device=device.type)
    generator.manual_seed(_AUX_INPUT_SEED)
    return torch.randn((1, 7), generator=generator, device=device)
REGIONS = {
    "Left Eye":   [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7],
    "Right Eye":  [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382],
    "Nose":       [1,2,98,327,168,6,197,195,5,4,45,275,220,115,48,64,97,326],
    "Mouth":      [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,
                   37,39,40,185,78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191],
    "L Eyebrow":  [46,53,52,65,55,70,63,105,66,107],
    "R Eyebrow":  [276,283,282,295,285,300,293,334,296,336],
}
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
             152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
DISPLAY = {
    "Left Eye":"Left Eye", "Right Eye":"Right Eye", "Nose":"Nose",
    "Mouth":"Mouth", "L Eyebrow":"L Eyebrow", "R Eyebrow":"R Eyebrow",
    "Skin":"Skin/Cheeks", "Hair":"Hair/Forehead"
}
import mediapipe as mp
_landmarker = None
_MODEL_PATH = os.path.join(ROOT, "face_landmarker.task")
def get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"MediaPipe model not found: {_MODEL_PATH}")
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1, min_face_detection_confidence=0.5))
    return _landmarker
def get_landmarks_mp(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    try:
        result = get_landmarker().detect(mp_img)
        if result.face_landmarks:
            h, w = img_bgr.shape[:2]
            return [(int(lm.x*w), int(lm.y*h)) for lm in result.face_landmarks[0]]
    except Exception as e:
        print(f"MediaPipe error: {e}")
    return None
def region_mask_from_landmarks(landmarks, indices, h, w):
    pts = [landmarks[i] for i in indices if i < len(landmarks)]
    if len(pts) < 3: return np.zeros((h,w), dtype=np.uint8)
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=2)
def create_all_masks(img_bgr, target=224):
    h, w = img_bgr.shape[:2]
    lms = get_landmarks_mp(cv2.resize(img_bgr, (w, h)))
    if lms is None:
        return _haar_masks(img_bgr, target)
    masks = {}
    for name, indices in REGIONS.items():
        mask = region_mask_from_landmarks(lms, indices, h, w)
        masks[name] = cv2.resize(mask, (target,target), interpolation=cv2.INTER_NEAREST)
    face = region_mask_from_landmarks(lms, FACE_OVAL, h, w)
    all_feat = np.zeros((h,w), dtype=np.uint8)
    for m in masks.values():
        all_feat = cv2.bitwise_or(all_feat, cv2.resize(m, (w,h), interpolation=cv2.INTER_NEAREST))
    skin = cv2.bitwise_and(face, cv2.bitwise_not(all_feat))
    masks["Skin"] = cv2.resize(cv2.erode(skin, np.ones((3,3),np.uint8), iterations=1),
                               (target,target), interpolation=cv2.INTER_NEAREST)
    ey_y = min(lms[i][1] for name in ["L Eyebrow","R Eyebrow"] for i in REGIONS[name] if i < len(lms))
    hair = np.zeros((h,w), dtype=np.uint8)
    hair[0:ey_y, :] = 255
    hair = cv2.bitwise_and(hair, cv2.dilate(face, np.ones((5,5),np.uint8), iterations=2))
    masks["Hair"] = cv2.resize(hair, (target,target), interpolation=cv2.INTER_NEAREST)
    return masks
def _haar_masks(img_bgr, target=224):
    h,w=img_bgr.shape[:2]
    fx,fy,fw,fh=(w//4, h//6, w//2, h*2//3)
    try:
        c=cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml"))
        faces=c.detectMultiScale(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(60,60))
        if len(faces)>0: fx,fy,fw,fh=faces[0]
    except: pass
    fx,fy=max(0,fx),max(0,fy); fw,fh=min(fw,w-fx),min(fh,h-fy)
    defs={"Left Eye":(0.20,0.44,0.05,0.44),"Right Eye":(0.20,0.44,0.56,0.95),
          "Nose":(0.38,0.58,0.28,0.72),"Mouth":(0.55,0.80,0.18,0.82),
          "L Eyebrow":(0.14,0.24,0.05,0.44),"R Eyebrow":(0.14,0.24,0.56,0.95),
          "Skin":(0.22,0.65,0.05,0.30),"Hair":(0.00,0.16,0.05,0.95)}
    masks={}
    for n,(y1,y2,x1,x2) in defs.items():
        m=np.zeros((h,w),np.uint8)
        yi1,yi2=int(fy+fh*y1),int(fy+fh*y2)
        xi1,xi2=int(fx+fw*x1),int(fx+fw*x2)
        m[max(0,yi1):min(h,yi2),max(0,xi1):min(w,xi2)]=255
        if n=="Skin":
            xi1r,xi2r=int(fx+fw*(1-x2)),int(fx+fw*(1-x1))
            m[max(0,yi1):min(h,yi2),max(0,xi1r):min(w,xi2r)]=255
        masks[n]=cv2.resize(m,(target,target),interpolation=cv2.INTER_NEAREST)
    return masks
def blur_occlude(img_t, mask, k=31):
    if img_t.ndim != 4 or img_t.shape[0] != 1:
        raise ValueError("img_t must have shape (1, channels, height, width)")
    if not isinstance(k, int) or k <= 0 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer")
    _,C,H,W=img_t.shape
    if mask.shape != (H, W):
        raise ValueError(f"mask must have shape {(H, W)}")
    im=img_t.squeeze(0).cpu().numpy()
    bl=np.array([cv2.GaussianBlur(im[c],(k,k),k//4) for c in range(C)])
    m=mask.astype(np.float32)/255.0; m3=np.stack([m]*C,axis=0)
    return torch.from_numpy(im*(1-m3)+bl*m3).unsqueeze(0).to(img_t.device)
def analyze_face(path, mat_path=None, cb=None):
    if cb: cb("Model...")
    model=Net().to(device)
    p=os.path.join(ROOT,"code","pretrain_model","net_cross_1.weight")
    if os.path.exists(p): model.load_state_dict(torch.load(p,map_location=device),strict=False)
    model.eval()
    if cb: cb("Image...")
    img=Image.open(path).convert("RGB")
    orig=np.array(img)
    bgr=cv2.cvtColor(orig,cv2.COLOR_RGB2BGR)
    T=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
         transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    img_t=T(img).unsqueeze(0).to(device)
    if mat_path and os.path.exists(mat_path):
        mr=np.load(mat_path)
        if mr.ndim != 3 or mr.shape[0] < 18:
            raise ValueError("Parsing map must have shape (at least 18, height, width)")
        mp_=mat_process(mr).transpose(1,2,0)
        mat_t=torch.from_numpy(cv2.resize(mp_,(224,224),interpolation=cv2.INTER_CUBIC)
                               .transpose(2,0,1)).float().unsqueeze(0).to(device)
    else:
        mat_t=torch.zeros(1,7,224,224).to(device)
    i2=make_aux_input()
    if cb: cb("Baseline...")
    with torch.no_grad(): base=model(img_t,mat_t,i2).item()
    if cb: cb("Face landmarks...")
    masks=create_all_masks(cv2.resize(bgr,(224,224)), 224)
    if cb: cb("Occlusion...")
    deltas={}
    for name,mask in masks.items():
        if cb: cb(f"  {DISPLAY.get(name,name)}...")
        occ=blur_occlude(img_t,mask)
        with torch.no_grad(): s=model(occ,mat_t,i2).item()
        deltas[name]=base-s
    if cb: cb("Heatmap...")
    hm=make_heatmap(cv2.resize(bgr,(224,224)),masks,deltas)
    def to10(raw): return (max(1.0,min(5.0,raw))-1.0)/4.0*10.0
    s10=to10(base)
    sorted_items=sorted(deltas.items(),key=lambda x:x[1],reverse=True)
    mx=max(abs(v) for v in deltas.values()) if deltas else 1e-8
    if mx == 0:
        mx = 1e-8
    lines=[f"SCORE: {s10:.1f}/10"]
    for name,delta in sorted_items:
        norm=5.0+(delta/mx)*5.0; norm=max(0,min(10,norm))
        dn=DISPLAY.get(name,name)
        tag="DEP" if delta>0.003 else ("XAU" if delta<-0.003 else "---")
        bar="".join(["\u2581","\u2582","\u2583","\u2584","\u2585","\u2586","\u2587","\u2588","\u2588"])[:max(1,int(norm/10*9))]
        lines.append(f"  {dn:14s} {norm:4.1f} [{tag}] {bar}")
    polygons = get_region_polygons(cv2.resize(bgr, (224, 224)))
    return {"score":base,"score_10":s10,"heatmap":hm,"deltas":deltas,"summary":"\n".join(lines), "region_polygons": polygons}
def get_region_polygons(img_bgr):
    h, w = img_bgr.shape[:2]
    lms = get_landmarks_mp(img_bgr)
    if lms is None:
        return _fallback_polygons()
    polygons = {}
    for name, indices in REGIONS.items():
        pts = []
        for i in indices:
            if i < len(lms):
                x, y = lms[i]
                pts.append({"x": round(x / w, 5), "y": round(y / h, 5)})
        if len(pts) >= 3:
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            polygons[name] = pts
    face_pts = []
    for i in FACE_OVAL:
        if i < len(lms):
            x, y = lms[i]
            face_pts.append({"x": round(x / w, 5), "y": round(y / h, 5)})
    if len(face_pts) >= 3:
        if face_pts[0] != face_pts[-1]:
            face_pts.append(face_pts[0])
        polygons["Skin"] = face_pts
    ey_y = 0
    for name in ["L Eyebrow", "R Eyebrow"]:
        if name in REGIONS:
            for i in REGIONS[name]:
                if i < len(lms):
                    ey_y = max(ey_y, lms[i][1])
    if ey_y > 0:
        face_top = min(lms[i][1] for i in FACE_OVAL if i < len(lms)) if lms else 0
        fx0, fx1 = 0.0, 1.0
        if lms:
            xs = [lms[i][0] for i in FACE_OVAL if i < len(lms)]
            if xs:
                fx0 = max(0.0, min(xs) / w - 0.05)
                fx1 = min(1.0, max(xs) / w + 0.05)
        polygons["Hair"] = [
            {"x": round(fx0, 5), "y": round(face_top / h, 5)},
            {"x": round(fx1, 5), "y": round(face_top / h, 5)},
            {"x": round(fx1, 5), "y": round(ey_y / h, 5)},
            {"x": round(fx0, 5), "y": round(ey_y / h, 5)},
            {"x": round(fx0, 5), "y": round(face_top / h, 5)},
        ]
    return polygons
def _fallback_polygons():
    poly = {}
    defaults = {
        "Left Eye":   [(0.25,0.32),(0.42,0.32),(0.42,0.42),(0.25,0.42),(0.25,0.32)],
        "Right Eye":  [(0.58,0.32),(0.75,0.32),(0.75,0.42),(0.58,0.42),(0.58,0.32)],
        "Nose":       [(0.38,0.42),(0.62,0.42),(0.62,0.58),(0.38,0.58),(0.38,0.42)],
        "Mouth":      [(0.32,0.62),(0.68,0.62),(0.68,0.78),(0.32,0.78),(0.32,0.62)],
        "L Eyebrow":  [(0.22,0.22),(0.44,0.22),(0.44,0.28),(0.22,0.28),(0.22,0.22)],
        "R Eyebrow":  [(0.56,0.22),(0.78,0.22),(0.78,0.28),(0.56,0.28),(0.56,0.22)],
        "Skin":       [(0.15,0.22),(0.85,0.22),(0.85,0.82),(0.15,0.82),(0.15,0.22)],
        "Hair":       [(0.05,0.00),(0.95,0.00),(0.95,0.22),(0.05,0.22),(0.05,0.00)],
    }
    for name, pts in defaults.items():
        poly[name] = [{"x": x, "y": y} for x, y in pts]
    return poly
def make_heatmap(img224,masks,deltas):
    hm=np.zeros((224,224,3),np.float32); mx=max(abs(v) for v in deltas.values()) if deltas else 1
    if mx == 0:
        mx = 1
    for n,m in masks.items():
        d=deltas.get(n,0); v=max(-1,min(1,d/mx))
        c=(0,v,0) if v>0 else (0,0,-v)
        m3=np.stack([m.astype(np.float32)/255]*3,axis=-1)
        hm=hm*(1-m3)+np.array(c)*m3
    h8=(hm*255).astype(np.uint8); h8=cv2.GaussianBlur(h8,(25,25),0)
    ov=cv2.addWeighted(img224,0.6,h8,0.4,0)
    leg=np.zeros((20,224,3),np.uint8)
    for x in range(224):
        t=x/224; leg[:,x]=(0,int(255*t*2),255) if t<.5 else (0,255,int(255*(2-t*2)))
    comb=np.vstack([ov,leg])
    cv2.putText(comb,"BAD",(5,ov.shape[0]+15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(comb,"GOOD",(170,ov.shape[0]+15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return Image.fromarray(cv2.cvtColor(comb,cv2.COLOR_BGR2RGB))
