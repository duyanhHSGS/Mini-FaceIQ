"""SCUT original ONNX feature rating and occlusion heatmap on Android.

The model was exported with the original checkpoint, zero parsing tensor, and
Linux's deterministic auxiliary vector held as fixed ONNX constants.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def _asset_bytes(name: str) -> bytes:
    if Path(name).name != name:
        raise ValueError('Invalid model asset name')
    inside = 'assets/models/features/' + name
    file = Path(__file__).resolve().parent / inside
    if file.is_file():
        return file.read_bytes()
    path = str(__file__)
    if '.zip/' in path:
        with zipfile.ZipFile(path.split('.zip/', 1)[0] + '.zip') as app_zip:
            return app_zip.read(inside)
    raise FileNotFoundError(inside)


@lru_cache(maxsize=1)
def _model_session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        _asset_bytes('scut_original.onnx'), sess_options=opts,
        providers=['CPUExecutionProvider'],
    )


@lru_cache(maxsize=1)
def _region_config():
    return json.loads(_asset_bytes('regions.json'))


def _rgb_photo(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert('RGB')


def resized_face_photo(data: bytes) -> bytes:
    """MediaPipe on Linux receives a 224x224 OpenCV-resized BGR photo."""
    orig = np.asarray(_rgb_photo(data))
    bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, (224, 224))
    ok, encoded = cv2.imencode('.png', small)
    if not ok:
        raise ValueError('Cannot encode MediaPipe face image')
    return bytes(encoded)


def _preprocess(data: bytes) -> np.ndarray:
    """Match torchvision Resize(256), CenterCrop(224), ToTensor, Normalize."""
    image = _rgb_photo(data)
    w, h = image.size
    if w < h:
        ow, oh = 256, int(256 * h / w)
    else:
        oh, ow = 256, int(256 * w / h)
    resized = image.resize((ow, oh), resample=Image.Resampling.BILINEAR)
    left, top = round((ow - 224) / 2), round((oh - 224) / 2)
    square = resized.crop((left, top, left + 224, top + 224))
    rgb = np.asarray(square, dtype=np.float32)
    return np.ascontiguousarray(((rgb / 255.0 - .5) / .5).transpose(2, 0, 1)[None])


def _mask(pts: list[tuple[int, int]], indices: list[int]) -> np.ndarray:
    vertices = [pts[i] for i in indices if i < len(pts)]
    mask = np.zeros((224, 224), dtype=np.uint8)
    if len(vertices) >= 3:
        cv2.fillPoly(mask, [np.array(vertices, np.int32)], 255)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)


def _fallback_masks() -> dict[str, np.ndarray]:
    # Same facebox-relative regions as face_analyzer._haar_masks fallback,
    # with a centered fallback facebox if native landmarks are unavailable.
    fx, fy, fw, fh = 56, 37, 112, 149
    defs = {'Left Eye': (.20, .44, .05, .44), 'Right Eye': (.20, .44, .56, .95),
            'Nose': (.38, .58, .28, .72), 'Mouth': (.55, .80, .18, .82),
            'L Eyebrow': (.14, .24, .05, .44), 'R Eyebrow': (.14, .24, .56, .95),
            'Skin': (.22, .65, .05, .30), 'Hair': (.00, .16, .05, .95)}
    result = {}
    for name, (y1,y2,x1,x2) in defs.items():
        mask = np.zeros((224,224),dtype=np.uint8)
        ya,yb = int(fy+fh*y1),int(fy+fh*y2)
        xa,xb = int(fx+fw*x1),int(fx+fw*x2)
        mask[ya:yb,xa:xb] = 255
        if name == 'Skin':
            r1,r2=int(fx+fw*(1-x2)),int(fx+fw*(1-x1))
            mask[ya:yb,r1:r2] = 255
        result[name] = mask
    return result


def _masks_and_polygons(normalized: list[list[float]]):
    config = _region_config()
    points = [(int(p[0] * 224), int(p[1] * 224)) for p in normalized]
    if not points:
        return _fallback_masks(), {}
    groups = config['REGIONS']
    masks = {name: _mask(points, indices) for name, indices in groups.items()}
    oval = _mask(points, config['FACE_OVAL'])
    aggregate = np.zeros((224, 224), np.uint8)
    for submask in masks.values():
        aggregate = cv2.bitwise_or(aggregate, submask)
    skin = cv2.bitwise_and(oval, cv2.bitwise_not(aggregate))
    masks['Skin'] = cv2.erode(skin, np.ones((3,3),np.uint8), iterations=1)
    eyebrows = [points[i][1] for name in ('L Eyebrow', 'R Eyebrow') for i in groups[name] if i<len(points)]
    ey_y = min(eyebrows) if eyebrows else 0
    hair = np.zeros((224,224),np.uint8)
    hair[:ey_y,:] = 255
    masks['Hair'] = cv2.bitwise_and(hair,cv2.dilate(oval,np.ones((5,5),np.uint8),iterations=2))
    polygons = {}
    for name, indices in {**groups, 'Skin': config['FACE_OVAL']}.items():
        polygon = [{'x':round(points[i][0]/224,5),'y':round(points[i][1]/224,5)}
                   for i in indices if i<len(points)]
        if len(polygon)>=3:
            if polygon[0]!=polygon[-1]: polygon.append(polygon[0])
            polygons[name]=polygon
    if eyebrows:
        oval_points=[points[i] for i in config['FACE_OVAL'] if i<len(points)]
        if oval_points:
            xs=[p[0] for p in oval_points]
            fx0=max(0.,min(xs)/224-.05);fx1=min(1.,max(xs)/224+.05)
            ymin=min(p[1] for p in oval_points)/224
            ymax=max(eyebrows)/224
            polygons['Hair']=[{'x':round(x,5),'y':round(y,5)} for x,y in (
                (fx0,ymin),(fx1,ymin),(fx1,ymax),(fx0,ymax),(fx0,ymin))]
    return masks, polygons


def _blur_occlude(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = image[0]
    blurred = np.stack([cv2.GaussianBlur(rgb[i],(31,31),31//4) for i in range(3)])
    m = (mask.astype(np.float32) / 255.0)[None]
    return np.ascontiguousarray((rgb*(1-m)+blurred*m)[None],dtype=np.float32)


def _heatmap(bgr: np.ndarray, masks: dict, deltas: dict) -> bytes:
    hm=np.zeros((224,224,3),np.float32)
    mx=max((abs(v) for v in deltas.values()),default=1.0) or 1.0
    for name,mask in masks.items():
        v=max(-1,min(1,deltas.get(name,0)/mx))
        color=(0,v,0) if v>0 else (0,0,-v)
        m3=np.repeat((mask.astype(np.float32)/255.0)[:,:,None],3,axis=2)
        hm=hm*(1-m3)+np.array(color)*m3
    h8=cv2.GaussianBlur((hm*255).astype(np.uint8),(25,25),0)
    overlay=cv2.addWeighted(bgr,.6,h8,.4,0)
    legend=np.zeros((20,224,3),np.uint8)
    for x in range(224):
        t=x/224
        legend[:,x]=(0,int(255*t*2),255) if t<.5 else (0,255,int(255*(2-t*2)))
    combined=np.vstack([overlay,legend])
    cv2.putText(combined,'BAD',(5,239),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(combined,'GOOD',(170,239),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    buffer=io.BytesIO()
    Image.fromarray(cv2.cvtColor(combined,cv2.COLOR_BGR2RGB)).save(buffer,format='PNG')
    return buffer.getvalue()


def analyze_mobile_features(image_data: bytes, normalized: list[list[float]]) -> dict:
    """Run the same SCUT baseline + 8 occlusion evaluations fully offline."""
    if not image_data or len(image_data)>20*1024*1024:
        raise ValueError('Select a face photo under 20MB')
    image=_preprocess(image_data)
    masks,polygons=_masks_and_polygons(normalized)
    session=_model_session()
    base=float(cast(list[np.ndarray], session.run(None, {'image': image}))[0].ravel()[0])
    deltas={}
    for name,mask in masks.items():
        occluded=_blur_occlude(image,mask)
        score=float(cast(list[np.ndarray], session.run(None, {'image': occluded}))[0].ravel()[0])
        deltas[name]=base-score
    bgr=cv2.cvtColor(np.asarray(_rgb_photo(image_data)),cv2.COLOR_RGB2BGR)
    bgr=cv2.resize(bgr,(224,224))
    encoded=base64.b64encode(_heatmap(bgr,masks,deltas)).decode('ascii')
    mx=max((abs(v) for v in deltas.values()),default=1e-8) or 1e-8
    score10=(max(1.,min(5.,base))-1.)/4.*10.
    lines=[f'SCORE: {score10:.1f}/10']
    display=_region_config()['DISPLAY']
    regions=[]
    for name,delta in sorted(deltas.items(),key=lambda item:item[1],reverse=True):
        norm=max(0.,min(10.,5.+delta/mx*5.))
        tag='GOOD' if delta>.003 else ('BAD' if delta<-.003 else '---')
        bar='▁▂▃▄▅▆▇██'[:max(1,int(norm/10*9))]
        lines.append(f'  {display.get(name,name):14s} {norm:4.1f} [{tag}] {bar}')
        effect='helps prediction' if delta>.003 else ('lowers prediction' if delta<-.003 else 'neutral')
        regions.append({'name':name,'delta':round(delta,4),
                        'score':round(max(0,min(100,50+delta*200)),1),
                        'effect':effect,'polygon':polygons.get(name,[])})
    return {'rawScore':round(base,4),'score10':round(score10,2),
            'summary':'\n'.join(lines),'regions':regions,
            'heatmapPng':'data:image/png;base64,'+encoded}
