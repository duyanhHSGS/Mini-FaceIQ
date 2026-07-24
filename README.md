# mini-faceIQ

`mini-faceIQ` is the small, standalone version of the FaceIQ scoring stack. It keeps the parts needed for local front-profile landmark placement, front-profile geometry scoring, and learned attractiveness/feature heatmap scoring, while removing the larger Next.js onboarding/report app.

Think of it like Brain HQ after Uncle Manager cleaned the room: the big app's side-profile and dashboard machinery is mostly gone, but the front-profile calculator and AI feature scorer are still alive and useful.

## Current Scope

### Included

- Local Flask web app served from `main.py`.
- Plain HTML/CSS/JavaScript UI in `web/index.html`.
- Front-facing image upload.
- Manual front landmark placement on a canvas.
- MediaPipe-based front landmark auto-detection helper.
- Original 30 front-profile geometry metrics.
- Gender and ethnicity adjusted front-profile ideal ranges.
- Learned AI attractiveness score from the SCUT model code.
- Region occlusion feature analysis for eyes, brows, nose, mouth, skin, and hair.
- Heatmap output for the learned feature scorer.

### Not Included

- The full `LooksmaxAI` Next.js app.
- React/TSX dashboard, onboarding, or routing.
- 3DDFA side-profile geometric landmark detection.
- Side-profile geometric measurements.
- Production auth, deployment config, or database storage.

Important side-profile note: the original larger app has side-profile scoring, but this mini app does not. In `mini-faceIQ`, `sideMeasurements` is always empty and `sideScore` is currently `0`.

## Architecture

```text
mini-faceIQ/
  main.py                    Flask server and HTTP API
  web/index.html             Browser UI for upload, landmark placement, results
  front_landmarks.py         Front landmark definitions and normalization
  front_autolandmarks.py     MediaPipe-to-front-landmark mapping
  front_calculator.py        30 front-profile geometry metrics and scoring
  front_ideals.py            Gender/ethnicity-adjusted ideal ranges
  feature_scorer.py          CLI/API wrapper around learned model analysis
  face_analyzer.py           Model loading, MediaPipe masks, occlusion, heatmap
  face_landmarker.task       Local MediaPipe FaceLandmarker model
  code/scut/                 SCUT model architecture code
  code/pretrain_model/       Learned model weights
  outputs/
    uploads/                 Saved uploaded images
    heatmaps/                Saved heatmap PNGs
```

## Data Flow

### Front Geometry Flow

1. User uploads a front-facing image in the browser.
2. UI asks `/api/front-landmarks` for the required landmark list.
3. User places landmarks manually, or clicks auto-detect.
4. Auto-detect posts the image to `/api/front-autolandmarks`.
5. `front_autolandmarks.py` uses MediaPipe landmarks through `face_analyzer.get_landmarks_mp`.
6. Browser sends normalized landmark coordinates to `/api/front-metrics`.
7. `front_calculator.py` scales coordinates, computes 30 front metrics, scores each metric, applies group weights and penalties, then returns JSON.

### AI Feature Scoring Flow

1. User uploads an image to `/api/analyze`, or runs `feature_scorer.py` from CLI.
2. `face_analyzer.py` loads the SCUT model from `code/scut/` and `code/pretrain_model/net_cross_1.weight`.
3. The image is resized/cropped to the model input shape.
4. MediaPipe creates region masks for eyes, eyebrows, nose, mouth, skin, and hair.
5. Each region is blurred/occluded and scored again.
6. Score deltas estimate which regions helped or hurt the model output.
7. A heatmap PNG can be saved under `outputs/heatmaps/`.

## HTTP API

### `GET /`

Serves the web UI.

### `GET /api/front-landmarks`

Returns the front landmark definitions from `FRONT_LANDMARK_DEFS`.

Response shape:

```json
{
  "success": true,
  "landmarks": [
    { "id": "hairline", "label": "Hairline", "group": "head", "color": "#3b82f6" }
  ]
}
```

### `POST /api/front-autolandmarks`

Accepts a front-facing image upload and returns normalized front landmarks.

Form field:

- `image`: `jpg`, `jpeg`, `png`, or `webp`

Response shape:

```json
{
  "success": true,
  "landmarks": [
    { "id": "left_pupil", "x": 0.38, "y": 0.36, "label": "Left Pupil" }
  ]
}
```

### `POST /api/front-metrics`

Calculates front-profile geometry metrics. All required front landmarks must be present.

Request shape:

```json
{
  "gender": "male",
  "ethnicity": "asian",
  "frontAspect": 1.0,
  "landmarks": [
    { "id": "hairline", "x": 0.5, "y": 0.08, "label": "Hairline" },
    { "id": "left_pupil", "x": 0.38, "y": 0.36, "label": "Left Pupil" }
  ]
}
```

Response data includes:

- `frontMeasurements`: metric table with values, units, scores, ideal ranges, deviations, and interpretations.
- `frontScore`: weighted front-profile geometry score.
- `overallScore`: currently `frontScore * 0.60` because side scoring is not included.
- `sideMeasurements`: always empty in this mini app.
- `sideScore`: always `0` in this mini app.
- `groups`: `G_F1`, `G_F2`, `G_F3`, `G_S1`, `G_S2`, `G_S3`, `P_front`, `P_side`.

### `POST /api/analyze`

Runs learned model attractiveness scoring and region occlusion analysis.

Form field:

- `image`: `jpg`, `jpeg`, `png`, or `webp`

Response data includes:

- `overall_score`: model score scaled to 0-100.
- `score_10`: model score scaled to 1-10.
- `features`: per-region contribution estimates.
- `summary`: text summary from the occlusion run.
- `region_polygons`: normalized polygons used by the UI.
- `image_url`: saved upload path under `/outputs/uploads/`.
- `heatmap_url`: saved heatmap path under `/outputs/heatmaps/`.

## Front Geometry Specs

### Landmark Model

`front_landmarks.py` defines the required front landmarks. They cover:

- Head: hairline, temples.
- Eyes: pupils, canthi, eyelids, eyelid hood/crease.
- Brows: heads, inner corners, arches, peaks, tails.
- Nose: side points, bridge points, base, bottom.
- Mouth: corners, Cupid's bow, mouth middle, lower lip center.
- Jaw/chin: upper jaw angles, lower jaw angles, chin points, chin bottom.
- Cheeks: left and right cheekbones.

Coordinates are normalized browser/image coordinates:

- `x`: `0.0` left to `1.0` right.
- `y`: `0.0` top to `1.0` bottom.
- `frontAspect`: image width divided by image height.

The calculator scales normalized points by `SCALE = 1000`, applies `frontAspect` to x, and computes all distances/angles from that scaled coordinate space.

### Metrics

The current front calculator can produce 30 metrics:

- Lateral Canthal Tilt
- Nose Bridge to Nose Width Ratio
- Bitemporal Width
- Cheekbone Height
- Cupid's Bow Depth
- Bigonial Width
- Jaw Slope
- Middle Third
- Eye Aspect Ratio
- Mouth Corner Position
- Eye Separation Ratio
- Eyebrow Tilt
- Lower Third
- Face Width to Height Ratio
- Interpupillary-Mouth Width Ratio
- Jaw Frontal Angle
- Intercanthal-Nasal Width Ratio
- Top Third
- One Eye Apart Test
- Midface Ratio
- Ipsilateral Alar Angle
- Mouth Width to Nose Width Ratio
- Total Facial Width to Height Ratio
- Chin to Philtrum Ratio
- Eyebrow Low Setedness
- Brow Length to Face Width Ratio
- Nose Tip Position
- Deviation of IAA & JFA
- Lower Lip to Upper Lip Ratio
- Lower Third Proportion

### Scoring

Each metric is scored with a plateau Gaussian:

- Values inside the ideal range receive `10.0`.
- Values outside the ideal range decay smoothly toward a floor score.
- Very low metric scores contribute to a capped front penalty.

Front group weighting:

- `G_F1`: facial thirds, width/height, midface, jaw/face proportions.
- `G_F2`: eyes, brows, cheekbone-related measurements.
- `G_F3`: jaw, mouth, nose, lower-face measurements.

Final front score:

```text
frontScore = G_F1 * 0.40 + G_F2 * 0.30 + G_F3 * 0.30 - P_front
```

Mini overall score:

```text
overallScore = frontScore * 0.60
```

That overall behavior mirrors the larger app's front/side weighting, but because side scoring is absent here, it intentionally leaves the side contribution at zero.

## AI Feature Scoring Specs

`feature_scorer.py` wraps `face_analyzer.analyze_face`.

The learned scorer:

- Uses PyTorch and the SCUT model code under `code/scut/`.
- Loads `code/pretrain_model/net_cross_1.weight` when present.
- Uses CUDA if available, otherwise CPU.
- Uses MediaPipe for face and region landmarks.
- Falls back to coarse Haar-style masks if MediaPipe landmarks are unavailable.
- Produces a score plus region occlusion deltas.

Feature regions:

- Left Eye
- Right Eye
- Left Eyebrow
- Right Eyebrow
- Nose
- Mouth
- Skin
- Hair

## Usage

### Web UI

Run the app from this folder using your existing environment:

```powershell
python main.py
```

Then open:

```text
http://127.0.0.1:7860
```

Use the UI to upload a front-facing photo, place or auto-detect landmarks, choose gender/ethnicity, and calculate metrics.

### CLI Feature Scorer

```powershell
python feature_scorer.py path\to\face.jpg
```

Save a heatmap:

```powershell
python feature_scorer.py path\to\face.jpg --heatmap outputs\heatmap.png
```

## Dependencies

`requirements.txt` lists the expected Python packages:

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `pillow`
- `mediapipe`
- `scipy`
- `flask`

Install dependencies only when you choose to:

```powershell
pip install -r requirements.txt
```

## Limitations

- Front-profile geometry requires a clear, mostly front-facing photo.
- Auto-landmarks are a helper, not a guaranteed final answer; manual correction is still expected.
- The geometry scorer depends on landmark placement quality.
- The learned model score is separate from the geometry score.
- Side-profile scoring is not implemented in this mini project.
- 3DDFA assets and side-profile mesh mapping remain in the larger project, not here.

