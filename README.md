# mini-faceIQ

`mini-faceIQ` is the small, standalone version of the FaceIQ geometry stack. It keeps the parts needed for local front/side landmark placement and geometry scoring, while removing the larger Next.js onboarding/report app and the learned AI attractiveness scorer.

Think of it like Brain HQ after Uncle Manager cleaned the room: the big dashboard machinery is gone, the SCUT model is gone, heatmaps are gone, and the front/side geometry calculators are the main power source.

## Current Scope

### Included

- Local Flask web app served from `main.py`.
- Plain HTML/CSS/JavaScript UI in `web/index.html`.
- Front-facing or side-profile image upload.
- Manual front and side landmark placement on a canvas.
- MediaPipe-based front landmark auto-detection helper.
- Original 30 front-profile geometry metrics.
- Manual side-profile geometry metrics.
- Gender and ethnicity adjusted front/side ideal ranges.

### Not Included

- The full `LooksmaxAI` Next.js app.
- React/TSX dashboard, onboarding, or routing.
- 3DDFA side-profile geometric landmark detection.
- 3DDFA automatic side-profile landmark placement.
- PyTorch or TorchVision.
- SCUT model architecture code.
- Learned model weights.
- AI attractiveness scoring endpoint.
- Region occlusion feature analysis.
- Heatmap generation.
- Production auth, deployment config, or database storage.

Important side-profile note: side-profile scoring is manual-landmark based in this mini app. MediaPipe auto-landmarking is currently wired for front-facing images only.

## Architecture

```text
mini-faceIQ/
  main.py                    Flask server and HTTP API
  web/index.html             Browser UI for upload, landmark placement, results
  front_landmarks.py         Front landmark definitions and normalization
  front_autolandmarks.py     MediaPipe-to-front-landmark mapping
  front_calculator.py        30 front-profile geometry metrics and scoring
  front_ideals.py            Gender/ethnicity-adjusted ideal ranges
  side_landmarks.py          Side landmark definitions and normalization
  side_calculator.py         Side-profile geometry metrics and scoring
  side_ideals.py             Gender/ethnicity-adjusted side ideal ranges
  face_analyzer.py           Minimal MediaPipe FaceLandmarker helper
  face_landmarker.task       Local MediaPipe FaceLandmarker model
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

### Side Geometry Flow

1. User chooses side profile mode and uploads a side-profile image.
2. UI asks `/api/side-landmarks` for the required landmark list.
3. User places side landmarks manually.
4. Browser sends normalized landmark coordinates to `/api/side-metrics`.
5. `side_calculator.py` scales coordinates, computes side metrics, scores each metric, applies group weights and penalties, then returns JSON.

## HTTP API

### `GET /`

Serves the web UI.

### `GET /api/front-landmarks`

Returns the front landmark definitions from `FRONT_LANDMARK_DEFS`.

### `GET /api/side-landmarks`

Returns the side landmark definitions from `SIDE_LANDMARK_DEFS`.

### `POST /api/front-autolandmarks`

Accepts a front-facing image upload and returns normalized front landmarks.

Form field:

- `image`: `jpg`, `jpeg`, `png`, or `webp`

### `POST /api/front-metrics`

Calculates front-profile geometry metrics. All required front landmarks must be present.

### `POST /api/side-metrics`

Calculates side-profile geometry metrics. All required side landmarks must be present.

## Front Geometry Specs

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

The current front calculator can produce 30 metrics.

## Side Geometry Specs

`side_landmarks.py` defines the required side landmarks. They cover:

- Head/profile: top of head, occiput, hairline, forehead, glabella.
- Nose: nasal bridge root, rhinion, supratip, nose tip, infratip, columella, subnasale, subalare.
- Mouth/chin: lips, mouth corner, labiomental fold, chin point, chin bottom.
- Ears/jaw/neck: porion, tragus, intertragic notch, jaw angles, cervical point, neck point.
- Eyes/cheeks: orbitale, corneal apex, eyelid end, lower eyelid, cheekbone.

Coordinates use the same normalized browser/image coordinate model as front landmarks:

- `x`: `0.0` left to `1.0` right.
- `y`: `0.0` top to `1.0` bottom.
- `sideAspect`: image width divided by image height.

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

Use the UI to choose front or side mode, upload a photo, place landmarks, choose gender/ethnicity, and calculate metrics. Front mode can use auto-detect as a helper; side mode is manual.

## Dependencies

`requirements.txt` lists the expected Python packages:

- `numpy`
- `opencv-python`
- `mediapipe`
- `flask`

Install dependencies only when you choose to:

```powershell
pip install -r requirements.txt
```

## Limitations

- Front-profile geometry requires a clear, mostly front-facing photo.
- Side-profile geometry requires a clear side photo.
- Auto-landmarks are a helper, not a guaranteed final answer; manual correction is still expected.
- The geometry scorer depends on landmark placement quality.
- MediaPipe front auto-landmarks do not replace manual side-profile landmark placement.
