# mini-faceIQ

Standalone FaceIQ utility extracted from the larger Facial-Scoring-System repo.

## What It Does

- Runs a local HTML app.
- Lets you upload a front-facing image and click front landmarks.
- Calculates the original 30 front-profile geometry metrics.
- Still includes the learned AI feature scorer and heatmap utility.

## Included Pieces

- `face_analyzer.py` - model loading, region masks, occlusion analysis, heatmap creation
- `feature_scorer.py` - small standalone CLI wrapper
- `front_calculator.py` - Python port of the original 30 front-profile metrics
- `front_ideals.py` - Python ideal values with gender/ethnicity adjustments
- `front_landmarks.py` - front landmark definitions and input normalization
- `main.py` - Flask web server
- `web/index.html` - HTML landmark placer and results table
- `code/scut/` - MobileNetV2/co-attention model code
- `code/pretrain_model/net_cross_1.weight` - pretrained attractiveness model weight
- `face_landmarker.task` - MediaPipe face landmarker model

## Usage

### Web UI

Run the local HTML app from this folder:

```powershell
python main.py
```

Then open:

```text
http://127.0.0.1:7860
```

Upload a front-facing face image. The page lets you place the front landmarks on a canvas.

Click each front landmark in order, choose gender/ethnicity, then press "Calculate metrics". The page shows the front score, group scores, and a 30-metric table.

The front metric API expects normalized landmarks:

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

Metrics only appear when their required landmarks are present.

The UI is plain HTML/CSS/JavaScript served by Flask. No Next.js, no TSX, no build step.

### CLI

Use your existing environment if it already has the project dependencies:

```powershell
python feature_scorer.py path\to\face.jpg
```

Save a heatmap too:

```powershell
python feature_scorer.py path\to\face.jpg --heatmap outputs\heatmap.png
```

## Dependencies

`requirements.txt` lists the Python packages this utility expects. Install only when you choose to:

```powershell
pip install -r requirements.txt
```

## Notes

- This mini project does not include the full Next.js app.
- This mini project does not include the 3DDFA geometric landmark scorer.
- Front-profile metrics are manual-landmark based in the HTML tool.
- The learned feature scorer still focuses on attractiveness score and occlusion region analysis.
