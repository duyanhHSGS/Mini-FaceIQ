# Discrete Side-Profile Trainer

This factory trains one completely independent model per landmark. It starts
with `porion`; the same data and checkpoint contract can later be used for all
31 canonical human-factory landmarks.

## Data flow

```text
human_data_factory/source_images/multipie/*.jpg
        +
human_data_factory/contributions/multipie-profile-v1/porion.jsonl
        -> placed labels only
        -> subject-disjoint train / validation / test split
        -> one MobileNetV3-Small encoder
        -> one heatmap
        -> porion/best.pt
```

`unavailable` rows are valid human decisions but are not coordinate targets.
Missing shard rows are unreviewed and do not enter training. The image filename
is the join key; no old 39-point annotation or worksheet mapping is used.

At the current Porion snapshot, 468 rows are usable `placed` targets and 66 are
human-marked `unavailable`.

## Launch the UI

From the Mini-FaceIQ repository root, using the project's virtual environment:

```powershell
.venv\Scripts\python.exe -m packages.side_trainer_discrete.ui
```

Headless Porion training:

```powershell
.venv\Scripts\python.exe -m packages.side_trainer_discrete.train --landmark porion --device auto
```

Do not launch from inside `packages/side_trainer_discrete`; package imports are
designed for repository-root execution.

## Artifacts

Every landmark owns a separate run tree:

```text
discrete_factory_runs/
`-- porion/
    `-- YYYYMMDD-HHMMSS/
        |-- config.json
        |-- split.json
        |-- status.json
        |-- training.jsonl
        |-- curves.json
        |-- best.pt
        `-- last.pt
```

`best.pt` minimizes validation normalized mean error. `last.pt` is resumable.
Checkpoint writes are atomic, and the UI's graceful stop finishes the current
batch before saving.

Each checkpoint declares `mini-faceiq-discrete-v1`, its one landmark ID, its
immutable subject split, and all architecture/training settings. A checkpoint
cannot accidentally resume another landmark.

## Inference

The Tkinter UI includes a **Brain playground** tab. Choose any local image and
any compatible `best.pt`, then run the specialist and inspect its point on the
original image. Arbitrary resolutions, including Ultra HD, are accepted: the
image is resized to the checkpoint's training size for inference and the
normalized prediction is mapped back to original pixel coordinates. EXIF phone
rotation is applied before prediction and display.

Every successful run automatically saves a full-resolution annotated PNG under
`packages/side_trainer_discrete/output/`. Output filenames contain the source
stem, predicted landmark, and a collision-safe timestamp; source images are
never modified or overwritten.

Resolution compatibility does not guarantee accuracy on unfamiliar crops,
poses, or image domains; the specialist still sees a resized 256x256 input.

```python
from PIL import Image
from packages.side_trainer_discrete.inference import DiscreteLandmarkPredictor

predictor = DiscreteLandmarkPredictor("path/to/porion/best.pt")
result = predictor.predict(Image.open("profile.jpg"))
```

The returned `x` and `y` are in the original image's pixel coordinates.

## Runtime design

The model has one output channel instead of a 31-channel shared head and uses
MobileNetV3-Small. DataLoader workers prefetch batches, CUDA copies can use
pinned memory, AMP is CUDA-only, and CuDNN benchmarking selects fast kernels.
The landmark choice is a direct dictionary/path lookup: O(1). Compile time may
take 1000 years; runtime speed is the only thing that matters. 🦅⚡

## Human transition

Human clicks turn unlabeled zombie pixels into a clean specialist that knows
exactly one anatomical job. Porion first; then the remaining 30 can receive
their own isolated brains. 🪢🔪🩸
