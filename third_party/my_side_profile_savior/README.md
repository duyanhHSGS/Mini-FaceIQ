# Side-Profile Landmark Model Factory

This directory is an experimental training and QA factory. It does not replace
Mini-FaceIQ's production `third_party.adapter_side` provider and is not imported
by `main.py`.

## Current contract

- The network emits 31 heatmaps in the worksheet's Mini-FaceIQ landmark order.
- The current 15 confirmed slots are trained; the other 16 are fully masked.
- `user-custom.txt` is human-owned and authoritative.
- Integer worksheet values are trained and evaluated.
- `NONE`, blank, `?`, and tentative values ending in `?` are masked.
- Every checkpoint embeds an immutable mapping snapshot and subject split.
- Only PyTorch `.pt` checkpoints are produced. There is no ONNX/app export.

The current dataset line contains an image path, four bounding-box values, five
auxiliary points, and 39 profile landmarks. Parsing retains that raw file
contract, then selects only the 15 human-confirmed Multi-PIE points into the 31
model slots. The other 24 dataset points and five auxiliary points never enter
the model targets.

## Launch the factory

Windows:

```powershell
.venv\Scripts\python.exe -m third_party.my_side_profile_savior.ui
```

Linux desktop:

```bash
python -m third_party.my_side_profile_savior.ui
```

Linux may require its system `python3-tk` package. The UI requires a graphical
desktop. Headless machines can use the CLI:

```bash
python -m third_party.my_side_profile_savior.train --device auto
```

List every CLI option:

```bash
python -m third_party.my_side_profile_savior.train --help
```

Do not run these commands from inside `third_party/my_side_profile_savior`;
run them from the Mini-FaceIQ repository root so package imports remain stable.

## GPU and CPU behavior

`--device auto` selects CUDA when PyTorch reports it available and otherwise
uses CPU. `--device cuda` fails loudly without CUDA. `--device cpu` always uses
CPU. AMP is enabled only on CUDA.

A CUDA-trained checkpoint is portable to CPU because factory inference loads it
with `map_location`. FaceBoxes remains an ONNX crop detector and may run on CPU
while the landmark network runs on CUDA.

The root `requirements.txt` remains the dependency list. A GPU workstation must
install a PyTorch build compatible with its OS, driver, and CUDA runtime; the
plain pinned requirement cannot guarantee the correct CUDA wheel for every
machine. The first pretrained run may download MobileNetV3-Large weights.

## Run artifacts

Each run gets a timestamped directory under
`git-plz-ignore/profile_factory_runs`:

```text
YYYYMMDD-HHMMSS/
|-- config.json
|-- mapping.json
|-- split.json
|-- status.json
|-- training.jsonl
|-- curves.json
|-- best.pt
|-- last.pt
|-- benchmark.json
`-- legacy_baseline_cache.json
```

`best.pt` minimizes validation NME. `last.pt` is resumable. Checkpoints are
written atomically. The Tkinter Stop button creates a `STOP` request; training
finishes the active batch and saves before exiting.

Resume requires the original mapping and architecture/loss settings. When the
human confirms additional dataset mappings later, use "Initialize weights from
checkpoint" instead: it starts a new immutable run, keeps compatible 31-slot
network weights, and creates a fresh optimizer and mapping snapshot. Checkpoints
without the complete 31-slot worksheet layout are rejected.

## Benchmark

The frozen test identities never overlap training or validation. Both custom
and legacy models receive annotation-derived crops. Missing or invalid points
receive NME `1.0`. Reports include NME and PCK at 2%, 5%, and 10%.

The experiment graduates only if custom mean test NME is lower than legacy
3DDFA-V2 for every confirmed landmark individually.

## QA rules

Dataset QA shows truth, custom predictions, legacy predictions, confidence, and
normalized error. Upload QA uses FaceBoxes and supports a manual mirror toggle.
Because uploads have no ground truth, the UI never claims upload accuracy.

Landmarks use one stable color in every QA panel; dataset index `0` (`porion`)
is blue. Labels stay hidden until hover, and hovering a landmark reveals the
matching label in every comparison panel. A panel reports `not predicted`
instead of inventing a point when that provider has no result. Point and label
opacity have separate live controls. A landmark color may be overridden with
the color picker and reset to its automatic color; overrides last only for the
current UI session.

Scroll over an image to zoom every comparison panel around the same location.
Right-drag pans the panels together, and double-click resets the shared view.
Changing color or opacity updates the existing plot and never reruns model
inference.

QA may run while training. It loads the latest atomically saved `best.pt` into a
separate model instance. If simultaneous CUDA use causes out-of-memory, QA
reports the failure and clears its cache without stopping the trainer.
