# Mini-FaceIQ Holy Flet UI — Local Inference

This interface uses the Mini-FaceIQ Python detectors/calculators **directly**.
It does not call `/api/front-autolandmarks` or any other HTTP endpoint. It does
not start Flask or run a separate backend server.

## Run on Lubuntu (desktop)

Run in the **Mini-FaceIQ repository checkout**: the UI imports local model
code and model assets from the parent directory. It is standalone as an
application **process**, not yet a relocatable copy of this folder.

1. Create/use a Python 3.11 virtual environment with the **root repository's**
   ML requirements installed (some pinned PyTorch builds do not support 3.12+).

   ```bash
   cd /home/labuntu/Desktop/CodeProjects-Linux/Mini-FaceIQ
   python3.11 -m venv .venv-linux
   source .venv-linux/bin/activate
   python -m pip install -r requirements.txt
   python -m pip install 'flet[all]==1.0.4'
   ```

   If the existing `.venv-linux` already contains the required model dependencies,
   reuse it and install Flet into that environment instead of recreating it.

2. Run the standalone Flet window, **without starting `python main.py`**:

   ```bash
   cd /home/labuntu/Desktop/CodeProjects-Linux/Mini-FaceIQ/flet_holy_ui
   source ../.venv/bin/activate
   flet run
   ```

   On Lubuntu, if the photo picker does not open:

   ```bash
   sudo apt install zenity
   ```

### How it calls Mini-FaceIQ

`src/main.py` calls `src/local_engine.py` directly. That module defines
`front_autolandmarks(image_bytes, filename)` and
`side_autolandmarks(image_bytes, filename)`, wrappers around the existing
`packages.detect_front_landmarks_from_upload` and
`packages.detect_side_landmarks_from_upload` functions. It then invokes
`calculate_front_analysis` or `calculate_side_analysis` in the same Python
process. `features_rating()` also invokes the existing analyzer locally.
The in-memory file adapter replaces Flask's `FileStorage` object.

No URL or API client exists. Images are not sent to Flask or any remote service
by this code. Any model assets must be available locally; separate model
libraries could attempt external downloads if weights are not preinstalled.

### Current limits

- **Linux desktop:** local inference supported when required native Python
  dependencies and model assets are installed on the machine.
- **Android APK:** NOT yet a standalone usable Android build. The root models
  and native packages (PyTorch 2.0.1, MediaPipe, OpenCV, ONNX Runtime) need
  Android-compatible builds/wheels, and the engine/model files need to be
  packaged into the app. Flet alone cannot convert Linux wheels to Android.
  Do not expect `flet build apk` to work correctly until the inference stack
  has been ported.
- Automatic landmarks can be incomplete. If required landmarks are missing,
  the Flet UI explains why no complete metric score is shown. Manual landmark
  dragging is still available only in the original browser UI.
- Scores reflect project-specific heuristics, not an objective measure of
  attractiveness or a person's worth.

### Tests

```bash
cd /home/labuntu/Desktop/CodeProjects-Linux/Mini-FaceIQ
python -m pytest flet_holy_ui/tests
```

No model weights are loaded by the unit tests.
