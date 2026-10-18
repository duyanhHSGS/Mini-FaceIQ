# Mini-FaceIQ — Android, Linux & experimental native iOS UI

This Flet app runs face geometry and AI analysis **locally**. It does not use Flask or an external inference API. The Android model ports reuse Mini-FaceIQ's original MediaPipe `.task`, 3DDFA ONNX checkpoints, original front/side geometry calculators, and SCUT checkpoint exported faithfully to ONNX.

## Features

| Feature | Lubuntu | Android implementation |
|---|---|---|
| Front auto-landmarking | Original Python MediaPipe | Native Android MediaPipe Tasks via bundled Flet extension |
| Side auto-landmarking | Original ONNX 3DDFA + Python helpers | Original ONNX models + CPU-only NumPy/OpenCV helpers |
| Manual photo landmark placement | Flet Align tab | Flet Align tab; tap, drag and undo |
| Original geometric scores | Root Python calculator modules | Exact copies packaged with the APK |
| Feature-rating SCUT | PyTorch checkpoint | Original checkpoint converted to ONNX; deterministic aux input preserved |
| Region occlusion/heatmap | PyTorch + OpenCV | ONNX + OpenCV + Pillow, eight occlusion passes |

The APK targets ARM64 (`arm64-v8a`) only. This keeps builds faster and avoids packaging an unused x86-64 emulator ABI. The models are packaged locally, not downloaded at runtime.

## Native iOS source port (not yet a tested IPA)

The iOS Flet app uses the existing offline Python geometry calculators and manual point editor. An experimental Swift MediaPipeTasksVision bridge provides front auto-detection and local HEIC conversion. Automatic 3DDFA side inference and SCUT feature rating are **not yet available on iOS**. Native signing and compilation require macOS. See [IOS_PORT.md](IOS_PORT.md) for build, device-installation and testing instructions.

## On Lubuntu

GUI: Open `src/main.py` in your editor and select the existing `.venv-linux/bin/python` environment.

Terminal fallback from the Mini-FaceIQ repository root:

```bash
.venv-linux/bin/python flet_holy_ui/src/main.py
```

The desktop app continues to use the project's original installed Python ML dependencies, without requiring a Flask server.

## Android APK

GUI: In PCManFM, open `flet_holy_ui/build/apk-linux-parity`. Once an APK has successfully been built, copy `minifaceiq-holy-ui.apk` to your Android device. Open it with the phone's Files app and approve installation if prompted.

Terminal fallback from the repository root:

```bash
cd flet_holy_ui
../.venv-linux/bin/flet build apk --arch arm64-v8a
```

The configuration in `pyproject.toml` includes the local native extension, Android ONNX Runtime, OpenCV, NumPy and Pillow. The neural models still make the APK substantial, but only the ARM64 native libraries are built and bundled.

## Verification

```bash
.venv-linux/bin/ruff check flet_holy_ui/src flet_holy_ui/tests
.venv-linux/bin/pyright flet_holy_ui/src/main.py --pythonpath .venv-linux/bin/python
.venv-linux/bin/python -m pytest flet_holy_ui/tests -q
```

Linux parity tests have established pixel-exact SCUT preprocessing and eight occlusion masks on a face-containing sample, matching front/side named landmark coordinates, and nearly identical ONNX and PyTorch SCUT scores. **An Android device/emulator run is still mandatory before claiming Android runtime parity.** APK packaging success alone does not prove a button works or models load on a device.

If an Android inference path fails, the error is shown in the UI; the manual landmark editor remains usable. Scores are project-specific estimates, not objective assessments of a person.
