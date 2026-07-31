# Human Data Factory

This package creates human-authored 31-point side-profile landmark datasets.
It imports images, records human clicks, and creates portable exports. It does
not train, predict, score, benchmark, or connect to Mini-FaceIQ's production
application.

## Launch

From the Mini-FaceIQ repository root:

```powershell
.venv\Scripts\python.exe -m third_party.human_data_factory
```

The server binds only to `127.0.0.1:8765` and opens the local workspace in the
default browser. Use `--no-browser` or `--port <number>` when needed.

## Workflow

Create a project from either the package's private unlabeled MultiPIE image
copy or a custom browser-selected image folder.

- **All landmarks:** annotate all 31 landmarks on one image, then continue to
  the next image.
- **One focused landmark:** select one landmark and annotate it across the
  entire image queue.
- **Auto Continue on:** a saved click advances immediately.
- **Auto Continue off:** a saved click stays visible until Continue is pressed.

Every edit is immediately committed to the project's SQLite database. Labels
are `unreviewed`, `placed`, or `unavailable`. Undo is auditable, coordinates
remain floating-point source-image pixels, and display mirroring never changes
the stored coordinate system.

## Private data

The following package-local paths are ignored by Git:

```text
source_images/
projects/
```

Custom and MultiPIE imports are EXIF-normalized into project-owned PNG files.
Each export creates a new timestamped directory containing `images/`,
`labels.jsonl`, `landmarks.json`, and `manifest.json`. Partial exports preserve
all untouched labels as `unreviewed`.

## Boundary

Runtime dependencies are Flask, Pillow, and Python's standard library,
including SQLite. This package must not import the application, the experimental
trainer, computer-vision providers, or any model runtime.
