"""Temporary comparison viewer for Multi-PIE and legacy 3DDFA-V2 landmarks.

This script is intentionally inspection-only: it draws the supplied bounding
box, the five auxiliary points, and dataset landmark indices 0 through 38.
Comparison mode runs the existing legacy 3DDFA-V2 provider on the same image
and draws its named Mini-FaceIQ landmarks beside the dataset annotations.

It does not guess anatomical names for the dataset points or connect them,
because ``user-custom.txt`` is the human-reviewed topology map.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, TextBox
from PIL import Image

try:
    from .dataset import ProfileLandmarkDataset
except ImportError:
    from dataset import ProfileLandmarkDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = (
    REPOSITORY_ROOT
    / "git-plz-ignore"
    / "MultiPIE"
    / "MultiPIE_profile_train.txt"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "git-plz-ignore"
    / "profile_landmarks_preview.png"
)
DEFAULT_MAPPING = Path(__file__).resolve().parent / "user-custom.txt"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _draw_original(
    axis,
    record,
    show_auxiliary: bool,
    *,
    show_labels: bool = True,
    alpha: float = 1.0,
    dot_size: float = 18,
) -> None:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
        axis.imshow(image)

    x1, y1, x2, y2 = record.bbox_xyxy
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="#39ff14",
            linewidth=1.5,
        )
    )

    points = record.landmarks_xy
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=dot_size,
        c="#ff2457",
        edgecolors="white",
        linewidths=0.4,
        alpha=alpha,
        zorder=3,
    )
    if show_labels:
        for landmark_index, (x, y) in enumerate(points):
            axis.annotate(
                str(landmark_index),
                (x, y),
                xytext=(3, -3),
                textcoords="offset points",
                color="#00e5ff",
                fontsize=7,
                weight="bold",
                alpha=alpha,
                zorder=4,
            )

    if show_auxiliary:
        auxiliary = record.auxiliary_points_xy
        axis.scatter(
            auxiliary[:, 0],
            auxiliary[:, 1],
            marker="x",
            s=max(40, dot_size * 1.8),
            c="#ffe600",
            linewidths=1.5,
            alpha=alpha,
            zorder=5,
            label="5 auxiliary points",
        )
        if show_labels:
            axis.legend(loc="lower right", fontsize=7)


def _draw_model_crop(axis, sample, show_auxiliary: bool) -> None:
    image = sample["image"].permute(1, 2, 0).numpy()
    height, width = image.shape[:2]
    axis.imshow(np.clip(image, 0.0, 1.0))

    points = sample["landmarks"].numpy()
    point_pixels = points * np.asarray([width, height], dtype=np.float32)
    axis.scatter(
        point_pixels[:, 0],
        point_pixels[:, 1],
        s=18,
        c="#ff2457",
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    for landmark_index, (x, y) in enumerate(point_pixels):
        axis.annotate(
            str(landmark_index),
            (x, y),
            xytext=(3, -3),
            textcoords="offset points",
            color="#00e5ff",
            fontsize=7,
            weight="bold",
            zorder=4,
        )

    if show_auxiliary:
        auxiliary = sample["auxiliary_points"].numpy()
        auxiliary_pixels = auxiliary * np.asarray(
            [width, height],
            dtype=np.float32,
        )
        axis.scatter(
            auxiliary_pixels[:, 0],
            auxiliary_pixels[:, 1],
            marker="x",
            s=40,
            c="#ffe600",
            linewidths=1.5,
            zorder=5,
            label="5 auxiliary points",
        )
        axis.legend(loc="lower right", fontsize=7)


def _run_legacy_3ddfa(image_path: Path) -> dict:
    """Lazy-load the heavyweight legacy provider only in comparison mode."""

    from third_party.providers.side_3ddfa import detect_side

    return detect_side(str(image_path))


def _draw_legacy_3ddfa(
    axis,
    record,
    result: dict,
    *,
    show_labels: bool = True,
    alpha: float = 1.0,
    dot_size: float = 24,
) -> None:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        axis.imshow(image)

    error = result.get("error")
    if error:
        axis.text(
            0.5,
            0.5,
            f"Legacy 3DDFA-V2 failed:\n{error}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            bbox={"facecolor": "#8b0000", "alpha": 0.85, "pad": 8},
        )
        return

    colors = {
        "2d_sparse": "#ff2bd6",
        "2d_dense": "#39ff14",
        "computed": "#ffe600",
        "unknown": "#ffffff",
    }
    landmarks = result.get("landmarks", {})
    for landmark_id, landmark in landmarks.items():
        x = float(landmark["x"]) * width
        y = float(landmark["y"]) * height
        model_type = landmark.get("model_type", "unknown")
        mesh_index = landmark.get("mesh_index", "?")
        color = colors.get(model_type, colors["unknown"])
        label = landmark.get("label", landmark_id)

        axis.scatter(
            [x],
            [y],
            s=dot_size,
            c=color,
            edgecolors="black",
            linewidths=0.5,
            alpha=alpha,
            zorder=3,
        )
        if show_labels:
            axis.annotate(
                f"{label}\n{model_type}:{mesh_index}",
                (x, y),
                xytext=(4, -4),
                textcoords="offset points",
                color=color,
                fontsize=5.5,
                weight="bold",
                alpha=alpha,
                zorder=4,
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.48,
                    "pad": 1,
                },
            )


def _load_mapping_file(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping worksheet not found: {mapping_path}")

    assignments: dict[str, str] = {}
    for raw_line in mapping_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        parts = raw_line.split("|")
        if len(parts) < 4:
            continue
        assignments[parts[0].strip()] = parts[-1].strip()
    return assignments


def _save_mapping_file(
    mapping_path: Path,
    assignments: dict[str, str],
) -> None:
    original_lines = mapping_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []

    for raw_line in original_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "|" not in raw_line:
            updated_lines.append(raw_line)
            continue

        prefix, separator, _old_value = raw_line.rpartition("|")
        landmark_id = raw_line.split("|", maxsplit=1)[0].strip()
        if not separator or landmark_id not in assignments:
            updated_lines.append(raw_line)
            continue

        value = assignments[landmark_id].strip()
        updated_lines.append(f"{prefix}| {value}")

    mapping_path.write_text(
        "\n".join(updated_lines) + "\n",
        encoding="utf-8",
    )


class InteractiveLandmarkViewer:
    """Clickable human-review cockpit for building ``user-custom.txt``."""

    def __init__(
        self,
        dataset: ProfileLandmarkDataset,
        *,
        start_index: int,
        show_auxiliary: bool,
        mapping_path: Path,
    ) -> None:
        if start_index < 0 or start_index >= len(dataset):
            raise IndexError(
                f"index must be between 0 and {len(dataset) - 1}, "
                f"got {start_index}"
            )

        self.dataset = dataset
        self.index = start_index
        self.show_auxiliary = show_auxiliary
        self.mapping_path = mapping_path.expanduser().resolve()
        self.assignments = _load_mapping_file(self.mapping_path)
        self.legacy_cache: dict[str, dict] = {}
        self.selected_dataset_index: int | None = None
        self.selected_legacy_id: str | None = None
        self.dirty = False
        self.current_legacy_result: dict = {}
        self._updating_index_box = False
        self.dataset_alpha = 0.85
        self.legacy_alpha = 0.85
        self.dot_size = 28.0
        self._pan_state: dict | None = None
        self.dataset_hover = None
        self.legacy_hover = None

        self.figure, (self.dataset_axis, self.legacy_axis) = plt.subplots(
            1,
            2,
            figsize=(16, 9),
        )
        self.figure.subplots_adjust(
            left=0.02,
            right=0.99,
            top=0.90,
            bottom=0.27,
            wspace=0.04,
        )
        try:
            self.figure.canvas.manager.set_window_title(
                "Mini-FaceIQ Landmark Mapping HQ"
            )
        except AttributeError:
            pass

        self.instruction_text = self.figure.text(
            0.02,
            0.235,
            (
                "Hover reveals labels. Scroll zooms both images. Right-drag "
                "pans. Double-click resets. Left-click selects for mapping."
            ),
            fontsize=10,
            weight="bold",
        )
        self.status_text = self.figure.text(
            0.02,
            0.195,
            "",
            fontsize=9,
            family="monospace",
        )

        self.previous_button = Button(
            self.figure.add_axes((0.02, 0.035, 0.07, 0.055)),
            "◀ Prev",
        )
        self.next_button = Button(
            self.figure.add_axes((0.10, 0.035, 0.07, 0.055)),
            "Next ▶",
        )
        self.index_box = TextBox(
            self.figure.add_axes((0.215, 0.035, 0.07, 0.055)),
            "Image # ",
            initial=str(self.index),
        )
        self.assign_button = Button(
            self.figure.add_axes((0.31, 0.035, 0.10, 0.055)),
            "ASSIGN",
            color="#a7f3d0",
            hovercolor="#6ee7b7",
        )
        self.none_button = Button(
            self.figure.add_axes((0.42, 0.035, 0.08, 0.055)),
            "NONE",
            color="#fecaca",
            hovercolor="#fca5a5",
        )
        self.uncertain_button = Button(
            self.figure.add_axes((0.51, 0.035, 0.06, 0.055)),
            "?",
            color="#fef3c7",
            hovercolor="#fde68a",
        )
        self.clear_button = Button(
            self.figure.add_axes((0.58, 0.035, 0.08, 0.055)),
            "Clear",
        )
        self.save_button = Button(
            self.figure.add_axes((0.67, 0.035, 0.10, 0.055)),
            "SAVE TXT",
            color="#bfdbfe",
            hovercolor="#93c5fd",
        )
        self.auxiliary_button = Button(
            self.figure.add_axes((0.78, 0.035, 0.10, 0.055)),
            self._auxiliary_button_label(),
        )
        self.dataset_alpha_slider = Slider(
            self.figure.add_axes((0.08, 0.125, 0.20, 0.025)),
            "Dataset opacity",
            0.0,
            1.0,
            valinit=self.dataset_alpha,
            valstep=0.05,
        )
        self.legacy_alpha_slider = Slider(
            self.figure.add_axes((0.40, 0.125, 0.20, 0.025)),
            "Legacy opacity",
            0.0,
            1.0,
            valinit=self.legacy_alpha,
            valstep=0.05,
        )
        self.dot_size_slider = Slider(
            self.figure.add_axes((0.72, 0.125, 0.18, 0.025)),
            "Dot size",
            8.0,
            120.0,
            valinit=self.dot_size,
            valstep=2.0,
        )

        self.previous_button.on_clicked(self._previous)
        self.next_button.on_clicked(self._next)
        self.index_box.on_submit(self._jump_to_index)
        self.assign_button.on_clicked(self._assign)
        self.none_button.on_clicked(self._mark_none)
        self.uncertain_button.on_clicked(self._mark_uncertain)
        self.clear_button.on_clicked(self._clear_assignment)
        self.save_button.on_clicked(self._save)
        self.auxiliary_button.on_clicked(self._toggle_auxiliary)
        self.dataset_alpha_slider.on_changed(self._set_dataset_alpha)
        self.legacy_alpha_slider.on_changed(self._set_legacy_alpha)
        self.dot_size_slider.on_changed(self._set_dot_size)
        self.figure.canvas.mpl_connect(
            "button_press_event",
            self._on_plot_click,
        )
        self.figure.canvas.mpl_connect(
            "button_release_event",
            self._on_button_release,
        )
        self.figure.canvas.mpl_connect(
            "motion_notify_event",
            self._on_pointer_motion,
        )
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._render()

    def _auxiliary_button_label(self) -> str:
        return f"Aux: {'ON' if self.show_auxiliary else 'OFF'}"

    def _legacy_result(self, image_path: Path) -> dict:
        cache_key = str(image_path)
        if cache_key not in self.legacy_cache:
            try:
                self.legacy_cache[cache_key] = _run_legacy_3ddfa(image_path)
            except Exception as exc:
                self.legacy_cache[cache_key] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        return self.legacy_cache[cache_key]

    def _render(self) -> None:
        record = self.dataset.records[self.index]
        self.dataset_axis.clear()
        self.legacy_axis.clear()

        _draw_original(
            self.dataset_axis,
            record,
            self.show_auxiliary,
            show_labels=False,
            alpha=self.dataset_alpha,
            dot_size=self.dot_size,
        )
        self.current_legacy_result = self._legacy_result(record.image_path)
        _draw_legacy_3ddfa(
            self.legacy_axis,
            record,
            self.current_legacy_result,
            show_labels=False,
            alpha=self.legacy_alpha,
            dot_size=self.dot_size,
        )

        if self.selected_dataset_index is not None:
            x, y = record.landmarks_xy[self.selected_dataset_index]
            self.dataset_axis.scatter(
                [x],
                [y],
                s=180,
                facecolors="none",
                edgecolors="#ffe600",
                linewidths=2.5,
                zorder=10,
            )

        selected_legacy = (
            self.current_legacy_result.get("landmarks", {}).get(
                self.selected_legacy_id
            )
            if self.selected_legacy_id
            else None
        )
        if selected_legacy:
            with Image.open(record.image_path) as source:
                width, height = source.size
            self.legacy_axis.scatter(
                [float(selected_legacy["x"]) * width],
                [float(selected_legacy["y"]) * height],
                s=180,
                facecolors="none",
                edgecolors="#00e5ff",
                linewidths=2.5,
                zorder=10,
            )

        self.dataset_hover = self.dataset_axis.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            color="white",
            fontsize=8,
            zorder=20,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "black",
                "edgecolor": "#00e5ff",
                "alpha": 0.88,
            },
            arrowprops={"arrowstyle": "->", "color": "#00e5ff"},
        )
        self.dataset_hover.set_visible(False)
        self.legacy_hover = self.legacy_axis.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            color="white",
            fontsize=8,
            zorder=20,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "black",
                "edgecolor": "#39ff14",
                "alpha": 0.88,
            },
            arrowprops={"arrowstyle": "->", "color": "#39ff14"},
        )
        self.legacy_hover.set_visible(False)

        identity = (
            f"#{self.index}/{len(self.dataset) - 1}  "
            f"{Path(record.relative_image_path).name}\n"
            f"subject={record.subject_id}  camera={record.camera_code}"
        )
        self.dataset_axis.set_title(
            f"HOVER OR CLICK DATASET DOT 0–38\n{identity}",
            fontsize=10,
            weight="bold",
        )
        self.legacy_axis.set_title(
            "HOVER OR CLICK LEGACY LANDMARK\n"
            "magenta=sparse, green=dense, yellow=computed",
            fontsize=10,
            weight="bold",
        )
        self.dataset_axis.axis("off")
        self.legacy_axis.axis("off")
        self._updating_index_box = True
        self.index_box.set_val(str(self.index))
        self._updating_index_box = False
        self._update_status()
        self.figure.canvas.draw_idle()

    def _update_status(self, extra: str = "") -> None:
        dataset_text = (
            str(self.selected_dataset_index)
            if self.selected_dataset_index is not None
            else "not selected"
        )
        legacy_text = self.selected_legacy_id or "not selected"
        existing = (
            self.assignments.get(self.selected_legacy_id, "")
            if self.selected_legacy_id
            else ""
        )
        dirty_text = "UNSAVED CHANGES" if self.dirty else "saved"
        suffix = f" | {extra}" if extra else ""
        self.status_text.set_text(
            f"dataset={dataset_text} | legacy={legacy_text} | "
            f"current mapping={existing or '<blank>'} | {dirty_text}{suffix}"
        )
        self.figure.canvas.draw_idle()

    def _set_index(self, new_index: int) -> None:
        new_index = max(0, min(len(self.dataset) - 1, new_index))
        if new_index == self.index:
            return
        self.index = new_index
        self.selected_dataset_index = None
        self._render()

    def _previous(self, _event) -> None:
        self._set_index(self.index - 1)

    def _next(self, _event) -> None:
        self._set_index(self.index + 1)

    def _jump_to_index(self, text: str) -> None:
        if self._updating_index_box:
            return
        try:
            requested_index = int(text)
        except ValueError:
            self.index_box.set_val(str(self.index))
            self._update_status("image index must be an integer")
            return
        self._set_index(requested_index)

    def _nearest_dataset_index(self, x: float, y: float) -> int | None:
        record = self.dataset.records[self.index]
        points = record.landmarks_xy
        distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
        nearest = int(np.argmin(distances))
        with Image.open(record.image_path) as source:
            threshold = max(source.size) * 0.04
        return nearest if float(distances[nearest]) <= threshold else None

    def _nearest_legacy_id(self, x: float, y: float) -> str | None:
        landmarks = self.current_legacy_result.get("landmarks", {})
        if not landmarks:
            return None

        record = self.dataset.records[self.index]
        with Image.open(record.image_path) as source:
            width, height = source.size
            threshold = max(source.size) * 0.04

        landmark_ids = list(landmarks)
        points = np.asarray(
            [
                [
                    float(landmarks[landmark_id]["x"]) * width,
                    float(landmarks[landmark_id]["y"]) * height,
                ]
                for landmark_id in landmark_ids
            ],
            dtype=np.float32,
        )
        distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
        nearest = int(np.argmin(distances))
        return (
            landmark_ids[nearest]
            if float(distances[nearest]) <= threshold
            else None
        )

    def _rerender_preserving_view(self) -> None:
        x_limits = self.dataset_axis.get_xlim()
        y_limits = self.dataset_axis.get_ylim()
        self._render()
        self.dataset_axis.set_xlim(x_limits)
        self.dataset_axis.set_ylim(y_limits)
        self.legacy_axis.set_xlim(x_limits)
        self.legacy_axis.set_ylim(y_limits)
        self.figure.canvas.draw_idle()

    def _set_dataset_alpha(self, value: float) -> None:
        self.dataset_alpha = float(value)
        self._rerender_preserving_view()

    def _set_legacy_alpha(self, value: float) -> None:
        self.legacy_alpha = float(value)
        self._rerender_preserving_view()

    def _set_dot_size(self, value: float) -> None:
        self.dot_size = float(value)
        self._rerender_preserving_view()

    def _dataset_hover_target(self, event):
        record = self.dataset.records[self.index]
        candidates = [
            (
                float(x),
                float(y),
                f"Dataset landmark #{point_index}",
                str(point_index),
            )
            for point_index, (x, y) in enumerate(record.landmarks_xy)
        ]
        if self.show_auxiliary:
            candidates.extend(
                (
                    float(x),
                    float(y),
                    f"Auxiliary point #{point_index}",
                    None,
                )
                for point_index, (x, y) in enumerate(
                    record.auxiliary_points_xy
                )
            )

        display_points = self.dataset_axis.transData.transform(
            np.asarray([[item[0], item[1]] for item in candidates])
        )
        distances = np.sqrt(
            (display_points[:, 0] - event.x) ** 2
            + (display_points[:, 1] - event.y) ** 2
        )
        nearest = int(np.argmin(distances))
        threshold = max(10.0, np.sqrt(self.dot_size) + 7.0)
        if float(distances[nearest]) > threshold:
            return None

        x, y, label, dataset_index = candidates[nearest]
        if dataset_index is None:
            detail = "Alignment helper; not one of the 39 training targets"
        else:
            mapped_names = [
                landmark_name
                for landmark_name, assignment in self.assignments.items()
                if assignment == dataset_index
            ]
            mapping = ", ".join(mapped_names) if mapped_names else "unmapped"
            detail = f"Current worksheet mapping: {mapping}"
        return x, y, f"{label}\n{detail}"

    def _legacy_hover_target(self, event):
        landmarks = self.current_legacy_result.get("landmarks", {})
        if not landmarks:
            return None

        record = self.dataset.records[self.index]
        with Image.open(record.image_path) as source:
            width, height = source.size

        candidates = []
        for landmark_id, landmark in landmarks.items():
            x = float(landmark["x"]) * width
            y = float(landmark["y"]) * height
            label = landmark.get("label", landmark_id)
            model_type = landmark.get("model_type", "unknown")
            mesh_index = landmark.get("mesh_index", "?")
            assignment = self.assignments.get(landmark_id, "")
            candidates.append(
                (
                    x,
                    y,
                    (
                        f"{label}\n"
                        f"id={landmark_id}\n"
                        f"{model_type}:{mesh_index}\n"
                        f"Dataset mapping: {assignment or '<blank>'}"
                    ),
                )
            )

        display_points = self.legacy_axis.transData.transform(
            np.asarray([[item[0], item[1]] for item in candidates])
        )
        distances = np.sqrt(
            (display_points[:, 0] - event.x) ** 2
            + (display_points[:, 1] - event.y) ** 2
        )
        nearest = int(np.argmin(distances))
        threshold = max(10.0, np.sqrt(self.dot_size) + 7.0)
        if float(distances[nearest]) > threshold:
            return None
        return candidates[nearest]

    def _hide_hover_labels(self) -> None:
        changed = False
        for annotation in (self.dataset_hover, self.legacy_hover):
            if annotation is not None and annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        if changed:
            self.figure.canvas.draw_idle()

    def _show_hover_label(self, annotation, target) -> None:
        x, y, text = target
        annotation.xy = (x, y)
        annotation.set_text(text)
        annotation.set_visible(True)
        self.figure.canvas.draw_idle()

    def _on_pointer_motion(self, event) -> None:
        if self._pan_state is not None:
            axis = self._pan_state["axis"]
            x_limits = self._pan_state["x_limits"]
            y_limits = self._pan_state["y_limits"]
            delta_x = event.x - self._pan_state["mouse_x"]
            delta_y = event.y - self._pan_state["mouse_y"]
            x_shift = -delta_x * (x_limits[1] - x_limits[0]) / axis.bbox.width
            y_shift = -delta_y * (y_limits[1] - y_limits[0]) / axis.bbox.height
            new_x_limits = (x_limits[0] + x_shift, x_limits[1] + x_shift)
            new_y_limits = (y_limits[0] + y_shift, y_limits[1] + y_shift)
            for target_axis in (self.dataset_axis, self.legacy_axis):
                target_axis.set_xlim(new_x_limits)
                target_axis.set_ylim(new_y_limits)
            self._hide_hover_labels()
            self.figure.canvas.draw_idle()
            return

        if event.inaxes is self.dataset_axis:
            target = self._dataset_hover_target(event)
            if target is None:
                self._hide_hover_labels()
            else:
                if self.legacy_hover is not None:
                    self.legacy_hover.set_visible(False)
                self._show_hover_label(self.dataset_hover, target)
        elif event.inaxes is self.legacy_axis:
            target = self._legacy_hover_target(event)
            if target is None:
                self._hide_hover_labels()
            else:
                if self.dataset_hover is not None:
                    self.dataset_hover.set_visible(False)
                self._show_hover_label(self.legacy_hover, target)
        else:
            self._hide_hover_labels()

    def _on_scroll(self, event) -> None:
        if event.inaxes not in (self.dataset_axis, self.legacy_axis):
            return
        if event.xdata is None or event.ydata is None:
            return

        scale = 0.8 if event.button == "up" else 1.25
        x_limits = event.inaxes.get_xlim()
        y_limits = event.inaxes.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)
        new_x_limits = (
            x - (x - x_limits[0]) * scale,
            x + (x_limits[1] - x) * scale,
        )
        new_y_limits = (
            y - (y - y_limits[0]) * scale,
            y + (y_limits[1] - y) * scale,
        )
        for axis in (self.dataset_axis, self.legacy_axis):
            axis.set_xlim(new_x_limits)
            axis.set_ylim(new_y_limits)
        self._hide_hover_labels()
        self.figure.canvas.draw_idle()

    def _on_button_release(self, _event) -> None:
        self._pan_state = None

    def _on_plot_click(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in (self.dataset_axis, self.legacy_axis):
            return
        if event.dblclick:
            self._render()
            return
        if event.button == 3:
            self._pan_state = {
                "axis": event.inaxes,
                "mouse_x": event.x,
                "mouse_y": event.y,
                "x_limits": event.inaxes.get_xlim(),
                "y_limits": event.inaxes.get_ylim(),
            }
            self._hide_hover_labels()
            return
        if event.button != 1:
            return
        if event.inaxes is self.dataset_axis:
            nearest = self._nearest_dataset_index(event.xdata, event.ydata)
            if nearest is None:
                self._update_status("click closer to a numbered dataset dot")
                return
            self.selected_dataset_index = nearest
            self._rerender_preserving_view()
        elif event.inaxes is self.legacy_axis:
            nearest = self._nearest_legacy_id(event.xdata, event.ydata)
            if nearest is None:
                self._update_status("click closer to a named legacy dot")
                return
            self.selected_legacy_id = nearest
            self._rerender_preserving_view()

    def _assign(self, _event) -> None:
        if self.selected_legacy_id is None:
            self._update_status("select a named legacy landmark first")
            return
        if self.selected_dataset_index is None:
            self._update_status("select a numbered dataset dot first")
            return
        self.assignments[self.selected_legacy_id] = str(
            self.selected_dataset_index
        )
        self.dirty = True
        self._update_status("assignment staged; press SAVE TXT")

    def _set_special_value(self, value: str) -> None:
        if self.selected_legacy_id is None:
            self._update_status("select a named legacy landmark first")
            return
        self.assignments[self.selected_legacy_id] = value
        self.dirty = True
        self._update_status(f"{value or 'blank'} staged; press SAVE TXT")

    def _mark_none(self, _event) -> None:
        self._set_special_value("NONE")

    def _mark_uncertain(self, _event) -> None:
        self._set_special_value("?")

    def _clear_assignment(self, _event) -> None:
        self._set_special_value("")

    def _save(self, _event) -> None:
        _save_mapping_file(self.mapping_path, self.assignments)
        self.dirty = False
        self._update_status(f"saved to {self.mapping_path.name}")

    def _toggle_auxiliary(self, _event) -> None:
        self.show_auxiliary = not self.show_auxiliary
        self.auxiliary_button.label.set_text(self._auxiliary_button_label())
        self._render()

    def _on_key_press(self, event) -> None:
        if event.key == "left":
            self._previous(event)
        elif event.key == "right":
            self._next(event)
        elif event.key in ("ctrl+s", "cmd+s"):
            self._save(event)

    def show(self) -> None:
        plt.show()


def render_preview(
    dataset: ProfileLandmarkDataset,
    *,
    start_index: int,
    count: int,
    view: str,
    show_auxiliary: bool,
    output_path: Path,
    show_window: bool,
) -> None:
    if start_index < 0 or start_index >= len(dataset):
        raise IndexError(
            f"index must be between 0 and {len(dataset) - 1}, got {start_index}"
        )
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    final_index = min(len(dataset), start_index + count)
    indices = list(range(start_index, final_index))
    if view == "compare":
        figure, axes = plt.subplots(
            len(indices),
            2,
            figsize=(14, 7 * len(indices)),
            squeeze=False,
        )
        for row_index, sample_index in enumerate(indices):
            record = dataset.records[sample_index]
            dataset_axis = axes[row_index, 0]
            legacy_axis = axes[row_index, 1]

            _draw_original(dataset_axis, record, show_auxiliary)
            legacy_result = _run_legacy_3ddfa(record.image_path)
            _draw_legacy_3ddfa(legacy_axis, record, legacy_result)

            identity = (
                f"#{sample_index}  {Path(record.relative_image_path).name} | "
                f"subject={record.subject_id}  camera={record.camera_code}"
            )
            dataset_axis.set_title(
                f"Multi-PIE: unknown landmarks 0–38\n{identity}",
                fontsize=10,
            )
            legacy_axis.set_title(
                "Legacy 3DDFA-V2: named Mini-FaceIQ landmarks\n"
                "magenta=sparse, green=dense, yellow=computed",
                fontsize=10,
            )
            dataset_axis.axis("off")
            legacy_axis.axis("off")
        title = (
            "Human landmark-mapping comparison | fill your answers in "
            "user-custom.txt"
        )
    else:
        columns = min(3, len(indices))
        rows = math.ceil(len(indices) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6 * columns, 6 * rows),
            squeeze=False,
        )
        for axis, sample_index in zip(axes.flat, indices):
            record = dataset.records[sample_index]
            if view == "original":
                _draw_original(axis, record, show_auxiliary)
            else:
                _draw_model_crop(axis, dataset[sample_index], show_auxiliary)

            axis.set_title(
                f"#{sample_index}  {Path(record.relative_image_path).name}\n"
                f"subject={record.subject_id}  camera={record.camera_code}",
                fontsize=10,
            )
            axis.axis("off")

        for unused_axis in axes.flat[len(indices):]:
            unused_axis.axis("off")
        title = "Profile landmarks 0–38 | auxiliary points are yellow X marks"

    figure.suptitle(title, fontsize=13, weight="bold")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved landmark preview to: {output_path}")

    if show_window:
        plt.show()
    plt.close(figure)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw Multi-PIE profile bounding boxes and landmark indices."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Path to MultiPIE_profile_train.txt.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="First annotation row to display.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of consecutive examples to place in the preview.",
    )
    parser.add_argument(
        "--view",
        choices=("compare", "original", "crop"),
        default="compare",
        help=(
            "Compare dataset dots with legacy 3DDFA-V2, show only original "
            "annotations, or show the exact model input crop."
        ),
    )
    parser.add_argument(
        "--show-auxiliary",
        action="store_true",
        help="Draw the five non-training auxiliary points as yellow X marks.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square model-input size used by the crop view.",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.25,
        help="Expansion applied to the supplied face bounding box.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG file to create.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive matplotlib window.",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open the interactive landmark-mapping UI (default). Use "
            "--no-ui to write the legacy static comparison instead."
        ),
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING,
        help="Human-reviewed mapping worksheet edited by UI saves.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    dataset = ProfileLandmarkDataset(
        args.annotations,
        image_size=args.image_size,
        bbox_scale=args.bbox_scale,
        verify_images=True,
    )
    if args.ui:
        viewer = InteractiveLandmarkViewer(
            dataset,
            start_index=args.index,
            show_auxiliary=args.show_auxiliary,
            mapping_path=args.mapping_file,
        )
        viewer.show()
        return

    render_preview(
        dataset,
        start_index=args.index,
        count=args.count,
        view=args.view,
        show_auxiliary=args.show_auxiliary,
        output_path=args.output.expanduser().resolve(),
        show_window=args.show,
    )


if __name__ == "__main__":
    main()
