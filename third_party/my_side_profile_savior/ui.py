"""Tkinter training and QA control room for the profile model factory."""

from __future__ import annotations

import colorsys
from datetime import datetime
import json
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image
import torch

from .benchmark import _legacy_on_crop, normalized_landmark_error
from .dataset import ProfileLandmarkDataset
from .factory_config import FactoryConfig, atomic_json_save
from .factory_state import (
    latest_best_checkpoint,
    request_graceful_stop,
    status_progress,
)
from .inference import FactoryPredictor
from .mapping import load_landmark_mapping
from .train import DEFAULT_ANNOTATIONS, DEFAULT_MAPPING, DEFAULT_RUNS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
QA_POINT_ALPHA_DEFAULT = 0.65
QA_LABEL_ALPHA_DEFAULT = 0.75
QA_HOVER_DISTANCE_PIXELS = 14.0


FIELD_DEFINITIONS = (
    ("Annotation file", "annotation_path", str),
    ("Mapping file", "mapping_path", str),
    ("Runs directory", "runs_root", str),
    ("Device", "device", str),
    ("Seed text", "seed_text", str),
    ("Image size", "image_size", int),
    ("Heatmap size", "heatmap_size", int),
    ("BBox scale", "bbox_scale", float),
    ("Gaussian sigma", "gaussian_sigma", float),
    ("Map-loss weight", "map_loss_weight", float),
    ("Coordinate-loss weight", "coordinate_loss_weight", float),
    ("Learning rate", "learning_rate", float),
    ("Weight decay", "weight_decay", float),
    ("Batch size", "batch_size", int),
    ("Data workers", "workers", int),
    ("Maximum epochs", "epochs", int),
    ("Early-stop patience", "early_stopping_patience", int),
    ("Rotation degrees", "rotation_degrees", float),
    ("Translation fraction", "translation_fraction", float),
    ("Scale jitter", "scale_jitter", float),
    ("Brightness jitter", "brightness_jitter", float),
    ("Contrast jitter", "contrast_jitter", float),
    ("Blur probability", "blur_probability", float),
    ("Resume checkpoint", "resume_checkpoint", str),
    ("Initialize weights from checkpoint", "initial_checkpoint", str),
)

BOOLEAN_FIELDS = (
    ("ImageNet pretrained", "pretrained"),
    ("CUDA AMP", "amp"),
    ("Augmentation", "augmentation"),
    ("Run legacy benchmark", "run_benchmark"),
)


def _default_config() -> FactoryConfig:
    return FactoryConfig(
        annotation_path=str(DEFAULT_ANNOTATIONS),
        mapping_path=str(DEFAULT_MAPPING),
        runs_root=str(DEFAULT_RUNS),
    )


def _default_landmark_color(dataset_index: int) -> str:
    """Return one stable, high-contrast color for a 39-slot dataset index."""

    index = int(dataset_index)
    if not 0 <= index < 39:
        raise ValueError("dataset_index must be between 0 and 38")
    if index == 0:
        return "#0066ff"
    hue = ((2.0 / 3.0) + index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _landmark_color(
    name: str,
    dataset_index: int,
    overrides: dict[str, str],
) -> str:
    return overrides.get(name, _default_landmark_color(dataset_index))


def _hover_label_text(
    name: str,
    dataset_index: int,
    point: dict[str, Any] | None,
) -> str:
    details = [f"{name} [{dataset_index}]"]
    if point is None:
        details.append("not predicted")
        return "\n".join(details)
    if "confidence" in point:
        details.append(f'c={float(point["confidence"]):.3f}')
    if "error_nme" in point:
        details.append(f'e={float(point["error_nme"]):.4f}')
    return "\n".join(details)


def _nearest_landmark_name(
    points: list[dict[str, Any]],
    x: float,
    y: float,
    image_size: tuple[int, int],
    *,
    maximum_distance: float = QA_HOVER_DISTANCE_PIXELS,
) -> str | None:
    """Find the nearest normalized point using an opacity-independent hit box."""

    width, height = image_size
    nearest_name = None
    nearest_distance = float(maximum_distance)
    for point in points:
        point_x = float(point["x"]) * width
        point_y = float(point["y"]) * height
        distance = float(np.hypot(point_x - x, point_y - y))
        if distance <= nearest_distance:
            nearest_distance = distance
            nearest_name = str(point["name"])
    return nearest_name


def _zoomed_view_limits(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    cursor_xy: tuple[float, float],
    scale: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cursor_x, cursor_y = cursor_xy
    return (
        tuple(cursor_x + (value - cursor_x) * scale for value in x_limits),
        tuple(cursor_y + (value - cursor_y) * scale for value in y_limits),
    )


def _panned_view_limits(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    delta_fraction_xy: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    delta_x, delta_y = delta_fraction_xy
    x_shift = (x_limits[1] - x_limits[0]) * delta_x
    y_shift = (y_limits[1] - y_limits[0]) * delta_y
    return (
        (x_limits[0] - x_shift, x_limits[1] - x_shift),
        (y_limits[0] - y_shift, y_limits[1] - y_shift),
    )


def _synchronized_view_limits(
    panel_count: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple((x_limits, y_limits) for _ in range(panel_count))


def _draw_points(
    axis,
    image: Image.Image,
    points: list[dict[str, Any]],
    *,
    color_overrides: dict[str, str],
    point_alpha: float,
    label_alpha: float,
    title: str,
) -> dict[str, Any]:
    axis.imshow(image)
    entries: dict[str, dict[str, Any]] = {}
    for point in points:
        name = str(point["name"])
        dataset_index = int(point["dataset_index"])
        color = _landmark_color(name, dataset_index, color_overrides)
        x = float(point["x"]) * image.width
        y = float(point["y"]) * image.height
        artist = axis.scatter(
            [x],
            [y],
            s=30,
            c=[color],
            edgecolors="black",
            linewidths=0.5,
            alpha=point_alpha,
            zorder=5,
        )
        annotation = axis.annotate(
            _hover_label_text(name, dataset_index, point),
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=7,
            color=color,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "black",
                "alpha": label_alpha,
                "edgecolor": color,
            },
            arrowprops={"arrowstyle": "->", "color": color},
            zorder=20,
        )
        annotation.set_visible(False)
        entries[name] = {
            "point": point,
            "artist": artist,
            "annotation": annotation,
            "xy": (x, y),
        }
    missing_annotation = axis.annotate(
        "",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "black",
            "alpha": label_alpha,
            "edgecolor": "white",
        },
        zorder=20,
    )
    missing_annotation.set_visible(False)
    axis.set_xlim(-0.5, image.width - 0.5)
    axis.set_ylim(image.height - 0.5, -0.5)
    axis.set_title(title)
    axis.axis("off")
    return {
        "axis": axis,
        "image_size": image.size,
        "points": points,
        "entries": entries,
        "missing_annotation": missing_annotation,
    }


class FactoryApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mini-FaceIQ Side-Profile Model Factory")
        self.geometry("1380x900")
        self.minsize(1050, 700)

        self.variables: dict[str, tk.Variable] = {}
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.active_run_dir: Path | None = None
        self.predictor: FactoryPredictor | None = None
        self.predictor_key: tuple[str, int] | None = None
        self.qa_image_path: Path | None = None
        self.qa_figure = None
        self.qa_canvas: FigureCanvasTkAgg | None = None
        self.qa_point_alpha = tk.DoubleVar(value=QA_POINT_ALPHA_DEFAULT)
        self.qa_label_alpha = tk.DoubleVar(value=QA_LABEL_ALPHA_DEFAULT)
        self.qa_color_landmark = tk.StringVar()
        self.qa_color_overrides: dict[str, str] = {}
        self.qa_landmark_indices: dict[str, int] = {}
        self.qa_panels: list[dict[str, Any]] = []
        self.qa_hover_name: str | None = None
        self.qa_pan_state: dict[str, Any] | None = None
        self.qa_reset_limits: tuple[
            tuple[float, float],
            tuple[float, float],
        ] | None = None
        self.curve_figure = None
        self.curve_canvas: FigureCanvasTkAgg | None = None
        self.curves_mtime_ns: int | None = None

        self._build_ui()
        self._load_config_into_fields(_default_config())
        self.after(250, self._poll)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.training_tab = ttk.Frame(notebook)
        self.qa_tab = ttk.Frame(notebook)
        self.log_tab = ttk.Frame(notebook)
        self.curves_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text="Training Factory")
        notebook.add(self.qa_tab, text="QA Laboratory")
        notebook.add(self.curves_tab, text="Curves")
        notebook.add(self.log_tab, text="Logs")
        self._build_training_tab()
        self._build_qa_tab()
        self._build_curves_tab()
        self._build_log_tab()

    def _build_training_tab(self) -> None:
        canvas = tk.Canvas(self.training_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.training_tab,
            orient="vertical",
            command=canvas.yview,
        )
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        title = ttk.Label(
            inner,
            text="Side-Profile Landmark Training Factory",
            font=("TkDefaultFont", 16, "bold"),
        )
        title.grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=8)

        for row, (label, key, _converter) in enumerate(
            FIELD_DEFINITIONS,
            start=1,
        ):
            ttk.Label(inner, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=3,
            )
            variable = tk.StringVar()
            self.variables[key] = variable
            if key == "device":
                widget = ttk.Combobox(
                    inner,
                    textvariable=variable,
                    values=("auto", "cuda", "cpu"),
                    state="readonly",
                    width=58,
                )
            else:
                widget = ttk.Entry(inner, textvariable=variable, width=62)
            widget.grid(row=row, column=1, sticky="ew", padx=8, pady=3)

            if key in {
                "annotation_path",
                "mapping_path",
                "resume_checkpoint",
                "initial_checkpoint",
            }:
                ttk.Button(
                    inner,
                    text="Browse",
                    command=lambda target=key: self._browse_file(target),
                ).grid(row=row, column=2, padx=4)
            elif key == "runs_root":
                ttk.Button(
                    inner,
                    text="Browse",
                    command=lambda target=key: self._browse_directory(target),
                ).grid(row=row, column=2, padx=4)

        boolean_start = len(FIELD_DEFINITIONS) + 2
        architecture_frame = ttk.LabelFrame(inner, text="Locked architecture")
        architecture_frame.grid(
            row=boolean_start,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=8,
            pady=8,
        )
        ttk.Label(
            architecture_frame,
            text="Model: MobileNetV3-Large + 31 heatmaps",
        ).pack(side="left", padx=10, pady=8)
        ttk.Label(
            architecture_frame,
            text="Optimizer: AdamW + cosine decay",
        ).pack(side="left", padx=10, pady=8)
        ttk.Label(
            architecture_frame,
            text="Loss: Gaussian-map MSE + coordinate SmoothL1",
        ).pack(side="left", padx=10, pady=8)

        boolean_frame = ttk.LabelFrame(inner, text="Switches")
        boolean_frame.grid(
            row=boolean_start + 1,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=8,
            pady=8,
        )
        for column, (label, key) in enumerate(BOOLEAN_FIELDS):
            variable = tk.BooleanVar()
            self.variables[key] = variable
            ttk.Checkbutton(
                boolean_frame,
                text=label,
                variable=variable,
            ).grid(row=0, column=column, padx=10, pady=8, sticky="w")

        controls = ttk.Frame(inner)
        controls.grid(
            row=boolean_start + 2,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=8,
            pady=8,
        )
        ttk.Button(controls, text="START", command=self._start_training).pack(
            side="left",
            padx=4,
        )
        ttk.Button(controls, text="GRACEFUL STOP", command=self._stop_training).pack(
            side="left",
            padx=4,
        )
        ttk.Button(
            controls,
            text="Reset defaults",
            command=lambda: self._load_config_into_fields(_default_config()),
        ).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(inner, mode="determinate", maximum=100)
        self.progress.grid(
            row=boolean_start + 3,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=8,
            pady=5,
        )
        self.training_status = tk.StringVar(value="Idle")
        ttk.Label(inner, textvariable=self.training_status).grid(
            row=boolean_start + 4,
            column=0,
            columnspan=3,
            sticky="w",
            padx=8,
            pady=5,
        )
        inner.columnconfigure(1, weight=1)

    def _build_qa_tab(self) -> None:
        controls = ttk.LabelFrame(self.qa_tab, text="QA inputs")
        controls.pack(fill="x", padx=8, pady=8)

        self.qa_checkpoint = tk.StringVar()
        self.qa_index = tk.StringVar(value="0")
        self.qa_mirror = tk.BooleanVar(value=False)
        ttk.Label(controls, text="Checkpoint").grid(row=0, column=0, padx=5)
        ttk.Entry(
            controls,
            textvariable=self.qa_checkpoint,
            width=90,
        ).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(
            controls,
            text="Browse",
            command=self._browse_qa_checkpoint,
        ).grid(row=0, column=2, padx=5)
        ttk.Label(controls, text="Dataset index").grid(row=1, column=0, padx=5)
        ttk.Entry(controls, textvariable=self.qa_index, width=12).grid(
            row=1,
            column=1,
            sticky="w",
            padx=5,
        )
        ttk.Button(
            controls,
            text="Random dataset sample",
            command=self._random_qa_index,
        ).grid(row=1, column=2, padx=5)
        ttk.Checkbutton(
            controls,
            text="Mirror uploaded profile before custom inference",
            variable=self.qa_mirror,
        ).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Button(
            controls,
            text="RUN DATASET QA",
            command=self._run_dataset_qa,
        ).grid(row=3, column=0, padx=5, pady=6)
        ttk.Button(
            controls,
            text="Choose upload",
            command=self._choose_upload,
        ).grid(row=3, column=1, sticky="w", padx=5, pady=6)
        ttk.Button(
            controls,
            text="RUN UPLOAD QA",
            command=self._run_upload_qa,
        ).grid(row=3, column=2, padx=5, pady=6)

        appearance = ttk.LabelFrame(controls, text="Landmark appearance")
        appearance.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=5,
            pady=6,
        )
        ttk.Label(appearance, text="Point opacity").grid(
            row=0,
            column=0,
            sticky="w",
            padx=5,
        )
        ttk.Scale(
            appearance,
            from_=0.0,
            to=1.0,
            variable=self.qa_point_alpha,
            command=self._on_qa_appearance_change,
        ).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(appearance, text="Hover-label opacity").grid(
            row=1,
            column=0,
            sticky="w",
            padx=5,
        )
        ttk.Scale(
            appearance,
            from_=0.0,
            to=1.0,
            variable=self.qa_label_alpha,
            command=self._on_qa_appearance_change,
        ).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(appearance, text="Landmark color").grid(
            row=0,
            column=2,
            sticky="w",
            padx=5,
        )
        self.qa_color_selector = ttk.Combobox(
            appearance,
            textvariable=self.qa_color_landmark,
            state="readonly",
            width=24,
        )
        self.qa_color_selector.grid(row=0, column=3, sticky="ew", padx=5)
        self.qa_color_selector.bind(
            "<<ComboboxSelected>>",
            self._on_qa_color_selection,
        )
        self.qa_color_button = tk.Button(
            appearance,
            text="Choose color",
            command=self._choose_qa_landmark_color,
        )
        self.qa_color_button.grid(row=1, column=2, padx=5, pady=3)
        ttk.Button(
            appearance,
            text="Reset color",
            command=self._reset_qa_landmark_color,
        ).grid(row=1, column=3, padx=5, pady=3, sticky="w")
        appearance.columnconfigure(1, weight=1)
        appearance.columnconfigure(3, weight=1)
        controls.columnconfigure(1, weight=1)

        self.qa_status = tk.StringVar(value="Choose a checkpoint and sample.")
        ttk.Label(self.qa_tab, textvariable=self.qa_status).pack(
            fill="x",
            padx=8,
            pady=4,
        )
        self.qa_plot_frame = ttk.Frame(self.qa_tab)
        self.qa_plot_frame.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_log_tab(self) -> None:
        self.log_text = tk.Text(self.log_tab, wrap="none")
        self.log_text.pack(fill="both", expand=True)

    def _build_curves_tab(self) -> None:
        ttk.Label(
            self.curves_tab,
            text="Training and validation curves refresh after every epoch.",
        ).pack(fill="x", padx=8, pady=8)
        self.curve_plot_frame = ttk.Frame(self.curves_tab)
        self.curve_plot_frame.pack(fill="both", expand=True, padx=8, pady=8)

    def _load_config_into_fields(self, config: FactoryConfig) -> None:
        values = config.to_dict()
        for _label, key, _converter in FIELD_DEFINITIONS:
            self.variables[key].set(str(values[key]))
        for _label, key in BOOLEAN_FIELDS:
            self.variables[key].set(bool(values[key]))

    def _config_from_fields(self) -> FactoryConfig:
        values: dict[str, Any] = {}
        for _label, key, converter in FIELD_DEFINITIONS:
            raw = str(self.variables[key].get()).strip()
            values[key] = converter(raw) if converter is not str else raw
        for _label, key in BOOLEAN_FIELDS:
            values[key] = bool(self.variables[key].get())
        config = FactoryConfig(**values)
        config.validate()
        load_landmark_mapping(config.mapping_path)
        return config

    def _browse_file(self, key: str) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.variables[key].set(path)

    def _browse_directory(self, key: str) -> None:
        path = filedialog.askdirectory()
        if path:
            self.variables[key].set(path)

    def _browse_qa_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt")])
        if path:
            self.qa_checkpoint.set(path)

    def _start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Training active", "A training process is already running.")
            return
        try:
            config = self._config_from_fields()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        runs_root = Path(config.runs_root).expanduser().resolve()
        runs_root.mkdir(parents=True, exist_ok=True)
        launcher_path = runs_root / (
            "launcher-"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".json"
        )
        atomic_json_save(config.to_dict(), launcher_path)
        command = [
            sys.executable,
            "-m",
            "third_party.my_side_profile_savior.train",
            "--config",
            str(launcher_path),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.training_status.set("Starting training process...")
        threading.Thread(
            target=self._read_process_output,
            daemon=True,
        ).start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.output_queue.put(line.rstrip())

    def _stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            messagebox.showinfo("No training", "No active training process.")
            return
        if self.active_run_dir is None:
            messagebox.showwarning(
                "Run starting",
                "The run directory is not available yet. Try Stop again shortly.",
            )
            return
        request_graceful_stop(self.active_run_dir)
        self.training_status.set("Graceful stop requested; finishing current batch...")

    def _poll(self) -> None:
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")

        runs_root_raw = str(self.variables.get("runs_root", tk.StringVar()).get())
        if runs_root_raw:
            active_path = Path(runs_root_raw).expanduser() / "active_run.json"
            if active_path.is_file():
                try:
                    with active_path.open("r", encoding="utf-8") as handle:
                        active = json.load(handle)
                    self.active_run_dir = Path(active["run_dir"])
                    status_path = Path(active["status_path"])
                    if status_path.is_file():
                        with status_path.open("r", encoding="utf-8") as handle:
                            status = json.load(handle)
                        percent, text = status_progress(status)
                        self.progress["value"] = percent
                        self.training_status.set(text)
                    best_path = latest_best_checkpoint(self.active_run_dir)
                    if best_path is not None:
                        self.qa_checkpoint.set(str(best_path))
                    self._refresh_curves()
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    pass

        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            if code == 0:
                self.training_status.set("Training process finished.")
            else:
                self.training_status.set(f"Training process failed with exit code {code}.")
            self.process = None
        self.after(350, self._poll)

    def _refresh_curves(self) -> None:
        if self.active_run_dir is None:
            return
        curves_path = self.active_run_dir / "curves.json"
        if not curves_path.is_file():
            return
        mtime_ns = curves_path.stat().st_mtime_ns
        if self.curves_mtime_ns == mtime_ns:
            return
        with curves_path.open("r", encoding="utf-8") as handle:
            curves = json.load(handle)
        if not curves:
            return

        epochs = [int(item["epoch"]) + 1 for item in curves]
        train_nme = [float(item["train"]["nme"]) for item in curves]
        validation_nme = [
            float(item["validation"]["nme"]) for item in curves
        ]
        train_loss = [float(item["train"]["loss"]) for item in curves]
        validation_loss = [
            float(item["validation"]["loss"]) for item in curves
        ]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs, train_nme, label="train")
        axes[0].plot(epochs, validation_nme, label="validation")
        axes[0].set_title("Normalized landmark error")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(epochs, train_loss, label="train")
        axes[1].plot(epochs, validation_loss, label="validation")
        axes[1].set_title("Masked combined loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        figure.tight_layout()

        if self.curve_canvas is not None:
            self.curve_canvas.get_tk_widget().destroy()
        if self.curve_figure is not None:
            plt.close(self.curve_figure)
        self.curve_figure = figure
        self.curve_canvas = FigureCanvasTkAgg(
            figure,
            master=self.curve_plot_frame,
        )
        self.curve_canvas.draw()
        self.curve_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.curves_mtime_ns = mtime_ns

    def _predictor_for_qa(self) -> FactoryPredictor:
        checkpoint = Path(self.qa_checkpoint.get()).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Choose a valid best.pt or last.pt checkpoint")
        key = (str(checkpoint), checkpoint.stat().st_mtime_ns)
        if self.predictor is None or self.predictor_key != key:
            device = str(self.variables["device"].get())
            self.predictor = FactoryPredictor(checkpoint, device=device)
            self.predictor_key = key
        return self.predictor

    def _dataset_for_qa(self) -> ProfileLandmarkDataset:
        mapping = load_landmark_mapping(
            str(self.variables["mapping_path"].get())
        )
        return ProfileLandmarkDataset(
            str(self.variables["annotation_path"].get()),
            image_size=int(str(self.variables["image_size"].get())),
            bbox_scale=float(str(self.variables["bbox_scale"].get())),
            verify_images=True,
            mapping=mapping,
        )

    def _random_qa_index(self) -> None:
        try:
            dataset = self._dataset_for_qa()
            self.qa_index.set(str(random.randrange(len(dataset))))
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc))

    def _configure_qa_landmarks(self, mapping) -> None:
        self.qa_landmark_indices = {
            entry.name: int(entry.dataset_index)
            for entry in mapping.confirmed_entries
        }
        names = list(self.qa_landmark_indices)
        self.qa_color_selector.configure(values=names)
        selected = self.qa_color_landmark.get()
        if selected not in self.qa_landmark_indices:
            self.qa_color_landmark.set(names[0] if names else "")
        self._refresh_qa_color_button()

    def _selected_qa_color(self) -> str:
        name = self.qa_color_landmark.get()
        dataset_index = self.qa_landmark_indices.get(name)
        if dataset_index is None:
            return "#d9d9d9"
        return _landmark_color(
            name,
            dataset_index,
            self.qa_color_overrides,
        )

    def _refresh_qa_color_button(self) -> None:
        color = self._selected_qa_color()
        self.qa_color_button.configure(
            background=color,
            activebackground=color,
        )

    def _on_qa_color_selection(self, _event=None) -> None:
        self._refresh_qa_color_button()

    def _choose_qa_landmark_color(self) -> None:
        name = self.qa_color_landmark.get()
        if name not in self.qa_landmark_indices:
            messagebox.showwarning(
                "No landmark",
                "Run QA or select a confirmed landmark first.",
            )
            return
        _rgb, color = colorchooser.askcolor(
            color=self._selected_qa_color(),
            title=f"Choose color for {name}",
            parent=self,
        )
        if color:
            self.qa_color_overrides[name] = color
            self._refresh_qa_color_button()
            self._apply_qa_appearance()

    def _reset_qa_landmark_color(self) -> None:
        name = self.qa_color_landmark.get()
        if not name:
            return
        self.qa_color_overrides.pop(name, None)
        self._refresh_qa_color_button()
        self._apply_qa_appearance()

    def _on_qa_appearance_change(self, _value=None) -> None:
        self._apply_qa_appearance()

    def _apply_qa_appearance(self) -> None:
        point_alpha = float(self.qa_point_alpha.get())
        label_alpha = float(self.qa_label_alpha.get())
        for panel in self.qa_panels:
            for name, entry in panel["entries"].items():
                point = entry["point"]
                color = _landmark_color(
                    name,
                    int(point["dataset_index"]),
                    self.qa_color_overrides,
                )
                entry["artist"].set_facecolor(color)
                entry["artist"].set_alpha(point_alpha)
                annotation = entry["annotation"]
                annotation.set_color(color)
                annotation.get_bbox_patch().set_alpha(label_alpha)
                annotation.get_bbox_patch().set_edgecolor(color)
                if annotation.arrow_patch is not None:
                    annotation.arrow_patch.set_color(color)
            missing = panel["missing_annotation"]
            missing.get_bbox_patch().set_alpha(label_alpha)
            if self.qa_hover_name is not None:
                color = _landmark_color(
                    self.qa_hover_name,
                    self.qa_landmark_indices[self.qa_hover_name],
                    self.qa_color_overrides,
                )
                missing.set_color(color)
                missing.get_bbox_patch().set_edgecolor(color)
        if self.qa_canvas is not None:
            self.qa_canvas.draw_idle()

    def _hide_qa_hover(self, *, redraw: bool = True) -> None:
        changed = False
        for panel in self.qa_panels:
            for entry in panel["entries"].values():
                annotation = entry["annotation"]
                if annotation.get_visible():
                    annotation.set_visible(False)
                    changed = True
            missing = panel["missing_annotation"]
            if missing.get_visible():
                missing.set_visible(False)
                changed = True
        self.qa_hover_name = None
        if redraw and changed and self.qa_canvas is not None:
            self.qa_canvas.draw_idle()

    def _show_qa_hover(self, name: str) -> None:
        if name not in self.qa_landmark_indices:
            self._hide_qa_hover()
            return
        self._hide_qa_hover(redraw=False)
        self.qa_hover_name = name
        dataset_index = self.qa_landmark_indices[name]
        color = _landmark_color(
            name,
            dataset_index,
            self.qa_color_overrides,
        )
        for panel in self.qa_panels:
            entry = panel["entries"].get(name)
            if entry is not None:
                entry["annotation"].set_text(
                    _hover_label_text(name, dataset_index, entry["point"])
                )
                entry["annotation"].set_visible(True)
                continue
            missing = panel["missing_annotation"]
            missing.set_text(_hover_label_text(name, dataset_index, None))
            missing.set_color(color)
            missing.get_bbox_patch().set_edgecolor(color)
            missing.set_visible(True)
        self._apply_qa_appearance()

    def _apply_synchronized_limits(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        limits = _synchronized_view_limits(
            len(self.qa_panels),
            x_limits,
            y_limits,
        )
        for panel, (panel_x, panel_y) in zip(self.qa_panels, limits):
            panel["axis"].set_xlim(panel_x)
            panel["axis"].set_ylim(panel_y)
        if self.qa_canvas is not None:
            self.qa_canvas.draw_idle()

    def _reset_qa_view(self) -> None:
        if self.qa_reset_limits is None:
            return
        self._apply_synchronized_limits(*self.qa_reset_limits)

    def _qa_panel_for_axis(self, axis) -> dict[str, Any] | None:
        return next(
            (
                panel
                for panel in self.qa_panels
                if panel["axis"] is axis
            ),
            None,
        )

    def _on_qa_button_press(self, event) -> None:
        if self._qa_panel_for_axis(event.inaxes) is None:
            return
        if event.dblclick:
            self.qa_pan_state = None
            self._hide_qa_hover(redraw=False)
            self._reset_qa_view()
            return
        if event.button == 3:
            axis = event.inaxes
            self.qa_pan_state = {
                "mouse_xy": (float(event.x), float(event.y)),
                "x_limits": tuple(float(value) for value in axis.get_xlim()),
                "y_limits": tuple(float(value) for value in axis.get_ylim()),
                "axis_width": max(1.0, float(axis.bbox.width)),
                "axis_height": max(1.0, float(axis.bbox.height)),
            }
            self._hide_qa_hover()

    def _on_qa_button_release(self, _event) -> None:
        self.qa_pan_state = None

    def _on_qa_pointer_motion(self, event) -> None:
        if self.qa_pan_state is not None:
            state = self.qa_pan_state
            delta_fraction = (
                (float(event.x) - state["mouse_xy"][0])
                / state["axis_width"],
                (float(event.y) - state["mouse_xy"][1])
                / state["axis_height"],
            )
            x_limits, y_limits = _panned_view_limits(
                state["x_limits"],
                state["y_limits"],
                delta_fraction,
            )
            self._apply_synchronized_limits(x_limits, y_limits)
            return
        panel = self._qa_panel_for_axis(event.inaxes)
        if panel is None or event.xdata is None or event.ydata is None:
            self._hide_qa_hover()
            return
        name = _nearest_landmark_name(
            panel["points"],
            float(event.xdata),
            float(event.ydata),
            panel["image_size"],
        )
        if name is None:
            self._hide_qa_hover()
        elif name != self.qa_hover_name:
            self._show_qa_hover(name)

    def _on_qa_scroll(self, event) -> None:
        panel = self._qa_panel_for_axis(event.inaxes)
        if panel is None or event.xdata is None or event.ydata is None:
            return
        step = float(getattr(event, "step", 0.0))
        scale = 1.0 / 1.2 if step > 0 else 1.2
        axis = panel["axis"]
        x_limits, y_limits = _zoomed_view_limits(
            tuple(float(value) for value in axis.get_xlim()),
            tuple(float(value) for value in axis.get_ylim()),
            (float(event.xdata), float(event.ydata)),
            scale,
        )
        self._hide_qa_hover(redraw=False)
        self._apply_synchronized_limits(x_limits, y_limits)

    def _activate_qa_interactions(
        self,
        panels: list[dict[str, Any]],
    ) -> None:
        self.qa_panels = panels
        self.qa_hover_name = None
        self.qa_pan_state = None
        if not panels:
            self.qa_reset_limits = None
            return
        first_axis = panels[0]["axis"]
        self.qa_reset_limits = (
            tuple(float(value) for value in first_axis.get_xlim()),
            tuple(float(value) for value in first_axis.get_ylim()),
        )
        canvas = self.qa_figure.canvas
        canvas.mpl_connect("button_press_event", self._on_qa_button_press)
        canvas.mpl_connect("button_release_event", self._on_qa_button_release)
        canvas.mpl_connect("motion_notify_event", self._on_qa_pointer_motion)
        canvas.mpl_connect("scroll_event", self._on_qa_scroll)
        self._apply_qa_appearance()

    def _show_figure(self, figure) -> None:
        if self.qa_canvas is not None:
            self.qa_canvas.get_tk_widget().destroy()
        if self.qa_figure is not None:
            plt.close(self.qa_figure)
        self.qa_panels = []
        self.qa_hover_name = None
        self.qa_pan_state = None
        self.qa_reset_limits = None
        self.qa_figure = figure
        self.qa_canvas = FigureCanvasTkAgg(figure, master=self.qa_plot_frame)
        self.qa_canvas.draw()
        self.qa_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _run_dataset_qa(self) -> None:
        try:
            predictor = self._predictor_for_qa()
            dataset = self._dataset_for_qa()
            index = int(self.qa_index.get())
            sample = dataset[index]
            record = dataset.records[index]
            crop_box = tuple(int(value) for value in sample["crop_xyxy"].tolist())
            with Image.open(record.image_path) as source:
                crop = source.convert("RGB").crop(crop_box)
            custom = predictor.predict_crop(crop)
            legacy = _legacy_on_crop(crop)
            mapping = load_landmark_mapping(
                str(self.variables["mapping_path"].get())
            )
            self._configure_qa_landmarks(mapping)
            if predictor.names_by_index != mapping.names_by_dataset_index:
                raise ValueError(
                    "Checkpoint mapping does not match the current "
                    "user-custom.txt. Select the matching worksheet/checkpoint."
                )
            targets = sample["landmarks"].numpy()
            truth_points = [
                {
                    "name": entry.name,
                    "dataset_index": int(entry.dataset_index),
                    "x": float(targets[entry.model_index, 0]),
                    "y": float(targets[entry.model_index, 1]),
                }
                for entry in mapping.confirmed_entries
            ]
            legacy_points = []
            for entry in mapping.confirmed_entries:
                item = legacy.get("landmarks", {}).get(entry.name)
                if item is not None:
                    legacy_points.append(
                        {
                            "name": entry.name,
                            "dataset_index": int(entry.dataset_index),
                            "x": item["x"],
                            "y": item["y"],
                        }
                    )

            custom_by_name = {
                item["name"]: np.asarray([item["x"], item["y"]])
                for item in custom["predictions"]
            }
            legacy_by_name = {
                item["name"]: np.asarray([item["x"], item["y"]])
                for item in legacy_points
            }
            errors = []
            for entry in mapping.confirmed_entries:
                target = targets[entry.model_index]
                custom_error = normalized_landmark_error(
                    custom_by_name.get(entry.name),
                    target,
                    crop_xyxy=sample["crop_xyxy"].numpy(),
                    bbox_xyxy=sample["bbox_xyxy"].numpy(),
                )
                errors.append(custom_error)
                for item in custom["predictions"]:
                    if item["name"] == entry.name:
                        item["error_nme"] = custom_error
                for item in legacy_points:
                    if item["name"] == entry.name:
                        item["error_nme"] = normalized_landmark_error(
                            legacy_by_name.get(entry.name),
                            target,
                            crop_xyxy=sample["crop_xyxy"].numpy(),
                            bbox_xyxy=sample["bbox_xyxy"].numpy(),
                        )

            figure, axes = plt.subplots(1, 3, figsize=(15, 5))
            panels = [
                _draw_points(
                    axes[0],
                    crop,
                    truth_points,
                    color_overrides=self.qa_color_overrides,
                    point_alpha=float(self.qa_point_alpha.get()),
                    label_alpha=float(self.qa_label_alpha.get()),
                    title="Truth",
                ),
                _draw_points(
                    axes[1],
                    crop,
                    custom["predictions"],
                    color_overrides=self.qa_color_overrides,
                    point_alpha=float(self.qa_point_alpha.get()),
                    label_alpha=float(self.qa_label_alpha.get()),
                    title=f'Custom | {custom["device"]}',
                ),
                _draw_points(
                    axes[2],
                    crop,
                    legacy_points,
                    color_overrides=self.qa_color_overrides,
                    point_alpha=float(self.qa_point_alpha.get()),
                    label_alpha=float(self.qa_label_alpha.get()),
                    title="Legacy 3DDFA-V2",
                ),
            ]
            figure.tight_layout()
            self._show_figure(figure)
            self._activate_qa_interactions(panels)

            self.qa_status.set(
                f"Dataset #{index} | confirmed-point custom NME "
                f"{float(np.mean(errors)):.5f} | no graduation claim from one image"
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            messagebox.showerror("QA runtime error", str(exc))
        except Exception as exc:
            messagebox.showerror("QA error", str(exc))

    def _choose_upload(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.qa_image_path = Path(path)
            self.qa_status.set(f"Upload selected: {path}")

    def _run_upload_qa(self) -> None:
        if self.qa_image_path is None:
            messagebox.showwarning("No upload", "Choose an upload first.")
            return
        try:
            predictor = self._predictor_for_qa()
            with Image.open(self.qa_image_path) as source:
                image = source.convert("RGB")
            custom = predictor.predict_upload(
                image,
                mirror=self.qa_mirror.get(),
            )
            if custom.get("error"):
                raise RuntimeError(custom["error"])

            from third_party.providers.side_3ddfa import detect_side

            legacy = detect_side(str(self.qa_image_path))
            mapping = load_landmark_mapping(
                str(self.variables["mapping_path"].get())
            )
            self._configure_qa_landmarks(mapping)
            if predictor.names_by_index != mapping.names_by_dataset_index:
                raise ValueError(
                    "Checkpoint mapping does not match the current "
                    "user-custom.txt. Select the matching worksheet/checkpoint."
                )
            legacy_points = []
            for entry in mapping.confirmed_entries:
                item = legacy.get("landmarks", {}).get(entry.name)
                if item is not None:
                    legacy_points.append(
                        {
                            "name": entry.name,
                            "dataset_index": int(entry.dataset_index),
                            "x": item["x"],
                            "y": item["y"],
                        }
                    )

            figure, axes = plt.subplots(1, 2, figsize=(12, 6))
            panels = [
                _draw_points(
                    axes[0],
                    image,
                    custom["predictions"],
                    color_overrides=self.qa_color_overrides,
                    point_alpha=float(self.qa_point_alpha.get()),
                    label_alpha=float(self.qa_label_alpha.get()),
                    title=(
                        f'Custom | {custom["device"]} | '
                        f'mirror={custom["mirror"]}'
                    ),
                ),
                _draw_points(
                    axes[1],
                    image,
                    legacy_points,
                    color_overrides=self.qa_color_overrides,
                    point_alpha=float(self.qa_point_alpha.get()),
                    label_alpha=float(self.qa_label_alpha.get()),
                    title="Legacy 3DDFA-V2",
                ),
            ]
            figure.tight_layout()
            self._show_figure(figure)
            self._activate_qa_interactions(panels)
            self.qa_status.set(
                "Upload QA shows predictions only; no ground truth means no "
                "accuracy or graduation claim. "
                f'FaceBoxes bbox={custom.get("bbox_xyxy")} '
                f'crop={custom.get("crop_xyxy")}.'
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            messagebox.showerror("QA runtime error", str(exc))
        except Exception as exc:
            messagebox.showerror("QA error", str(exc))


def main() -> None:
    app = FactoryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
