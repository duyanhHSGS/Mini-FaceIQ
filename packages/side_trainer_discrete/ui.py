"""Tkinter control room for independent landmark specialists."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from .config import (
    DEFAULT_CONTRIBUTIONS_ROOT,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_RUNS_ROOT,
    LANDMARK_IDS,
    TrainerConfig,
)
from .dataset import LandmarkRecord, load_records


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]


class DiscreteTrainerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mini-FaceIQ — Discrete Landmark HQ")
        self.geometry("1220x800")
        self.minsize(1020, 680)
        self.configure(bg="#10131a")
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.records: list[LandmarkRecord] = []
        self.preview_index = 0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.active_run: Path | None = None
        self._build_style()
        self._build_ui()
        self._reload_labels()
        self.after(250, self._poll)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#171b24", foreground="#edf2f7", fieldbackground="#0f131a")
        style.configure("TFrame", background="#171b24")
        style.configure("Card.TFrame", background="#1e2430")
        style.configure("TLabel", background="#171b24", foreground="#edf2f7")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#f472b6")
        style.configure("Sub.TLabel", foreground="#9ca3af")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="#10131a", background="#f472b6")
        style.map("Accent.TButton", background=[("active", "#fb9acb")])
        style.configure("TEntry", padding=6)
        style.configure("TCombobox", padding=6)
        style.configure("Horizontal.TProgressbar", background="#f472b6", troughcolor="#0f131a")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(22, 18))
        header.pack(fill="x")
        ttk.Label(header, text="🧠 DISCRETE LANDMARK HQ", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="One landmark. One model. Zero multi-head sibling drama. Porion is selected first.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        controls = ttk.Frame(body, style="Card.TFrame", padding=18)
        monitor = ttk.Frame(body, style="Card.TFrame", padding=18)
        body.add(controls, weight=2)
        body.add(monitor, weight=3)
        self._build_controls(controls)
        self._build_monitor(monitor)

    def _field(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable, width=20).grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Specialist settings", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.landmark = tk.StringVar(value="porion")
        self.device = tk.StringVar(value="auto")
        self.epochs = tk.IntVar(value=150)
        self.batch_size = tk.IntVar(value=32)
        self.workers = tk.IntVar(value=4)
        self.patience = tk.IntVar(value=20)
        self.learning_rate = tk.DoubleVar(value=3e-4)
        self.pretrained = tk.BooleanVar(value=True)
        self.amp = tk.BooleanVar(value=True)
        self.augmentation = tk.BooleanVar(value=True)
        self.resume = tk.StringVar(value="")

        ttk.Label(parent, text="Landmark").grid(row=1, column=0, sticky="w", pady=5)
        landmark_box = ttk.Combobox(parent, textvariable=self.landmark, values=LANDMARK_IDS, state="readonly")
        landmark_box.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=5)
        landmark_box.bind("<<ComboboxSelected>>", lambda _event: self._reload_labels())
        ttk.Label(parent, text="Device").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(parent, textvariable=self.device, values=("auto", "cuda", "cpu"), state="readonly").grid(row=2, column=1, sticky="ew", padx=(12, 0), pady=5)
        self._field(parent, 3, "Epochs", self.epochs)
        self._field(parent, 4, "Batch size", self.batch_size)
        self._field(parent, 5, "Workers", self.workers)
        self._field(parent, 6, "Early-stop patience", self.patience)
        self._field(parent, 7, "Learning rate", self.learning_rate)
        ttk.Checkbutton(parent, text="ImageNet pretrained", variable=self.pretrained).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 2))
        ttk.Checkbutton(parent, text="CUDA mixed precision", variable=self.amp).grid(row=9, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(parent, text="Safe geometric + photo augmentation", variable=self.augmentation).grid(row=10, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Separator(parent).grid(row=11, column=0, columnspan=2, sticky="ew", pady=14)
        ttk.Label(parent, text="Resume checkpoint", style="Sub.TLabel").grid(row=12, column=0, columnspan=2, sticky="w")
        ttk.Entry(parent, textvariable=self.resume).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(5, 5))
        ttk.Button(parent, text="Choose last.pt…", command=self._choose_resume).grid(row=14, column=0, columnspan=2, sticky="ew")
        buttons = ttk.Frame(parent, style="Card.TFrame")
        buttons.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(buttons, text="TRAIN SPECIALIST ⚡", style="Accent.TButton", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(buttons, text="Graceful stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.label_count = ttk.Label(parent, text="Loading labels…", style="Sub.TLabel")
        self.label_count.grid(row=16, column=0, columnspan=2, sticky="w", pady=(14, 0))

    def _build_monitor(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Human truth preview", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.preview = tk.Label(parent, bg="#0b0e13", fg="#9ca3af", text="No preview", height=20)
        self.preview.pack(fill="both", expand=True, pady=(10, 8))
        nav = ttk.Frame(parent, style="Card.TFrame")
        nav.pack(fill="x")
        ttk.Button(nav, text="← Previous", command=lambda: self._move_preview(-1)).pack(side="left")
        self.preview_name = ttk.Label(nav, text="", style="Sub.TLabel")
        self.preview_name.pack(side="left", expand=True)
        ttk.Button(nav, text="Next →", command=lambda: self._move_preview(1)).pack(side="right")
        ttk.Separator(parent).pack(fill="x", pady=12)
        self.status_text = ttk.Label(parent, text="Idle — Uncle Manager awaits orders 🤖", style="Sub.TLabel")
        self.status_text.pack(anchor="w")
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 8))
        self.log = tk.Text(parent, height=9, bg="#0b0e13", fg="#d1d5db", insertbackground="white", relief="flat", wrap="word")
        self.log.pack(fill="x")

    def _config(self) -> TrainerConfig:
        return TrainerConfig(
            landmark_id=self.landmark.get(),
            images_root=str(DEFAULT_IMAGES_ROOT),
            contributions_root=str(DEFAULT_CONTRIBUTIONS_ROOT),
            runs_root=str(DEFAULT_RUNS_ROOT),
            device=self.device.get(),
            epochs=self.epochs.get(),
            batch_size=self.batch_size.get(),
            workers=self.workers.get(),
            early_stopping_patience=self.patience.get(),
            learning_rate=self.learning_rate.get(),
            pretrained=self.pretrained.get(),
            amp=self.amp.get(),
            augmentation=self.augmentation.get(),
            resume_checkpoint=self.resume.get().strip(),
        )

    def _reload_labels(self) -> None:
        try:
            shard = DEFAULT_CONTRIBUTIONS_ROOT / f"{self.landmark.get()}.jsonl"
            self.records = load_records(shard, DEFAULT_IMAGES_ROOT)
            self.preview_index = 0
            self.label_count.configure(text=f"{len(self.records)} placed human labels ready")
            self._draw_preview()
        except Exception as exc:
            self.records = []
            self.label_count.configure(text=f"Dataset error: {exc}")
            self.preview.configure(image="", text=str(exc))

    def _draw_preview(self) -> None:
        if not self.records:
            return
        record = self.records[self.preview_index]
        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        x, y = record.point_xy
        radius = max(4, round(min(image.size) * 0.014))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#f472b6", outline="white", width=2)
        image.thumbnail((620, 390), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")
        self.preview_name.configure(text=f"{self.preview_index + 1}/{len(self.records)}  •  {record.image_name}")

    def _move_preview(self, amount: int) -> None:
        if self.records:
            self.preview_index = (self.preview_index + amount) % len(self.records)
            self._draw_preview()

    def _choose_resume(self) -> None:
        selected = filedialog.askopenfilename(title="Choose discrete checkpoint", filetypes=(("PyTorch checkpoint", "*.pt"),))
        if selected:
            self.resume.set(selected)

    def _start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        try:
            config = self._config()
            config.validate()
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        command = [
            sys.executable, "-m", "packages.side_trainer_discrete.train",
            "--landmark", config.landmark_id,
            "--images-root", config.images_root,
            "--contributions-root", config.contributions_root,
            "--runs-root", config.runs_root,
            "--device", config.device,
            "--epochs", str(config.epochs),
            "--batch-size", str(config.batch_size),
            "--workers", str(config.workers),
            "--patience", str(config.early_stopping_patience),
            "--learning-rate", str(config.learning_rate),
            "--pretrained" if config.pretrained else "--no-pretrained",
            "--amp" if config.amp else "--no-amp",
            "--augmentation" if config.augmentation else "--no-augmentation",
        ]
        if config.resume_checkpoint:
            command.extend(("--resume", config.resume_checkpoint))
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creation_flags,
        )
        threading.Thread(target=self._read_output, daemon=True).start()
        self.active_run = None
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_text.configure(text=f"Starting {config.landmark_id} specialist…")
        self.log.delete("1.0", "end")

    def _read_output(self) -> None:
        if self.process and self.process.stdout:
            for line in self.process.stdout:
                self.output_queue.put(line)

    def _stop(self) -> None:
        if not self.active_run:
            messagebox.showinfo("Still starting", "The run directory is not ready yet.")
            return
        (self.active_run / "STOP").touch()
        self.status_text.configure(text="Stop requested — saving after the active batch…")

    def _poll_status(self) -> None:
        active_path = DEFAULT_RUNS_ROOT / self.landmark.get() / "active_run.json"
        if active_path.is_file() and self.process and self.process.poll() is None:
            try:
                active = json.loads(active_path.read_text(encoding="utf-8"))
                if int(active.get("pid", -1)) != self.process.pid:
                    return
                self.active_run = Path(active["run_dir"])
                status_path = Path(active["status_path"])
                if status_path.is_file():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    epoch = int(status.get("epoch", -1)) + 1
                    epochs = max(1, int(status.get("epochs", self.epochs.get())))
                    self.progress["value"] = 100 * epoch / epochs
                    best = status.get("best_validation_nme", "—")
                    self.status_text.configure(text=f"{status.get('state')} • epoch {epoch}/{epochs} • best NME {best}")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass

    def _poll(self) -> None:
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.log.insert("end", line)
            self.log.see("end")
        self._poll_status()
        if self.process and self.process.poll() is not None:
            code = self.process.returncode
            self.process = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if code == 0:
                self.status_text.configure(text="Complete — zombie pixels became Porion truth 🪢🔪🩸")
            else:
                self.status_text.configure(text=f"Trainer exited with code {code}; see log")
        self.after(250, self._poll)


def main() -> None:
    DiscreteTrainerApp().mainloop()


if __name__ == "__main__":
    main()
