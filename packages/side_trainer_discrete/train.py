"""Train one independent side-profile landmark model (Porion by default)."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from .config import (
    DEFAULT_CONTRIBUTIONS_ROOT,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_RUNS_ROOT,
    LANDMARK_IDS,
    TrainerConfig,
    atomic_json_save,
    atomic_torch_save,
    create_run_directory,
    resolve_device,
    seed_everything,
)
from .dataset import (
    AugmentationSettings,
    DiscreteLandmarkDataset,
    SubjectSplit,
    indices_for_subjects,
    make_subject_split,
)
from .model import DiscreteLandmarkModel, landmark_loss


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _split_from_dict(value: dict[str, Any]) -> SubjectSplit:
    return SubjectSplit(
        train=tuple(value["train"]),
        validation=tuple(value["validation"]),
        test=tuple(value["test"]),
        seed_text=str(value["seed_text"]),
        seed=int(value["seed"]),
    )


def _augmentation(config: TrainerConfig, enabled: bool) -> AugmentationSettings:
    return AugmentationSettings(
        enabled=enabled and config.augmentation,
        rotation_degrees=config.rotation_degrees,
        translation_fraction=config.translation_fraction,
        scale_jitter=config.scale_jitter,
        brightness_jitter=config.brightness_jitter,
        contrast_jitter=config.contrast_jitter,
        blur_probability=config.blur_probability,
    )


def _loader(dataset: Subset, config: TrainerConfig, device: torch.device, shuffle: bool) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.workers > 0,
    }
    if config.workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def _make_loaders(
    config: TrainerConfig,
    split: SubjectSplit,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int]:
    train_dataset = DiscreteLandmarkDataset(
        config.shard_path,
        config.images_root,
        image_size=config.image_size,
        augmentation=_augmentation(config, True),
    )
    evaluation_dataset = DiscreteLandmarkDataset(
        config.shard_path,
        config.images_root,
        image_size=config.image_size,
        augmentation=_augmentation(config, False),
    )
    train_indices = indices_for_subjects(train_dataset.records, split.train)
    validation_indices = indices_for_subjects(evaluation_dataset.records, split.validation)
    if not train_indices or not validation_indices:
        raise ValueError("Subject split produced an empty train or validation set")
    return (
        _loader(Subset(train_dataset, train_indices), config, device, True),
        _loader(Subset(evaluation_dataset, validation_indices), config, device, False),
        len(evaluation_dataset),
    )


def _run_epoch(
    model: DiscreteLandmarkModel,
    loader: DataLoader,
    config: TrainerConfig,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer | None,
    stop_path: Path,
) -> dict[str, float | int | bool]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "map_loss": 0.0, "coordinate_loss": 0.0, "nme": 0.0}
    examples = 0
    batches = 0
    stop_requested = False
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            targets = batch["target"].to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = config.amp and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = landmark_loss(
                    model(images),
                    targets,
                    sigma=config.gaussian_sigma,
                    map_weight=config.map_loss_weight,
                    coordinate_weight=config.coordinate_loss_weight,
                )
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = images.shape[0]
            normalized_errors = torch.linalg.vector_norm(
                losses["predicted"].detach() - targets,
                dim=-1,
            ) / math.sqrt(2.0)
            totals["loss"] += float(losses["total"].detach()) * batch_size
            totals["map_loss"] += float(losses["map"].detach()) * batch_size
            totals["coordinate_loss"] += float(losses["coordinate"].detach()) * batch_size
            totals["nme"] += float(normalized_errors.sum().detach())
            examples += batch_size
            batches += 1
            if stop_path.exists():
                stop_requested = True
                break
    denominator = max(1, examples)
    return {
        "loss": totals["loss"] / denominator,
        "map_loss": totals["map_loss"] / denominator,
        "coordinate_loss": totals["coordinate_loss"] / denominator,
        "nme": totals["nme"] / denominator,
        "examples": examples,
        "batches": batches,
        "stop_requested": stop_requested,
    }


def _validate_resume(checkpoint: dict[str, Any], config: TrainerConfig) -> None:
    if checkpoint.get("landmark_id") != config.landmark_id:
        raise ValueError("A checkpoint can only resume its own landmark specialist")
    saved = checkpoint.get("config", {})
    strict = (
        "image_size", "heatmap_size", "gaussian_sigma", "map_loss_weight",
        "coordinate_loss_weight", "learning_rate", "weight_decay",
    )
    for key in strict:
        if saved.get(key) != getattr(config, key):
            raise ValueError(f"Cannot resume with a different {key}")


def run_training(config: TrainerConfig) -> Path:
    config.validate()
    seed_everything(config.seed_text)
    device = resolve_device(config.device)
    resume_path = Path(config.resume_checkpoint).expanduser().resolve() if config.resume_checkpoint else None
    checkpoint: dict[str, Any] | None = None
    probe = DiscreteLandmarkDataset(config.shard_path, config.images_root, image_size=config.image_size)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        _validate_resume(checkpoint, config)
        split = _split_from_dict(checkpoint["split"])
        run_dir = resume_path.parent
    else:
        split = make_subject_split(probe.records, config.seed_text)
        run_dir = create_run_directory(config.landmark_runs_root)
    stop_path = run_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    status_path = run_dir / "status.json"
    active_path = config.landmark_runs_root / "active_run.json"
    atomic_json_save(config.to_dict(), run_dir / "config.json")
    atomic_json_save(split.to_dict(), run_dir / "split.json")
    atomic_json_save({"pid": os.getpid(), "run_dir": str(run_dir), "status_path": str(status_path)}, active_path)
    atomic_json_save(
        {"state": "starting", "landmark": config.landmark_id, "device": str(device), "labels": len(probe)},
        status_path,
    )
    train_loader, validation_loader, label_count = _make_loaders(config, split, device)
    model = DiscreteLandmarkModel(pretrained=config.pretrained and checkpoint is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    start_epoch = 0
    best_nme = math.inf
    stale_epochs = 0
    curves: list[dict[str, Any]] = []
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_nme = float(checkpoint["best_validation_nme"])
        stale_epochs = int(checkpoint.get("epochs_without_improvement", 0))
        curves = list(checkpoint.get("curves", []))
    stopped = False
    for epoch in range(start_epoch, config.epochs):
        started = time.time()
        train_metrics = _run_epoch(model, train_loader, config, device, scaler, optimizer, stop_path)
        if train_metrics["stop_requested"]:
            validation_metrics: dict[str, Any] = {"nme": math.inf, "stop_requested": True, "examples": 0}
            stopped = True
        else:
            validation_metrics = _run_epoch(model, validation_loader, config, device, scaler, None, stop_path)
            stopped = bool(validation_metrics["stop_requested"])
        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        curves.append(record)
        _append_jsonl(run_dir / "training.jsonl", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        improved = not stopped and float(validation_metrics["nme"]) < best_nme
        if improved:
            best_nme = float(validation_metrics["nme"])
            stale_epochs = 0
        else:
            stale_epochs += 1
        if not stopped:
            scheduler.step()
        payload = {
            "format": "mini-faceiq-discrete-v1",
            "landmark_id": config.landmark_id,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": config.to_dict(),
            "split": split.to_dict(),
            "label_count": label_count,
            "best_validation_nme": best_nme,
            "epochs_without_improvement": stale_epochs,
            "curves": curves,
        }
        atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best.pt")
        atomic_json_save(curves, run_dir / "curves.json")
        atomic_json_save(
            {
                "state": "stopped" if stopped else "training",
                "landmark": config.landmark_id,
                "epoch": epoch,
                "epochs": config.epochs,
                "device": str(device),
                "labels": label_count,
                "best_validation_nme": best_nme,
                "latest": record,
                "run_dir": str(run_dir),
            },
            status_path,
        )
        if stopped or stale_epochs >= config.early_stopping_patience:
            break
    atomic_json_save(
        {
            "state": "stopped" if stopped else "complete",
            "landmark": config.landmark_id,
            "device": str(device),
            "labels": label_count,
            "best_validation_nme": best_nme,
            "run_dir": str(run_dir),
        },
        status_path,
    )
    del model, optimizer, scheduler, scaler, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--landmark", choices=LANDMARK_IDS, default="porion")
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--contributions-root", default=str(DEFAULT_CONTRIBUTIONS_ROOT))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", default="Mini-FaceIQ-discrete")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainerConfig:
    if args.config:
        return TrainerConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    return TrainerConfig(
        landmark_id=args.landmark,
        images_root=args.images_root,
        contributions_root=args.contributions_root,
        runs_root=args.runs_root,
        device=args.device,
        seed_text=args.seed,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        learning_rate=args.learning_rate,
        resume_checkpoint=args.resume,
        pretrained=args.pretrained,
        amp=args.amp,
        augmentation=args.augmentation,
    )


def main() -> None:
    config = _config_from_args(_parser().parse_args())
    try:
        run_dir = run_training(config)
    except Exception as exc:
        active_path = config.landmark_runs_root / "active_run.json"
        if active_path.is_file():
            try:
                active = json.loads(active_path.read_text(encoding="utf-8"))
                if int(active.get("pid", -1)) == os.getpid():
                    atomic_json_save(
                        {"state": "failed", "error": str(exc), "run_dir": active.get("run_dir")},
                        Path(active["status_path"]),
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        print(f"Discrete trainer failed: {exc}", file=sys.stderr)
        raise
    print(f"Discrete run saved to: {run_dir}")


if __name__ == "__main__":
    main()
