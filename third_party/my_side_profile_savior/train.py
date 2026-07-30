"""Headless, resumable training CLI for the private profile model factory."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .benchmark import run_benchmark
from .dataset import (
    AugmentationSettings,
    ProfileLandmarkDataset,
    SubjectSplit,
    make_subject_split,
    subject_disjoint_indices,
)
from .factory_config import (
    FactoryConfig,
    atomic_json_save,
    atomic_torch_save,
    create_run_directory,
    resolve_device,
    seed_everything,
)
from .mapping import LandmarkMapping, load_landmark_mapping
from .model import ProfileLandmarkModel, masked_landmark_loss


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_ANNOTATIONS = (
    REPOSITORY_ROOT
    / "git-plz-ignore"
    / "MultiPIE"
    / "MultiPIE_profile_train.txt"
)
DEFAULT_MAPPING = PACKAGE_DIR / "user-custom.txt"
DEFAULT_RUNS = REPOSITORY_ROOT / "git-plz-ignore" / "profile_factory_runs"


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _split_from_dict(values: dict[str, Any]) -> SubjectSplit:
    return SubjectSplit(
        train=tuple(values["train"]),
        validation=tuple(values["validation"]),
        test=tuple(values["test"]),
        seed_text=str(values["seed_text"]),
        seed=int(values["seed"]),
    )


def _augmentation_from_config(config: FactoryConfig) -> AugmentationSettings:
    return AugmentationSettings(
        enabled=config.augmentation,
        rotation_degrees=config.rotation_degrees,
        translation_fraction=config.translation_fraction,
        scale_jitter=config.scale_jitter,
        brightness_jitter=config.brightness_jitter,
        contrast_jitter=config.contrast_jitter,
        blur_probability=config.blur_probability,
    )


def _make_loaders(
    config: FactoryConfig,
    split: SubjectSplit,
    *,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader, ProfileLandmarkDataset]:
    train_dataset = ProfileLandmarkDataset(
        config.annotation_path,
        image_size=config.image_size,
        bbox_scale=config.bbox_scale,
        verify_images=True,
        augmentation=_augmentation_from_config(config),
    )
    evaluation_dataset = ProfileLandmarkDataset(
        config.annotation_path,
        image_size=config.image_size,
        bbox_scale=config.bbox_scale,
        verify_images=True,
    )
    train_indices = subject_disjoint_indices(
        train_dataset.records,
        set(split.train),
    )
    validation_indices = subject_disjoint_indices(
        evaluation_dataset.records,
        set(split.validation),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.workers > 0,
        "worker_init_fn": _worker_seed,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        Subset(evaluation_dataset, validation_indices),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, evaluation_dataset


def _batch_nme(
    predicted: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
    visibility: torch.Tensor,
    crop_xyxy: torch.Tensor,
    bbox_xyxy: torch.Tensor,
) -> tuple[float, int]:
    crop_size = crop_xyxy[:, 2:4] - crop_xyxy[:, 0:2]
    bbox_size = bbox_xyxy[:, 2:4] - bbox_xyxy[:, 0:2]
    denominator = torch.linalg.vector_norm(bbox_size, dim=1).clamp_min(1e-6)
    error_pixels = (predicted - target) * crop_size.unsqueeze(1)
    error = torch.linalg.vector_norm(error_pixels, dim=2) / denominator.unsqueeze(1)
    mask = visibility.bool() & active_mask.bool().view(1, -1)
    return float(error[mask].sum().detach().cpu()), int(mask.sum().item())


def _run_epoch(
    *,
    model: ProfileLandmarkModel,
    loader: DataLoader,
    device: torch.device,
    active_mask: torch.Tensor,
    config: FactoryConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    stop_path: Path,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "map_loss": 0.0, "coordinate_loss": 0.0}
    batch_count = 0
    nme_sum = 0.0
    nme_count = 0
    stop_requested = False

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            landmarks = batch["landmarks"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            visibility = batch["visibility"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            crop_xyxy = batch["crop_xyxy"].to(device)
            bbox_xyxy = batch["bbox_xyxy"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = config.amp and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                losses = masked_landmark_loss(
                    logits,
                    landmarks,
                    active_mask,
                    visibility,
                    sigma=config.gaussian_sigma,
                    map_weight=config.map_loss_weight,
                    coordinate_weight=config.coordinate_loss_weight,
                )
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()

            totals["loss"] += float(losses["total"].detach().cpu())
            totals["map_loss"] += float(losses["map"].detach().cpu())
            totals["coordinate_loss"] += float(
                losses["coordinate"].detach().cpu()
            )
            batch_nme, batch_points = _batch_nme(
                losses["predicted_coordinates"].detach(),
                landmarks,
                active_mask,
                visibility,
                crop_xyxy,
                bbox_xyxy,
            )
            nme_sum += batch_nme
            nme_count += batch_points
            batch_count += 1

            if stop_path.exists():
                stop_requested = True
                break

    denominator = max(1, batch_count)
    return {
        "loss": totals["loss"] / denominator,
        "map_loss": totals["map_loss"] / denominator,
        "coordinate_loss": totals["coordinate_loss"] / denominator,
        "nme": nme_sum / max(1, nme_count),
        "batches": batch_count,
        "points": nme_count,
        "stop_requested": stop_requested,
    }


def _checkpoint_payload(
    *,
    epoch: int,
    model: ProfileLandmarkModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: FactoryConfig,
    mapping: LandmarkMapping,
    split: SubjectSplit,
    best_validation_nme: float,
    epochs_without_improvement: int,
    curves: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config.to_dict(),
        "mapping": mapping.snapshot(),
        "split": split.to_dict(),
        "best_validation_nme": best_validation_nme,
        "epochs_without_improvement": epochs_without_improvement,
        "curves": curves,
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    config: FactoryConfig,
    mapping: LandmarkMapping,
) -> None:
    saved_mapping = checkpoint.get("mapping", {})
    current_mapping = mapping.snapshot()
    if saved_mapping.get("entries") != current_mapping.get("entries"):
        raise ValueError(
            "Current user-custom.txt does not match the checkpoint mapping "
            "snapshot. Restore the original mapping or start a new run."
        )
    old = checkpoint.get("config", {})
    strict_keys = (
        "image_size",
        "heatmap_size",
        "bbox_scale",
        "gaussian_sigma",
        "map_loss_weight",
        "coordinate_loss_weight",
        "learning_rate",
        "weight_decay",
        "augmentation",
        "rotation_degrees",
        "translation_fraction",
        "scale_jitter",
        "brightness_jitter",
        "contrast_jitter",
        "blur_probability",
    )
    for key in strict_keys:
        if old.get(key) != getattr(config, key):
            raise ValueError(
                f"Cannot resume with a different {key}: "
                f"checkpoint={old.get(key)}, requested={getattr(config, key)}"
            )


def run_training(config: FactoryConfig) -> Path:
    """Run or resume one factory experiment and return its run directory."""

    config.validate()
    seed = seed_everything(config.seed_text)
    device = resolve_device(config.device)
    mapping = load_landmark_mapping(config.mapping_path)
    resume_path = (
        Path(config.resume_checkpoint).expanduser().resolve()
        if config.resume_checkpoint
        else None
    )

    checkpoint: dict[str, Any] | None = None
    initial_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        _validate_resume(checkpoint, config, mapping)
        run_dir = resume_path.parent
        split = _split_from_dict(checkpoint["split"])
    else:
        run_dir = create_run_directory(config.runs_root)
        if config.initial_checkpoint:
            initial_path = Path(config.initial_checkpoint).expanduser().resolve()
            initial_checkpoint = torch.load(initial_path, map_location=device)
            if "model_state" not in initial_checkpoint:
                raise ValueError(
                    f"Initial checkpoint has no model_state: {initial_path}"
                )
        split_dataset = ProfileLandmarkDataset(
            config.annotation_path,
            image_size=config.image_size,
            bbox_scale=config.bbox_scale,
            verify_images=True,
        )
        split = make_subject_split(
            split_dataset.records,
            seed_text=config.seed_text,
        )

    stop_path = run_dir / "STOP"
    if stop_path.exists():
        stop_path.unlink()
    status_path = run_dir / "status.json"
    log_path = run_dir / "training.jsonl"
    atomic_json_save(config.to_dict(), run_dir / "config.json")
    atomic_json_save(mapping.snapshot(), run_dir / "mapping.json")
    atomic_json_save(split.to_dict(), run_dir / "split.json")
    atomic_json_save(
        {
            "pid": os.getpid(),
            "run_dir": str(run_dir),
            "status_path": str(status_path),
        },
        Path(config.runs_root).expanduser().resolve() / "active_run.json",
    )
    atomic_json_save(
        {
            "state": "starting",
            "device": str(device),
            "confirmed_count": mapping.confirmed_count,
            "run_dir": str(run_dir),
        },
        status_path,
    )

    train_loader, validation_loader, evaluation_dataset = _make_loaders(
        config,
        split,
        device=device,
        seed=seed,
    )
    model = ProfileLandmarkModel(
        pretrained=(
            config.pretrained
            and checkpoint is None
            and initial_checkpoint is None
        )
    )
    model.to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.amp and device.type == "cuda"
    )
    active_mask = mapping.active_mask().to(device)
    start_epoch = 0
    best_validation_nme = math.inf
    epochs_without_improvement = 0
    curves: list[dict[str, Any]] = []

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_nme = float(checkpoint["best_validation_nme"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        curves = list(checkpoint.get("curves", []))

    stopped = False
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.time()
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            active_mask=active_mask,
            config=config,
            optimizer=optimizer,
            scaler=scaler,
            stop_path=stop_path,
        )
        if train_metrics["stop_requested"]:
            validation_metrics = {
                "loss": math.nan,
                "map_loss": math.nan,
                "coordinate_loss": math.nan,
                "nme": math.inf,
                "batches": 0,
                "points": 0,
                "stop_requested": True,
            }
            stopped = True
        else:
            validation_metrics = _run_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                active_mask=active_mask,
                config=config,
                optimizer=None,
                scaler=scaler,
                stop_path=stop_path,
            )
            stopped = bool(validation_metrics["stop_requested"])

        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        curves.append(epoch_record)
        _append_jsonl(log_path, epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

        improved = (
            not stopped
            and float(validation_metrics["nme"]) < best_validation_nme
        )
        if improved:
            best_validation_nme = float(validation_metrics["nme"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if not stopped:
            scheduler.step()
        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            mapping=mapping,
            split=split,
            best_validation_nme=best_validation_nme,
            epochs_without_improvement=epochs_without_improvement,
            curves=curves,
        )
        atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best.pt")
        atomic_json_save(curves, run_dir / "curves.json")
        atomic_json_save(
            {
                "state": "stopped" if stopped else "training",
                "epoch": epoch,
                "epochs": config.epochs,
                "device": str(device),
                "best_validation_nme": best_validation_nme,
                "epochs_without_improvement": epochs_without_improvement,
                "latest": epoch_record,
                "run_dir": str(run_dir),
            },
            status_path,
        )
        if stopped:
            break
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    state = "stopped" if stopped else "complete"
    benchmark_result = None
    best_path = run_dir / "best.pt"
    if not stopped and config.run_benchmark and best_path.is_file():
        del model, optimizer, scheduler, scaler, train_loader, validation_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        atomic_json_save(
            {
                "state": "benchmarking",
                "device": str(device),
                "best_validation_nme": best_validation_nme,
                "run_dir": str(run_dir),
            },
            status_path,
        )
        benchmark_result = run_benchmark(
            checkpoint_path=best_path,
            dataset=evaluation_dataset,
            split=split,
            mapping=mapping,
            output_path=run_dir / "benchmark.json",
            device=config.device,
        )
    atomic_json_save(
        {
            "state": state,
            "device": str(device),
            "best_validation_nme": best_validation_nme,
            "run_dir": str(run_dir),
            "benchmark": benchmark_result,
        },
        status_path,
    )
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", default="Mini-FaceIQ")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--bbox-scale", type=float, default=1.25)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--map-loss-weight", type=float, default=1.0)
    parser.add_argument("--coordinate-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rotation", type=float, default=8.0)
    parser.add_argument("--translation", type=float, default=0.04)
    parser.add_argument("--scale-jitter", type=float, default=0.08)
    parser.add_argument("--brightness", type=float, default=0.12)
    parser.add_argument("--contrast", type=float, default=0.12)
    parser.add_argument("--blur-probability", type=float, default=0.10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> FactoryConfig:
    if args.config:
        with args.config.expanduser().resolve().open("r", encoding="utf-8") as handle:
            return FactoryConfig.from_dict(json.load(handle))
    return FactoryConfig(
        annotation_path=args.annotations,
        mapping_path=args.mapping,
        runs_root=args.runs_root,
        device=args.device,
        seed_text=args.seed,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        bbox_scale=args.bbox_scale,
        gaussian_sigma=args.gaussian_sigma,
        map_loss_weight=args.map_loss_weight,
        coordinate_loss_weight=args.coordinate_loss_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        pretrained=args.pretrained,
        amp=args.amp,
        augmentation=args.augmentation,
        rotation_degrees=args.rotation,
        translation_fraction=args.translation,
        scale_jitter=args.scale_jitter,
        brightness_jitter=args.brightness,
        contrast_jitter=args.contrast,
        blur_probability=args.blur_probability,
        resume_checkpoint=args.resume,
        initial_checkpoint=args.initial_checkpoint,
        run_benchmark=args.benchmark,
    )


def main() -> None:
    args = _parser().parse_args()
    config = _config_from_args(args)
    try:
        run_dir = run_training(config)
    except Exception as exc:
        active_path = (
            Path(config.runs_root).expanduser().resolve() / "active_run.json"
        )
        if active_path.is_file():
            try:
                with active_path.open("r", encoding="utf-8") as handle:
                    active = json.load(handle)
                if int(active.get("pid", -1)) == os.getpid():
                    atomic_json_save(
                        {
                            "state": "failed",
                            "error": str(exc),
                            "run_dir": active.get("run_dir"),
                        },
                        active["status_path"],
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        print(f"Factory training failed: {exc}", file=sys.stderr)
        raise
    print(f"Factory run saved to: {run_dir}")


if __name__ == "__main__":
    main()
