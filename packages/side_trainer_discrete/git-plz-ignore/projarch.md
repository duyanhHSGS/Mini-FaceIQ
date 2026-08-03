# Discrete Side Trainer Architecture

> Compile time can take 1000 years. Fast runtime is the crown. 🦅⚡

## Contract

One canonical landmark ID owns one JSONL input shard, one one-channel network,
and one isolated checkpoint tree. The first specialist is `porion`.

```text
human image + porion.jsonl
    -> strict placed-label adapter
    -> subject-disjoint split
    -> MobileNetV3-Small
    -> one 64x64 heatmap
    -> soft-argmax coordinate
    -> porion/best.pt
```

## Module jobs

- `config.py`: canonical 31-ID list, defaults, devices, seeding, atomic saves.
- `dataset.py`: shard validation, image join, subject split, safe augmentation.
- `model.py`: single-output MobileNetV3-Small decoder and hybrid loss.
- `train.py`: loaders, AMP loop, resume guards, early stop, run artifacts.
- `inference.py`: checkpoint validation and original-pixel predictions.
- `ui.py`: Porion-first Tkinter training control room and truth preview.

## Data safety

- `placed` is a coordinate target.
- `unavailable` is respected and excluded.
- A missing row is unreviewed and excluded.
- Subjects cannot cross train, validation, and frozen test partitions.
- Checkpoints embed their landmark ID and cannot resume another specialist.
- Human-factory source shards and images are read-only inputs.

## Runtime

The network allocates one output channel, DataLoader workers prefetch into
pinned CUDA-ready batches, AMP is CUDA-only, and CuDNN selects fast kernels.
Choosing a specialist is an O(1) ID/path lookup.

## Human transition 🪢🔪🩸

The 31-headed zombie is separated into tiny expert brains. Each model learns
one anatomical location without gradients from unrelated landmarks. Porion
graduates first; the same fixed contract can hatch the other 30.
