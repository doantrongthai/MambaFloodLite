# MambaFloodLite: A Low-Parameter Mamba-Based Network for Flood Area Segmentation

Training pipeline for flood area segmentation with strict reproducibility using PyTorch.

---

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Commands](#training-commands)
- [Multi-Seed Experiments](#multi-seed-experiments)

---

## Dataset

### Flood Area Segmentation (FloodKaggle)
- **Task**: Binary flood segmentation (flooded vs. non-flooded pixels)
- **Total images**: 290 RGB aerial images
- **Annotation**: Pixel-wise binary masks (Label Studio)
- **Split**: 70% train / 15% val / 15% test
- **Image sources**: UAV and helicopter imagery across urban, peri-urban, and rural environments
- **Auto-download**: Yes
- **Structure**:
  ```
  floodkaggle/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  └── test/
  ```

---

## Installation

```bash
git clone https://github.com/doantrongthai/MambaFloodLite
cd MambaFloodLite

pip install torch torchvision albumentations opencv-python tqdm gdown numpy pillow
```

---

## Quick Start

```bash
# Download dataset
python main.py --download

# Train MambaFloodLite
python main.py \
  --model mambaunet \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

---

## Training Commands

### Basic Training

```bash
python main.py --model mambaunet --epochs 50
```

### All Options

```bash
python main.py \
  --model mambaunet \
  --size 256 \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42 \
  --output_path outputs/exp1
```

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Model name (e.g. `mambaunet`, `unet`, `segformer`) |
| `--size` | `256` | Input image size |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `0.001` | Learning rate (Adam optimizer) |
| `--seed` | `42` | Random seed |
| `--output_path` | `outputs` | Output directory |
| `--download` | flag | Download dataset |
| `--multiseed` | flag | Run multi-seed experiment |
| `--seeds` | `42 123 456 789 2024` | Seeds for multi-seed experiment |

---

## Multi-Seed Experiments

Each experiment is repeated across 5 independent random seeds to ensure statistical reliability, consistent with the paper's evaluation protocol.

```bash
# Default seeds: 42, 123, 456, 789, 2024
python main.py --model mambaunet --multiseed --epochs 50

# Custom seeds
python main.py --model mambaunet --multiseed --seeds 42 100 200 300 400 --epochs 50
```

**Output:**
```
STATISTICS FOR PAPER
======================================================================
Test Loss:    0.1234 +/- 0.0056
IoU:          0.7398 +/- 0.0079
Dice Score:   0.8504 +/- 0.0052
Val Loss:     0.1456 +/- 0.0078
Parameters:   285,465
======================================================================

LaTeX format:
Test Loss: $0.1234 \pm 0.0056$
IoU:       $0.7398 \pm 0.0079$
Dice:      $0.8504 \pm 0.0052$
```

Results saved to: `outputs/{model}_floodkaggle_multiseed.json`

---

## Supported Models

| Model | Parameters | IoU | DSC |
|---|---|---|---|
| MambaFloodLite (Ours) | **285,465** | **0.7398 ± 0.0079** | **0.8504 ± 0.0052** |
| ENet | 365,950 | 0.7233 ± 0.0148 | 0.8394 ± 0.0100 |
| EDANet | 678,580 | 0.7238 ± 0.0050 | 0.8398 ± 0.0034 |
| U-Lite | 878,417 | 0.7365 ± 0.0060 | 0.8482 ± 0.0040 |
| ERFNet | 2,063,021 | 0.7123 ± 0.0155 | 0.8319 ± 0.0107 |
| STDC1 | 7,786,144 | 0.6988 ± 0.0305 | 0.8223 ± 0.0215 |
| SegFormer | 24,722,369 | 0.7206 ± 0.0109 | 0.8376 ± 0.0074 |
| SegNet | 29,480,129 | 0.6466 ± 0.0282 | 0.7850 ± 0.0213 |
| U-Net | 31,043,521 | 0.6778 ± 0.0115 | 0.8079 ± 0.0082 |
| Attention U-Net | 34,878,573 | 0.6805 ± 0.0208 | 0.8097 ± 0.0148 |

---

## Reproducibility

- Fixed seeds for Python, NumPy, PyTorch, CUDA
- Deterministic CUDA operations (`torch.use_deterministic_algorithms(True)`)
- Disabled cuDNN benchmark and TF32
- Per-sample deterministic augmentation
- Fixed DataLoader worker seeds
- RNG states saved in checkpoints

---

## Troubleshooting

**CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
Reduce batch size: `--batch_size 2`

**Dataset not found**
```
FileNotFoundError: floodkaggle not found
```
Run with `--download` flag first.