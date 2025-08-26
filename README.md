# 🧠 Vision Transformer on CIFAR-10 (with Attention Maps)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sagarramteke19/vision-transformer-cifar10/blob/main/notebooks/vision_transformer_cifar10.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A clean, reproducible implementation of a **Vision Transformer (ViT)** on **CIFAR-10**, with utilities to:
- Train & evaluate ViT (optionally compare vs CNN baseline)
- **Visualize attention** maps / class tokens
- Log metrics (accuracy, loss) and save confusion matrix
- Export a minimal **Streamlit** demo (optional)

---

## 📂 Project Structure
```
vision-transformer-cifar10/
├─ notebooks/
│  └─ vision_transformer_cifar10.ipynb   # your notebook goes here
├─ src/
│  ├─ train.py                           # CLI training script (torch)
│  ├─ datasets.py                        # CIFAR-10 dataloaders & transforms
│  ├─ visualize_attention.py             # attention rollout / heatmap helpers
│  └─ models/
│     ├─ vit_timm.py                     # ViT via timm (simple)
│     └─ cnn_baseline.py                 # (optional) small CNN for baseline
├─ data/                                 # (ignored) CIFAR-10 cache
├─ results/                              # saved weights, metrics, plots
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## 🚀 Quickstart

```bash
# 1) Install
pip install -r requirements.txt

# 2) Train ViT (timm-backed)
python -m src.train --model vit_tiny_patch16_224 --epochs 20 --batch-size 128

# 3) Evaluate a checkpoint
python -m src.train --eval --ckpt results/best_vit.pt

# 4) (Optional) Save attention maps for a few samples
python -m src.visualize_attention --ckpt results/best_vit.pt --num-samples 8
```

> Tip: Add `--use-cuda` if you have a GPU. Use `--model resnet18` to compare with a CNN baseline.

---

## 🧪 Notebook (Colab)
Open the notebook directly in Colab with the badge above. It includes:
- Dataset EDA (sample grid, class distribution)
- Training loop with early stopping
- Accuracy, confusion matrix, and per-class metrics
- Attention maps overlay

---

## 📈 Results (placeholders)
| Model | Top-1 Acc. | Notes |
|------|-------------|-------|
| ViT (tiny, patch16, 224) |  97.05% | After 20 epochs, RandAugment + CutMix |
| ResNet-18 (baseline) | 95% | Strong CNN baseline for comparison |

Add your plots to **`results/`** and embed them here:

![Accuracy Curve](./results/model1.png)

![Confusion Matrix](./results/confusionmatricbasline.png)
```

---

## 📦 Dataset
Uses torchvision **CIFAR-10** (automatically downloaded on first run). If you use a custom dataset, update `datasets.py` accordingly.

---

## 📜 License
MIT


---

## 🖥️ Streamlit Demo
Run an interactive demo locally or deploy to Streamlit Cloud.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> Place your trained weights at `results/best_vit.pt` to get meaningful predictions.

