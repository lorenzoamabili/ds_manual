# 11 · Computer Vision

Understanding images and video. Cross-domain: medical imaging, industrial
inspection, autonomous systems, document processing.

## The task taxonomy
| Task | Output | Example |
|------|--------|---------|
| **Classification** | One label per image | Defect / no-defect |
| **Object detection** | Boxes + labels | Count items on a shelf |
| **Semantic segmentation** | Class per pixel | Tumour region in a scan |
| **Instance segmentation** | Per-object masks | Separate touching cells |
| **Keypoint / pose** | Landmark coordinates | Body-pose estimation |
| **OCR** | Text from image | Invoice digitisation |
| **Generation / restoration** | New/cleaned images | Super-resolution, synthesis |

## Models
- **CNNs** (ResNet, EfficientNet) — the reliable backbone for classification and
  feature extraction. Still an excellent default.
- **Detection/segmentation architectures** — YOLO (fast, real-time detection),
  Faster R-CNN (accuracy), U-Net (segmentation, dominant in medical), Mask R-CNN
  (instance segmentation), SAM (promptable segmentation).
- **Vision Transformers (ViT)** — competitive-to-better than CNNs given enough
  data; the trajectory of the field.

## The practices that actually decide success

- **Transfer learning is the default, not the exception.** Almost never train from
  scratch. Start from an ImageNet- (or domain-) pretrained backbone and fine-tune.
  It slashes the data and compute you need and usually wins.
- **Data augmentation** — flips, crops, colour jitter, rotation (only those that
  preserve the label — don't flip a "6" into a "9", don't mirror an X-ray if
  laterality matters). This is often the highest-leverage lever.
- **Data quality > model choice.** Label noise, class imbalance, and distribution
  shift between your training images and deployment cameras sink more CV projects
  than architecture ever does.
- **Watch for shortcut learning.** The classic failure: the model learns the
  hospital's scanner watermark or the ruler in dermatology photos instead of the
  pathology. Always inspect what the model attends to (Grad-CAM) and validate on
  data from a *different* source/site.
- **Evaluation** — accuracy for balanced classification; **mAP** (mean average
  precision) for detection; **IoU / Dice** for segmentation. Split so the same
  patient/scene/site never spans train and test (group leakage is rampant in CV).

## Practical stack
`torchvision` / `timm` for backbones, `ultralytics` for YOLO, `segmentation-models`
and `MONAI` (medical) for segmentation, `albumentations` for augmentation. Label
with CVAT/Label Studio. For most business problems, a fine-tuned pretrained model
+ good augmentation + clean labels beats a novel architecture.

---

## Python example — image classification with transfer learning (torchvision)

```python
"""
Transfer learning demo: fine-tune a ResNet-18 backbone on a small dataset.
Uses torchvision's built-in datasets (no download needed for CIFAR-10).
Demonstrates: pretrained backbone, freeze → fine-tune schedule, data augmentation.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Data: CIFAR-10 (auto-downloads ~170MB, then cached) ──────────────────────
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])

try:
    train_ds = torchvision.datasets.CIFAR10(".", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(".", train=False, download=True, transform=test_tf)
except Exception as e:
    print(f"Download failed: {e}")
    raise

# Subset for fast demo: 2000 train, 400 test
rng = np.random.default_rng(42)
train_idx = rng.choice(len(train_ds), 2000, replace=False)
test_idx  = rng.choice(len(test_ds),  400,  replace=False)
train_dl = DataLoader(Subset(train_ds, train_idx), batch_size=64, shuffle=True)
test_dl  = DataLoader(Subset(test_ds,  test_idx),  batch_size=64)

# ── Model: ResNet-18 pretrained, replace head for 10 classes ─────────────────
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# Freeze backbone, train only the final layer first
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 10)  # new head always trainable
model = model.to(device)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── Phase 1: train head only (5 epochs) ──────────────────────────────────────
print("\nPhase 1: head only")
for epoch in range(5):
    model.train()
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_dl:
            preds = model(X.to(device)).argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total   += len(y)
    print(f"  Epoch {epoch+1}/5 — test acc: {correct/total:.2%}")

# ── Phase 2: unfreeze and fine-tune full network (2 more epochs) ─────────────
print("\nPhase 2: full fine-tune")
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

for epoch in range(2):
    model.train()
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_dl:
            preds = model(X.to(device)).argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total   += len(y)
    print(f"  Epoch {epoch+1}/2 — test acc: {correct/total:.2%}")

print("\nLesson: freeze → fine-tune schedule prevents overwriting pretrained")
print("features before the new head learns a sensible gradient direction.")
```

---

## Cross-references

- [34](34-manufacturing.md) — visual quality control in manufacturing
- [33](33-healthtech.md) — medical imaging (X-ray, pathology)
- [03](03-data-and-feature-engineering.md) — data augmentation is feature engineering
