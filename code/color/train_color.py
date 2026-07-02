# /// script
# dependencies = ["torch", "torchvision", "scikit-learn", "pillow"]
# ///
"""Fine-tune a small color classifier (10 classes) on CompCars sv_data color
labels, decoupled from the make/model/year model.pt: only sv_data has
per-image color ground truth, a narrow single-domain slice of the pool --
not enough to justify a joint multi-task head yet.

Data: DATA/color_vmmr/<color_name>/*.jpg (built by prepare_color_data.py).
Backbone: torchvision resnet18, ImageNet-pretrained, fc replaced with a
10-class head -- resnet18 not resnet50 because 44441 images / 10 classes is
small relative to model.pt's training set, and a smaller backbone overfits
less and trains fast enough for MPS/CPU on this machine.
Class imbalance (black/silver ~13k each, purple/green ~200-250 each) is
handled with inverse-frequency loss weighting, not resampling.
No color-altering augmentation (no ColorJitter/grayscale/hue) -- that would
corrupt the label.

Checkpoint format matches model.pt's own convention: {"state_dict":...,
"classes": [...]}.

Usage:
  uv run code/color/train_color.py --smoke   # fast correctness check (2 classes, 1 epoch)
  uv run code/color/train_color.py           # full run
"""
import argparse
import random
import sys
from collections import defaultdict

sys.path.insert(
    0,
    "/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/.claude/worktrees/vmmr-multi-dataset-prep/code/preprocess",
)
from naming import DATA, MAIN_REPO  # noqa: E402

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

DATA_ROOT = DATA / "color_vmmr"
OUT_CKPT = MAIN_REPO / "models" / "color_model.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ColorDataset(Dataset):
    def __init__(self, samples, classes, train):
        self.samples = samples
        self.classes = classes
        tf = [transforms.Resize((224, 224))]
        if train:
            tf.append(transforms.RandomHorizontalFlip())
        tf += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, label


def collect_samples(root, classes, per_class_cap=None):
    by_class = defaultdict(list)
    for ci, cname in enumerate(classes):
        files = sorted((root / cname).glob("*.*"))
        if per_class_cap:
            files = files[:per_class_cap]
        by_class[ci] = [(str(f), ci) for f in files]
    return by_class


def stratified_split(by_class, val_frac, test_frac, seed):
    train, val, test = [], [], []
    for items in by_class.values():
        if len(items) < 3:
            train += items  # too few to split meaningfully
            continue
        tr, rest = train_test_split(items, test_size=val_frac + test_frac, random_state=seed)
        if rest:
            v, te = train_test_split(
                rest, test_size=test_frac / (val_frac + test_frac), random_state=seed
            )
        else:
            v, te = [], []
        train += tr
        val += v
        test += te
    return train, val, test


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify correctness, not quality")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())
    if not classes:
        print(f"ERROR: no classes found under {DATA_ROOT} -- run prepare_color_data.py first.")
        return

    per_class_cap = 40 if args.smoke else None
    if args.smoke:
        classes = classes[:2]
    by_class = collect_samples(DATA_ROOT, classes, per_class_cap)
    train, val, test = stratified_split(by_class, args.val, args.test, args.seed)
    print(f"classes={len(classes)} train={len(train)} val={len(val)} test={len(test)}")

    train_ds = ColorDataset(train, classes, train=True)
    val_ds = ColorDataset(val, classes, train=False)
    test_ds = ColorDataset(test, classes, train=False)

    bs = min(args.batch_size, max(1, len(train_ds)))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    # inverse-frequency class weights -- black/silver dominate 60x over purple/green
    counts = torch.tensor([max(1, len(by_class[ci])) for ci in range(len(classes))], dtype=torch.float)
    weights = (counts.sum() / counts)
    weights = (weights / weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = 1 if args.smoke else args.epochs
    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, optimizer, device, train=False)
        print(f"epoch {epoch}/{epochs}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if test_dl and len(test_ds):
        model.load_state_dict(best_state)
        te_loss, te_acc = run_epoch(model, test_dl, criterion, optimizer, device, train=False)
        print(f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

    if not args.smoke:
        OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "classes": classes}, OUT_CKPT)
        print(f"wrote {OUT_CKPT} (best_val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
