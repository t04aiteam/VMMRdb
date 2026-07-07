"""Finetune a pretrained ResNet-50 on VMMRdb (ImageFolder layout: data/<class>/*.jpg).

Usage: uv run train.py --data data --epochs 10
Writes model.pt (weights + class names) next to this script.
"""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), NORM,
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), NORM,
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="model.pt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", default=None, help="checkpoint to continue from (e.g. model.pt)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision (diagnostic: rule out GradScaler collapse)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda" and not a.no_amp
    full = datasets.ImageFolder(a.data, allow_empty=True)  # tolerate empty class dirs (e.g. bmw_528_2013)
    n_val = max(1, int(0.1 * len(full)))
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_ds.dataset.transform = TRAIN_TF
    # val_ds shares the same underlying dataset object, so wrap eval transform separately:
    val_view = datasets.ImageFolder(a.data, transform=EVAL_TF, allow_empty=True)
    val_ds = torch.utils.data.Subset(val_view, val_ds.indices)

    classes = full.classes
    print(f"{len(full)} imgs, {len(classes)} classes, dev={dev}")

    dl_args = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, **dl_args)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_args)

    model = models.resnet50(weights=None if a.resume else models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=True)
        sd = ck["state_dict"]
        if ck["classes"] != classes:
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}  # head dim differs -> reinit it
            print(f"class set changed ({len(ck['classes'])}->{len(classes)}); resuming backbone only, fresh head")
        model.load_state_dict(sd, strict=False)
        print(f"resumed from {a.resume}")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(a.epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if i % 50 == 0:
                print(f"ep{ep} it{i}/{len(train_dl)} loss {loss.item():.3f}")
        # val
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                correct += (model(x).argmax(1) == y).sum().item(); tot += y.numel()
        print(f"== epoch {ep}: val acc {correct/tot:.3f}")

    sd = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
    torch.save({"state_dict": sd, "classes": classes}, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(classes))
    print(f"saved {a.out} ({len(classes)} classes)")


if __name__ == "__main__":
    main()
