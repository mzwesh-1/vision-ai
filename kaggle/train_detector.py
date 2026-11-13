"""Fine-tune an RT-DETR object detector (Apache-2.0, OK to sell) on a merged
COCO-format dataset produced by tools/merge_datasets.py.

Run on Kaggle with a GPU:
  python train_detector.py --ann merged/annotations.json --out /kaggle/working/model --epochs 10
"""
import argparse, json, math, os, random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

BASE = "PekingU/rtdetr_r50vd_coco_o365"  # pretrained on COCO + Objects365


class CocoDet(Dataset):
    def __init__(self, images, by_img, processor, train):
        self.images, self.by_img, self.proc, self.train = images, by_img, processor, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = Image.open(im["file_name"]).convert("RGB")
        anns = [dict(a) for a in self.by_img.get(im["id"], [])]
        if self.train and random.random() < 0.5:  # horizontal flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for a in anns:
                x, y, w, h = a["bbox"]
                a["bbox"] = [img.width - x - w, y, w, h]
        enc = self.proc(images=img, annotations={"image_id": im["id"], "annotations": anns}, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0]


def collate(batch):
    return {"pixel_values": torch.stack([b[0] for b in batch]), "labels": [b[1] for b in batch]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", default="model")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for smoke tests)")
    args = ap.parse_args()

    d = json.load(open(args.ann))
    id2label = {c["id"]: c["name"] for c in d["categories"]}
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images = [im for im in d["images"] if im["id"] in by_img]
    random.seed(0); random.shuffle(images)
    n_val = max(1, int(len(images) * args.val_frac))
    val_imgs, train_imgs = images[:n_val], images[n_val:]
    print(f"{len(train_imgs)} train / {len(val_imgs)} val images, {len(id2label)} classes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = RTDetrImageProcessor.from_pretrained(args.base)
    model = RTDetrForObjectDetection.from_pretrained(
        args.base, id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True).to(device)

    mk = lambda ims, tr: DataLoader(CocoDet(ims, by_img, proc, tr), batch_size=args.batch, shuffle=tr,
                                    num_workers=args.workers, collate_fn=collate)
    train_dl, val_dl = mk(train_imgs, True), mk(val_imgs, False)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
                            weight_decay=1e-4)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * s / total)))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best = 0, float("inf")
    for ep in range(args.epochs):
        model.train()
        for b in train_dl:
            labels = [{k: v.to(device) for k, v in l.items()} for l in b["labels"]]
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = model(pixel_values=b["pixel_values"].to(device), labels=labels).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"epoch {ep} step {step}/{total} loss {loss.item():.3f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for b in val_dl:
                labels = [{k: v.to(device) for k, v in l.items()} for l in b["labels"]]
                vl += model(pixel_values=b["pixel_values"].to(device), labels=labels).loss.item(); n += 1
        vl /= max(n, 1)
        print(f"== epoch {ep} val loss {vl:.3f}")
        if vl < best:
            best = vl
            model.save_pretrained(args.out); proc.save_pretrained(args.out)
            print(f"   saved best model -> {args.out}")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
