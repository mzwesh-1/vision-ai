"""Merge many detection datasets (COCO-JSON and/or YOLO-txt format) into ONE
COCO-format dataset with a unified, contiguous label list.

Classes are matched by name (case-insensitive), so "Cup" from one dataset and
"cup" from another become the same class. Use --rename to fix synonyms,
e.g. --rename "cell phone=phone" --rename "mobile=phone".

Examples:
  python tools/merge_datasets.py --out merged \
      --coco /data/coco/annotations/instances_val2017.json:/data/coco/val2017 \
      --yolo /data/redbull_yolo            # folder containing data.yaml
"""
import argparse, json, os, glob
from PIL import Image
import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_coco(ann_path, img_dir):
    d = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img = {}
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        by_img.setdefault(a["image_id"], []).append((cats[a["category_id"]], a["bbox"]))
    for im in d["images"]:
        yield os.path.join(img_dir, im["file_name"]), im["width"], im["height"], by_img.get(im["id"], [])


def load_yolo(root):
    cfg_path = next(iter(glob.glob(os.path.join(root, "*.yaml"))), None)
    if cfg_path is None:
        raise SystemExit(f"No data.yaml found in {root}")
    cfg = yaml.safe_load(open(cfg_path))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for img_path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not img_path.lower().endswith(IMG_EXT):
            continue
        # YOLO convention: .../images/x.jpg  ->  .../labels/x.txt
        lbl = os.path.splitext(img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))[0] + ".txt"
        if not os.path.exists(lbl):
            continue
        w, h = Image.open(img_path).size
        objs = []
        for line in open(lbl):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            objs.append((names[c], [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]))
        yield img_path, w, h, objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder (writes annotations.json)")
    ap.add_argument("--coco", action="append", default=[], help="ANN_JSON:IMAGE_DIR (repeatable)")
    ap.add_argument("--yolo", action="append", default=[], help="YOLO dataset root (repeatable)")
    ap.add_argument("--rename", action="append", default=[], help="'old=new' class rename (repeatable)")
    ap.add_argument("--max-images-per-source", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rename = {k.strip().lower(): v.strip().lower() for k, v in (r.split("=", 1) for r in args.rename)}
    sources = [load_coco(*s.rsplit(":", 1)) for s in args.coco] + [load_yolo(r) for r in args.yolo]

    label2id, images, anns = {}, [], []
    for src in sources:
        for n, (path, w, h, objs) in enumerate(src):
            if args.max_images_per_source and n >= args.max_images_per_source:
                break
            img_id = len(images)
            images.append({"id": img_id, "file_name": os.path.abspath(path), "width": w, "height": h})
            for name, (x, y, bw, bh) in objs:
                name = rename.get(name.lower(), name.lower())
                if bw <= 1 or bh <= 1:
                    continue
                cid = label2id.setdefault(name, len(label2id))
                anns.append({"id": len(anns), "image_id": img_id, "category_id": cid,
                             "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})

    os.makedirs(args.out, exist_ok=True)
    out = {"images": images, "annotations": anns,
           "categories": [{"id": i, "name": n} for n, i in label2id.items()]}
    json.dump(out, open(os.path.join(args.out, "annotations.json"), "w"))
    print(f"Merged {len(images)} images, {len(anns)} boxes, {len(label2id)} classes -> {args.out}/annotations.json")


if __name__ == "__main__":
    main()
