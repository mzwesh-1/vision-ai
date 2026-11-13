"""Builds kaggle/train_on_kaggle.ipynb, a self-contained notebook (scripts embedded)."""
import json, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
read = lambda p: open(os.path.join(root, p)).read()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
    md("# Vision AI: train our own object detector\n"
       "**Before running:** Settings → Accelerator → **GPU T4 x2** (or P100), and Internet → **On**.\n"
       "Add data with **+ Add Input**: search `coco-2017-dataset` (by awsaf49), plus any YOLO-format datasets "
       "(Red Bull cans, phones, tools…). You can also list GitHub dataset repos in the cell below."),
    code("!pip -q install 'transformers>=4.45' pyyaml scipy"),
    code("%%writefile merge_datasets.py\n" + read("tools/merge_datasets.py")),
    code("%%writefile train_detector.py\n" + read("kaggle/train_detector.py")),
    md("## 1. Collect data from Kaggle inputs + GitHub"),
    code("""import glob, os, subprocess

# Datasets hosted on GitHub (YOLO format with a data.yaml). Add repo URLs here.
GITHUB_DATASETS = [
    # "https://github.com/<user>/<yolo-dataset-repo>.git",
]
for url in GITHUB_DATASETS:
    subprocess.run(["git", "clone", "--depth", "1", url, "gh_" + url.rstrip("/").split("/")[-1].removesuffix(".git")])

COCO_IMAGES_PER_SOURCE = 20000   # raise for the full run, lower for a quick test

args = []
for ann in glob.glob("/kaggle/input/**/instances_train2017.json", recursive=True):
    img_dir = glob.glob(os.path.join(os.path.dirname(os.path.dirname(ann)), "**", "train2017"), recursive=True)
    img_dir = [d for d in img_dir if os.path.isdir(d)]
    if img_dir:
        args += ["--coco", f"{ann}:{img_dir[0]}"]
for y in glob.glob("/kaggle/input/**/data.yaml", recursive=True) + glob.glob("gh_*/**/data.yaml", recursive=True):
    args += ["--yolo", os.path.dirname(y)]
print("Sources:", args)
assert args, "No datasets found: add an input dataset or a GitHub repo above"

renames = ["cell phone=phone", "mobile phone=phone", "redbull=red bull can"]
cmd = ["python", "merge_datasets.py", "--out", "merged", "--max-images-per-source", str(COCO_IMAGES_PER_SOURCE)]
cmd += args + sum([["--rename", r] for r in renames], [])
subprocess.run(cmd, check=True)"""),
    md("## 2. Train (≈ 1–2 h per 10k images per epoch on a T4)"),
    code("!python train_detector.py --ann merged/annotations.json --out /kaggle/working/model --epochs 10 --batch 8"),
    md("## 3. Test on a picture"),
    code("""import torch, json, random
from PIL import Image, ImageDraw
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
proc = RTDetrImageProcessor.from_pretrained("/kaggle/working/model")
model = RTDetrForObjectDetection.from_pretrained("/kaggle/working/model").cuda().eval()
im = random.choice(json.load(open("merged/annotations.json"))["images"])
img = Image.open(im["file_name"]).convert("RGB")
with torch.no_grad():
    out = model(**proc(images=img, return_tensors="pt").to("cuda"))
res = proc.post_process_object_detection(out, target_sizes=[img.size[::-1]], threshold=0.5)[0]
d = ImageDraw.Draw(img)
for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
    d.rectangle(b.tolist(), outline="lime", width=3)
    d.text((b[0] + 4, b[1] + 4), f"{model.config.id2label[l.item()]} {s:.0%}", fill="lime")
    print("This is a", model.config.id2label[l.item()], f"({s:.0%})")
img"""),
    md("## 4. Download the model\nZip it, then grab `model.zip` from the **Output** panel and copy the folder to your laptop "
       "for `python webcam_demo.py --model model`."),
    code("!cd /kaggle/working && zip -qr model.zip model && ls -lh model.zip"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(root, "kaggle", "train_on_kaggle.ipynb"), "w"), indent=1)
print("wrote kaggle/train_on_kaggle.ipynb")
