# Where the training data comes from

Most big image datasets are too large to keep *inside* a GitHub repo (GitHub caps
files at 100 MB), so GitHub "dataset" repos usually hold the labels and a download
script, with the images on Kaggle, Hugging Face or a university server. Our Kaggle
notebook can pull from all of them directly; nothing goes through your laptop.

Check the licence before using anything in the version you SELL.

| Dataset | What it has | Where | Licence / commercial use |
|---|---|---|---|
| COCO 2017 | 80 everyday objects (cup, phone, knife, bottle, laptop, person…) | Kaggle: `awsaf49/coco-2017-dataset` · github.com/cocodataset | Labels CC-BY 4.0; images are Flickr photos under mixed licences, so check with a lawyer |
| Open Images V7 | 600 classes, 16M boxes | github.com/openimages/dataset · Kaggle mirrors | Labels CC-BY 4.0; images "CC-BY 2.0 as listed", so check per image |
| Objects365 | 365 classes | objects365.org | Research-only; already baked into the pretrained base model |
| DeepFashion2 | Clothing type and position | github.com/switchablenorms/DeepFashion2 | **Non-commercial**, prototype use only |
| Roboflow Universe | Thousands of small YOLO-format sets (brands, cans, tools) | universe.roboflow.com, many mirrored on GitHub/Kaggle | Varies per dataset (many are CC-BY) |
| **Our own photos** | Red Bull, your products, your office | Label with Label Studio or CVAT (free, open source) | **Ours, 100% sellable** |

The merge tool (`tools/merge_datasets.py`) takes COCO-JSON and YOLO-txt datasets,
matches classes by name, and writes one combined dataset.
