# Vision AI: our own object recognition model

Point a camera at something and it says what it is ("cup", "phone", "knife",
"person wearing black top"). No API keys and no OpenAI: the model runs on your machine.

**Base model:** RT-DETR (Apache-2.0). We're allowed to fine-tune it and **sell** it.
(We avoid Ultralytics YOLO because its AGPL licence would force us to open-source the product.)

## 1. Try it right now on your laptop (no training needed)
The pretrained model already knows 80 everyday objects.
```bash
pip install -r requirements.txt
python webcam_demo.py
```
The first run downloads the model (about 170 MB). Press `q` to quit.

## 2. Train our own version on Kaggle (free GPU)
1. kaggle.com → **Create → New Notebook → File → Import notebook** → upload `kaggle/train_on_kaggle.ipynb`
2. Settings: **Accelerator = GPU**, **Internet = On**
3. **+ Add Input** → add `coco-2017-dataset` and any YOLO-format datasets you want (cans, brands, tools…).
   To pull a dataset straight from GitHub, paste its repo URL into `GITHUB_DATASETS` in the notebook.
4. **Run All**. When it finishes, download `model.zip` from Output and unzip it next to this README.
5. `python webcam_demo.py --model model`

## 3. Add things no public dataset has (Red Bull, your own products)
Take 200–500 photos per item, label them in **Label Studio** or **CVAT** (free), export as
**YOLO** format, upload as a private Kaggle dataset, and re-run the notebook. This is where the
product gets its value: labelled data we own.

## Files
| File | What it does |
|---|---|
| `webcam_demo.py` | Live camera detection and clothing-colour guess |
| `kaggle/train_on_kaggle.ipynb` | One-click Kaggle training notebook (rebuild with `python tools/build_notebook.py`) |
| `kaggle/train_detector.py` | The fine-tuning script |
| `tools/merge_datasets.py` | Combines COCO and YOLO datasets from anywhere into one |
| `data/SOURCES.md` | Dataset list and licences |

## Roadmap
- [x] Phase 1: detector, Kaggle training, webcam demo
- [ ] Phase 2: our own labelled data (brands, office items)
- [ ] Phase 3: proper clothing model (type + colour) to replace the colour heuristic
- [ ] Phase 4: fine-tune an open-weight vision-language model for full scene descriptions
