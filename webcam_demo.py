"""Live camera: boxes and names every object it sees, and guesses the colour
of your top ("person wearing black top").

  python webcam_demo.py                  # pretrained model, 80 everyday objects
  python webcam_demo.py --model model    # your own fine-tuned model from Kaggle
  python webcam_demo.py --image photo.jpg --save out.jpg   # single image
"""
import argparse
import cv2, numpy as np, torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

COLOURS = {  # name: BGR reference
    "black": (30, 30, 30), "white": (235, 235, 235), "grey": (128, 128, 128),
    "red": (40, 40, 200), "orange": (30, 130, 240), "yellow": (40, 220, 230),
    "green": (60, 160, 60), "blue": (180, 90, 30), "navy": (90, 40, 20),
    "purple": (140, 50, 120), "pink": (180, 130, 240), "brown": (40, 70, 120),
}


def colour_name(patch):
    """Rough heuristic: median colour of the torso -> nearest named colour."""
    if patch.size == 0:
        return None
    med = np.median(patch.reshape(-1, 3), axis=0)
    lab = lambda c: cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    m = lab(med)
    return min(COLOURS, key=lambda n: np.linalg.norm(lab(COLOURS[n]) - m))


def detect(model, proc, frame_bgr, thr, device):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_object_detection(out, target_sizes=[img.size[::-1]], threshold=thr)[0]
    dets = []
    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        name = model.config.id2label[l.item()]
        text = f"{name} {s:.0%}"
        if name == "person":  # torso = middle band of the person box
            h, w = y2 - y1, x2 - x1
            c = colour_name(frame_bgr[max(0, y1 + int(.25 * h)):y1 + int(.55 * h),
                                      max(0, x1 + int(.3 * w)):x1 + int(.7 * w)])
            if c:
                text = f"person wearing {c} top {s:.0%}"
        dets.append((x1, y1, x2, y2, text))
    return dets


def draw(frame, dets):
    for x1, y1, x2, y2, text in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="PekingU/rtdetr_r50vd_coco_o365")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image")
    ap.add_argument("--save")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = RTDetrImageProcessor.from_pretrained(a.model)
    model = RTDetrForObjectDetection.from_pretrained(a.model).to(device).eval()

    if a.image:
        frame = cv2.imread(a.image)
        dets = detect(model, proc, frame, a.threshold, device)
        for d in dets:
            print("I see:", d[4])
        if a.save:
            cv2.imwrite(a.save, draw(frame, dets))
        return

    cap = cv2.VideoCapture(a.camera)
    print("Press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Vision AI", draw(frame, detect(model, proc, frame, a.threshold, device)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
