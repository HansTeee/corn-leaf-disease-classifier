r"""
evaluate_model.py — get REAL resume metrics for the Corn Leaf Disease Classifier
=================================================================================
What it does
  1. Loads your saved Keras model (.keras or .h5) and the class-name list.
  2. Reads every image in a folder that has ONE SUBFOLDER PER CLASS, e.g.
         data\test\Blight\img001.jpg
         data\test\Common_Rust\img002.jpg
         data\test\Gray_Leaf_Spot\...
         data\test\Healthy\...
  3. Predicts each image and compares with the true class (= folder name).
  4. Prints + saves: accuracy, per-class precision / recall / F1, macro F1,
     confusion matrix (PNG), and a list of the images it got wrong.

IMPORTANT — which folder to evaluate on
  * Use ONLY images the model never trained on:
      - the held-out TEST split (best for "test accuracy" on the resume), or
      - a small "real_world" folder of photos you took / found online
        (best for the "real-world accuracy" bullet).
  * If you point it at the full training folder the number will be inflated
    and NOT honest to put on a resume.

Usage (Windows CMD, venv activated, from the project folder):
    python evaluate_model.py --model models\corn_disease_model.keras --data data\test
    python evaluate_model.py --model models\corn_disease_model.keras --data data\real_world --tag real_world

Optional flags:
    --classes models\class_names.json   list of class names in training order
                                         (if missing, the sorted subfolder names are used,
                                          which matches Keras' image_dataset_from_directory)
    --preprocess none | mobilenet | rescale
         none      -> model already contains preprocessing layers (Project 9 default)
         mobilenet -> apply tf.keras.applications.mobilenet_v2.preprocess_input ([-1, 1])
         rescale   -> divide pixels by 255 ([0, 1])
       If accuracy comes out near 1/num_classes (e.g. ~25% for 4 classes) the
       preprocessing choice is almost certainly wrong — try the other options.
    --out results                         folder where JSON / PNG / CSV are saved
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the corn leaf disease model.")
    p.add_argument("--model", required=True, help="Path to .keras or .h5 model file")
    p.add_argument("--data", required=True, help="Folder with one subfolder per class")
    p.add_argument("--classes", default=None, help="JSON file with class names (optional)")
    p.add_argument("--preprocess", default="none", choices=["none", "mobilenet", "rescale"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="test", help="Name used for output files, e.g. test / real_world")
    return p.parse_args()


def load_class_names(classes_path, data_dir):
    """Class order MUST match the order used at training time."""
    if classes_path and Path(classes_path).exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Accept either ["Blight", ...] or {"0": "Blight", ...} / {"class_names": [...]}
        if isinstance(obj, dict):
            if "class_names" in obj:
                obj = obj["class_names"]
            else:
                obj = [obj[k] for k in sorted(obj, key=lambda k: int(k))]
        print(f"[info] Class names from {classes_path}: {obj}")
        return list(obj)
    names = sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())
    print(f"[info] No class file given -> using sorted folder names: {names}")
    return names


def collect_images(data_dir, class_names):
    """Return list of (path, true_index). Folders not in class_names are skipped with a warning."""
    import re
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # "Corn_(maize)___Common_rust_" -> "cornmaizecommonrust"
    items = []
    lower_map = {c.lower(): i for i, c in enumerate(class_names)}
    for sub in sorted(Path(data_dir).iterdir()):
        if not sub.is_dir():
            continue
        idx = lower_map.get(sub.name.lower())
        if idx is None:
            # Short folder names are fine: "Common_rust" matches "Corn_(maize)___Common_rust_"
            cands = [i for i, c in enumerate(class_names) if norm(sub.name) and norm(sub.name) in norm(c)]
            if len(cands) == 1:
                idx = cands[0]
                print(f"[info] Folder '{sub.name}' -> class '{class_names[idx]}'")
        if idx is None:
            print(f"[warn] Folder '{sub.name}' is not one of the model's classes -> skipped")
            continue
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f, idx))
    return items


def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs
    import tensorflow as tf

    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    if not Path(args.model).exists():
        sys.exit(f"[error] Model not found: {args.model}  (run  dir models  to see the file name)")
    if not Path(args.data).is_dir():
        sys.exit(f"[error] Data folder not found: {args.data}")

    # ---- 1. Load model -------------------------------------------------
    print(f"[info] Loading model {args.model} ...")
    model = tf.keras.models.load_model(args.model, compile=False)
    in_shape = model.input_shape  # e.g. (None, 224, 224, 3)
    img_h, img_w = int(in_shape[1]), int(in_shape[2])
    print(f"[info] Model expects images of size {img_w}x{img_h}")

    class_names = load_class_names(args.classes, args.data)
    n_out = int(model.output_shape[-1])
    if n_out != len(class_names):
        sys.exit(f"[error] Model outputs {n_out} classes but {len(class_names)} class names were found. "
                 f"Pass the right --classes file.")

    # ---- 2. Collect images --------------------------------------------
    items = collect_images(args.data, class_names)
    if not items:
        sys.exit("[error] No images found. Expected data\\<ClassName>\\*.jpg")
    print(f"[info] Found {len(items)} images")
    for i, c in enumerate(class_names):
        print(f"         {c:<20} {sum(1 for _, t in items if t == i)}")

    # ---- 3. Predict in batches ----------------------------------------
    def load_img(path):
        img = tf.keras.utils.load_img(path, target_size=(img_h, img_w))
        return tf.keras.utils.img_to_array(img)  # float32, 0..255

    y_true, y_pred, confs, unreadable = [], [], [], []
    start = time.perf_counter()
    for b in range(0, len(items), args.batch):
        chunk = items[b:b + args.batch]
        arrs, labels = [], []
        for path, t in chunk:
            try:
                arrs.append(load_img(path))
                labels.append(t)
            except Exception as e:  # corrupt / unsupported file
                unreadable.append((str(path), str(e)))
        if not arrs:
            continue
        x = np.stack(arrs)
        if args.preprocess == "mobilenet":
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        elif args.preprocess == "rescale":
            x = x / 255.0
        probs = model.predict(x, verbose=0)
        # If the last layer has no softmax (logits), convert to probabilities
        if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
            probs = tf.nn.softmax(probs).numpy()
        y_true.extend(labels)
        y_pred.extend(probs.argmax(axis=1).tolist())
        confs.extend(probs.max(axis=1).tolist())
        print(f"\r[info] Predicted {min(b + args.batch, len(items))}/{len(items)}", end="")
    elapsed = time.perf_counter() - start
    print()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels_idx = list(range(len(class_names)))

    # ---- 4. Metrics ---------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_idx, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels_idx, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels_idx, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    ms_per_img = 1000 * elapsed / max(len(y_true), 1)

    print("\n" + "=" * 62)
    print(f" RESULTS on '{args.data}'  ({len(y_true)} images)")
    print("=" * 62)
    print(f" Accuracy           : {acc * 100:.2f}%")
    print(f" Macro F1           : {macro_f1:.3f}")
    print(f" Weighted F1        : {weighted_f1:.3f}")
    print(f" Avg time per image : {ms_per_img:.1f} ms (CPU, batch {args.batch})")
    print("-" * 62)
    print(classification_report(y_true, y_pred, labels=labels_idx, target_names=class_names,
                                digits=3, zero_division=0))
    # Short display names: "Corn_(maize)___Common_rust_" -> "Common_rust"
    short = [c.split("___")[-1].strip("_")[:14] for c in class_names]
    print("Confusion matrix (rows = true, cols = predicted):")
    print(" " * 16 + " ".join(f"{c:>14}" for c in short))
    for i, row in enumerate(cm):
        print(f"{short[i]:>14}  " + " ".join(f"{v:>14}" for v in row))

    chance = 1 / len(class_names)
    if acc < chance + 0.10:
        print(f"\n[WARNING] Accuracy is close to random guessing ({chance:.0%}). "
              f"Try a different --preprocess option or check --classes order.")

    # ---- 5. Save outputs ---------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_folder": str(args.data),
        "n_images": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "ms_per_image_cpu": round(ms_per_img, 1),
        "per_class": {c: {k: round(float(v), 4) for k, v in report[c].items()} for c in class_names},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "preprocess": args.preprocess,
        "unreadable_files": len(unreadable),
    }
    json_path = out / f"metrics_{args.tag}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Misclassified images -> useful for the "honest limitations" part of the README
    wrong_path = out / f"misclassified_{args.tag}.csv"
    readable = [it for it in items if str(it[0]) not in {u[0] for u in unreadable}]
    with open(wrong_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "predicted", "confidence"])
        for (path, _), t, p_, c in zip(readable, y_true, y_pred, confs):
            if t != p_:
                w.writerow([str(path), class_names[t], class_names[p_], f"{c:.3f}"])

    # Confusion matrix PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 1.4 * len(class_names) + 1.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(labels_idx, short, rotation=30, ha="right")
        ax.set_yticks(labels_idx, short)
        for i in labels_idx:
            for j in labels_idx:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Corn leaf disease — {args.tag} (acc {acc * 100:.1f}%)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        png_path = out / f"confusion_matrix_{args.tag}.png"
        fig.savefig(png_path, dpi=150)
        print(f"\n[saved] {png_path}")
    except Exception as e:
        print(f"[warn] Could not draw confusion matrix PNG: {e}")

    print(f"[saved] {json_path}")
    print(f"[saved] {wrong_path}")
    if unreadable:
        print(f"[warn] {len(unreadable)} files could not be read and were skipped.")

    print("\nRESUME LINE (copy, then check it reads right):")
    print(f"  ... achieving {acc * 100:.1f}% accuracy (macro F1 {macro_f1:.2f}) on "
          f"{len(y_true)} held-out images across {len(class_names)} classes")


if __name__ == "__main__":
    main()
