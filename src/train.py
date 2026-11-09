"""
Phase 2: fine-tune MobileNetV2 on corn leaves, on a CPU.

    python -m src.train --quick     # ~300 images, 2 epochs/stage: smoke test
    python -m src.train             # the real run

Two-stage transfer learning:
  Stage 1 - freeze the pretrained network, train only a small new "head".
  Stage 2 - unfreeze the top of the network and fine-tune it with a tiny
            learning rate.
WHY two stages: the new head starts out random. If we trained everything at
once, the big random gradients from that head would flow back and wreck the
pretrained features in the first few steps. Training the head first means that
by stage 2 the gradients are small and sensible.
"""
import argparse
import json
import math
import os
import platform
import time
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's C++ info spam

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data import build_datasets, fill_background, group_split, list_images

keras = tf.keras
SERIES = {"train": "#2a78d6", "val": "#eb6834"}  # fixed colours, colour-blind validated
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def log(msg=""):
    # WHY flush: when the run is redirected to a log file, Python buffers output
    # and you'd see nothing for minutes. Flushing shows progress as it happens.
    print(msg, flush=True)


def fmt_min(seconds):
    return f"{seconds / 60:.1f} min"


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def prepare(raw_train, raw_val, raw_test, quick):
    """
    Turn the raw Keras datasets into fast, CPU-friendly pipelines.

    WHY .cache(): without it, every epoch re-reads and re-decodes every JPEG. On
    a CPU that decoding can cost more than the training step itself. We cache
    once, in RAM, as uint8 (4x smaller than float32: ~350 MB for train).

    WHY the background fill comes AFTER the cache for train: the fill colour is
    random and must change every epoch. If it were cached, epoch 1's colours
    would be frozen in, and the model could memorise them.
    For val/test the fill is fixed, so it's applied before caching (done once).

    WHY .prefetch(AUTOTUNE): prepares the next batch while the current one
    trains, so the CPU cores are never idle waiting for data.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    to_uint8 = lambda x, y: (tf.cast(x, tf.uint8), y)  # noqa: E731
    to_float = lambda x, y: (tf.cast(x, tf.float32), y)  # noqa: E731

    train = raw_train.unbatch()
    val = raw_val.unbatch()
    test = raw_test.unbatch()
    n_train, n_val, n_test = (len(d.file_paths) for d in (raw_train, raw_val, raw_test))
    if quick:
        # train is shuffled, so its first 300 images are a random mix.
        # val/test are NOT shuffled (sorted by class), so .take() would grab
        # only rust. Keeping every 5th image samples all classes AND keeps the
        # order aligned with file_paths[::5].
        train = train.take(300)
        val, test = val.shard(5, 0), test.shard(5, 0)
        n_train, n_val, n_test = 300, math.ceil(n_val / 5), math.ceil(n_test / 5)

    # WHY assert_cardinality: after unbatch()/take(), TensorFlow no longer knows
    # how many images there are, and Keras prints a scary "ran out of data"
    # warning at the end of epoch 1. Telling it the count removes that.
    count = tf.data.experimental.assert_cardinality
    train, val, test = train.apply(count(n_train)), val.apply(count(n_val)), test.apply(count(n_test))

    train = (
        train.map(to_uint8).cache()
        # WHY shuffle after cache: the cache stores a fixed order; reshuffling
        # here gives a new order (and new batch mixes) every epoch.
        .shuffle(n_train, seed=config.SEED, reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE)
        .map(to_float)
        .map(lambda x, y: (fill_background(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    def eval_pipe(ds):
        return (
            ds.batch(config.BATCH_SIZE)
            .map(lambda x, y: (fill_background(x, training=False), y), num_parallel_calls=AUTOTUNE)
            .map(to_uint8).cache().map(to_float)
            .prefetch(AUTOTUNE)
        )

    return train, eval_pipe(val), eval_pipe(test), n_train


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_augmentation(include_rotation=True):
    """
    Random changes applied to training images only (Keras turns these layers
    off automatically at prediction time).

    WHY augment: 2,350 training images is small. Flips, zooms and rotations
    show the model the same leaf in new positions so it learns "what rust looks
    like", not "where rust was in this particular photo".
    WHY RandomContrast especially: it's partial insurance against the model
    keying on lighting differences between classes (Phase 1: healthy images
    are brighter on average).
    """
    layers = [keras.layers.RandomFlip("horizontal")]
    if include_rotation:
        layers.append(keras.layers.RandomRotation(0.2))
    layers += [keras.layers.RandomZoom(0.2), keras.layers.RandomContrast(0.2)]
    return keras.Sequential(layers, name="augmentation")


def mobilenet_preprocessing():
    """
    Scale pixels from [0, 255] to [-1, 1] - exactly what
    tf.keras.applications.mobilenet_v2.preprocess_input does (x / 127.5 - 1).

    WHY this matters: the ImageNet weights were trained on [-1, 1] inputs. Feed
    them [0, 1] (Rescaling(1./255)) and every activation is off - accuracy drops
    with no error message.

    WHY a Rescaling layer instead of calling preprocess_input in a Lambda:
    it's the identical maths (checked in check_preprocessing below), but it
    saves and reloads cleanly. A Lambda layer can only be reloaded with
    safe_mode=False, which you don't want in a public app.

    CONSEQUENCE FOR THE APP: preprocessing is INSIDE the model. The app must
    feed raw 0-255 pixels and must NOT call preprocess_input again, or the
    pixels get scaled twice.
    """
    return keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocess")


def check_preprocessing():
    x = tf.random.uniform([2, 8, 8, 3], 0, 255)
    ours = mobilenet_preprocessing()(x)
    official = keras.applications.mobilenet_v2.preprocess_input(tf.identity(x))
    assert np.allclose(ours.numpy(), np.asarray(official), atol=1e-5), "preprocessing mismatch!"


def build_model(img_size, include_augmentation=True, include_rotation=True):
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base.trainable = False  # stage 1: pretrained features stay fixed

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image_0_255")
    x = inputs
    if include_augmentation:
        x = make_augmentation(include_rotation)(x)  # BEFORE preprocessing: works on 0-255 pixels
    x = mobilenet_preprocessing()(x)
    # WHY training=False: it keeps BatchNorm layers in inference mode, using
    # the running statistics learned on ImageNet. Without it, BatchNorm would
    # update those statistics from our small corn batches and corrupt the
    # pretrained network - even with trainable=False. It also stays in
    # inference mode in stage 2, which is what we want when fine-tuning.
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)  # 7x7x1280 feature map -> 1280 numbers
    x = keras.layers.Dropout(0.2)(x)  # randomly drop 20% during training: fights overfitting
    outputs = keras.layers.Dense(len(config.CLASS_NAMES), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="corn_mobilenetv2"), base


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
class EpochTimer(keras.callbacks.Callback):
    """Prints epoch time, an estimate for the rest of the stage, and a warning
    if an epoch takes far longer than expected."""

    def __init__(self, stage, max_epochs, expected_epoch_s=None):
        super().__init__()
        self.stage, self.max_epochs, self.expected = stage, max_epochs, expected_epoch_s
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.perf_counter() - self.t0
        self.times.append(dt)
        if len(self.times) == 1:
            remaining = (self.max_epochs - 1) * dt
            log(f"  [{self.stage}] epoch 1 took {dt:.0f}s. Up to {self.max_epochs - 1} more epochs "
                f"-> at most ~{fmt_min(remaining)} more (early stopping may end sooner).")
        # Epoch 1 includes one-off graph compilation, so only judge later epochs.
        if self.expected and len(self.times) > 1 and dt > 3 * self.expected:
            log(f"  !! [{self.stage}] epoch took {dt:.0f}s, >3x the {self.expected:.0f}s estimate.")


def time_train_steps(model, ds, n_steps=8, warmup=2):
    """Average seconds per training step, after warm-up (the first steps include
    one-off graph compilation and would skew the number)."""
    it = iter(ds.repeat())
    for _ in range(warmup):
        model.train_on_batch(*next(it))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        model.train_on_batch(*next(it))
    return (time.perf_counter() - t0) / n_steps


def measure_augmentation_cost(train_ds, img_size):
    """
    WHY: augmentation layers run on the CPU too. If they doubled epoch time
    we'd want to know before committing to the run (and maybe drop the most
    expensive one, RandomRotation).
    """
    # Fill the cache first: otherwise the first configuration timed would also
    # pay for decoding every JPEG, and look unfairly slow.
    for _ in train_ds:
        pass
    results = {}
    for name, kwargs in [("no augmentation", dict(include_augmentation=False)),
                         ("all augmentation", dict()),
                         ("augmentation minus rotation", dict(include_rotation=False))]:
        m, _ = build_model(img_size, **kwargs)
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy")
        results[name] = time_train_steps(m, train_ds)
        del m
    base_t = results["no augmentation"]
    log("  Stage-1 step time:")
    for k, v in results.items():
        log(f"    {k:30s} {v * 1000:6.0f} ms/step  ({v / base_t:.2f}x)")
    ratio = results["all augmentation"] / base_t
    if ratio > 2:
        log(f"  !! Augmentation costs {ratio:.2f}x - more than double. Consider dropping RandomRotation.")
    return {k: round(v, 4) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_history(h1, h2, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    n1 = len(h1["accuracy"])
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
        for split, key in [("train", metric), ("val", f"val_{metric}")]:
            ys = h1[key] + h2[key]
            ax.plot(range(1, len(ys) + 1), ys, color=SERIES[split], linewidth=2,
                    marker="o", markersize=4, label=split)
        ax.axvline(n1 + 0.5, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.text(n1 + 0.6, ax.get_ylim()[1], " fine-tuning starts", color=INK_MUTED,
                va="top", fontsize=9)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("Epoch", color=INK_MUTED)
        style(ax)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Training curves: stage 1 (frozen base) then stage 2 (fine-tuning)", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, out_path):
    names = [config.DISPLAY_NAMES[c] for c in config.CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(len(names)), names, rotation=15)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    ax.set_title("Confusion matrix (test set)", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confident_mistakes(images, y_true, probs, paths, out_path, k=10):
    """
    WHY: a 99% accuracy hides WHICH images fail. The mistakes the model is most
    sure about show what it has actually learned (e.g. "any brown streak is
    blight") far better than a single number does.
    """
    pred = probs.argmax(1)
    wrong = np.nonzero(pred != y_true)[0]
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:k]
    rows = []
    if len(wrong) == 0:
        log("  No test mistakes at all - nothing to display.")
        return rows
    cols = min(5, len(wrong))
    nrows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.8, nrows * 3.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        t, p = config.DISPLAY_NAMES[config.CLASS_NAMES[y_true[i]]], config.DISPLAY_NAMES[config.CLASS_NAMES[pred[i]]]
        ax.imshow(images[i].astype(np.uint8))
        ax.set_title(f"true: {t}\npred: {p} ({probs[i, pred[i]]:.0%})", fontsize=9, color=INK)
        rows.append({"file": os.path.basename(paths[i]), "true": t, "pred": p,
                     "confidence": round(float(probs[i, pred[i]]), 4)})
    fig.suptitle(f"{len(wrong)} most confidently wrong test predictions", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return rows


def load_images(paths, img_size):
    """Load + resize like image_dataset_from_directory does (bilinear)."""
    arr = [tf.image.resize(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32),
                           (img_size, img_size)) for p in paths]
    return tf.stack(arr)


def shortcut_test(model, img_size, quick):
    """
    Did the background fix work? Take Northern Leaf Blight TEST leaves and put
    them on the same kind of background rust images have:

      (a) black background  - PlantVillage's own segmented version of the photo
      (b) the grey fill     - (a) after our background fix, as the pipeline would
                              see a rust image at test time

    If the model is reading the leaf, it should still say "blight". If it says
    "rust", it learned the background, not the disease.
    """
    nlb = config.CLASS_NAMES[1]
    test_paths = group_split(list_images())["test"][nlb]
    if quick:
        test_paths = test_paths[:32]
    seg_paths, orig_paths = [], []
    for p in test_paths:
        s = config.SEGMENTED_DIR / nlb / f"{p.stem}_final_masked.jpg"
        if s.exists():
            seg_paths.append(s)
            orig_paths.append(p)
    if not seg_paths:
        log("  Segmented images not found in data/kaggle/segmented - skipping shortcut test.")
        return None

    def class_shares(x):
        pred = model.predict(x, batch_size=config.BATCH_SIZE, verbose=0).argmax(1)
        return {config.DISPLAY_NAMES[c]: round(float(np.mean(pred == i)), 4)
                for i, c in enumerate(config.CLASS_NAMES)}

    seg = load_images(seg_paths, img_size)
    res = {
        "n_images": len(seg_paths),
        "original_photo": class_shares(fill_background(load_images(orig_paths, img_size), False)),
        "black_background": class_shares(seg),
        "grey_fill_background": class_shares(fill_background(seg, False)),
    }
    log(f"  Northern Leaf Blight test leaves (n={res['n_images']}), share of predictions:")
    log(f"    {'background':24s} {'rust':>7s} {'blight':>7s} {'healthy':>8s}")
    for k in ("original_photo", "black_background", "grey_fill_background"):
        v = list(res[k].values())
        log(f"    {k:24s} {v[0]:7.1%} {v[1]:7.1%} {v[2]:8.1%}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="300-image subset, 2 epochs per stage: verify the script end to end")
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE,
                        help="224 (default, MobileNetV2 native). 160 roughly halves CPU time.")
    args = parser.parse_args()

    run_start = time.perf_counter()
    tf.keras.utils.set_random_seed(config.SEED)  # seeds Python, NumPy and TF in one call
    check_preprocessing()

    quick, img_size = args.quick, args.img_size
    e1 = 2 if quick else config.STAGE1_EPOCHS
    e2 = 2 if quick else config.STAGE2_EPOCHS
    # Quick runs write to separate folders so they never overwrite a real model.
    models_dir = config.MODELS_DIR / ("quick" if quick else "")
    docs_dir = config.DOCS_DIR / ("quick" if quick else "")
    ckpt_dir = models_dir / "checkpoints"
    for d in (models_dir, docs_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    log(f"Mode: {'QUICK smoke test' if quick else 'FULL run'} | IMG_SIZE={img_size} | "
        f"TF {tf.__version__} | {os.cpu_count()} CPU threads")

    raw_train, raw_val, raw_test = build_datasets(img_size=img_size)
    test_paths = raw_test.file_paths[::5] if quick else raw_test.file_paths
    train_ds, val_ds, test_ds, n_train = prepare(raw_train, raw_val, raw_test, quick)
    steps_per_epoch = math.ceil(n_train / config.BATCH_SIZE)

    # ---- augmentation cost ------------------------------------------------
    log("\nMeasuring augmentation cost...")
    aug_cost = measure_augmentation_cost(train_ds, img_size)

    # ---- stage 1 -----------------------------------------------------------
    model, base = build_model(img_size)
    model.compile(optimizer=keras.optimizers.Adam(config.STAGE1_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    total_params = model.count_params()
    log(f"\nModel: {total_params:,} params total; "
        f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,} trainable in stage 1")

    est1 = aug_cost["all augmentation"] * steps_per_epoch
    log(f"\n=== STAGE 1: feature extraction (up to {e1} epochs, est. ~{est1:.0f}s/epoch "
        f"+ first-epoch caching) ===")
    t1 = EpochTimer("stage 1", e1, expected_epoch_s=est1 * 2)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=e1, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                      restore_best_weights=True),
        # WHY checkpoint every stage: if stage 2 crashes an hour in, the best
        # stage-1 model is already safe on disk.
        keras.callbacks.ModelCheckpoint(ckpt_dir / "stage1_best.keras", monitor="val_loss",
                                        save_best_only=True),
        t1,
    ]).history
    stage1_s = sum(t1.times)

    # ---- stage 2 -----------------------------------------------------------
    base.trainable = True
    for layer in base.layers[:config.FINE_TUNE_FROM]:
        # WHY keep the bottom 100 frozen: early layers detect generic edges and
        # textures that are useful for any image. Only the top layers, which
        # encode more ImageNet-specific shapes, need adapting to corn leaves.
        layer.trainable = False
    # WHY recompile: changes to .trainable only take effect after compile().
    # WHY 1e-5: the pretrained weights are already good; a normal learning
    # rate would overwrite them in the first few steps.
    model.compile(optimizer=keras.optimizers.Adam(config.STAGE2_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    n_trainable2 = sum(int(np.prod(w.shape)) for w in model.trainable_weights)

    step2 = time_train_steps(model, train_ds, n_steps=5)
    est2 = step2 * steps_per_epoch
    s1_epoch = t1.times[1] if len(t1.times) > 1 else t1.times[0]
    log(f"\n=== STAGE 2: fine-tuning ({n_trainable2:,} trainable params, up to {e2} epochs) ===")
    log(f"  Measured: ~{est2:.0f}s/epoch ({est2 / max(s1_epoch, 1e-9):.1f}x a cached stage-1 epoch). "
        f"Worst case for {e2} epochs: ~{fmt_min(est2 * e2)}.")
    t2 = EpochTimer("stage 2", e2, expected_epoch_s=est2)
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=len(h1["loss"]),
                   epochs=len(h1["loss"]) + e2, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_dir / "stage2_best.keras", monitor="val_loss",
                                        save_best_only=True),
        t2,
    ]).history
    stage2_s = sum(t2.times)
    train_wall = stage1_s + stage2_s

    # ---- evaluation ----------------------------------------------------------
    log("\n=== EVALUATION on the held-out test set ===")
    images, labels = [], []
    for x, y in test_ds:
        images.append(x.numpy())
        labels.append(y.numpy())
    images, y_true = np.concatenate(images), np.concatenate(labels).argmax(1)
    probs = model.predict(images, batch_size=config.BATCH_SIZE, verbose=0)
    y_pred = probs.argmax(1)
    names = [config.DISPLAY_NAMES[c] for c in config.CLASS_NAMES]
    test_acc = float(np.mean(y_pred == y_true))
    report = classification_report(y_true, y_pred, labels=range(3), target_names=names,
                                   digits=4, output_dict=True, zero_division=0)
    log(classification_report(y_true, y_pred, labels=range(3), target_names=names,
                              digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(3))
    log(f"Confusion matrix (rows=true, cols=pred, order {names}):\n{cm}")

    plot_history(h1, h2, docs_dir / "training_curves.png")
    plot_confusion(cm, docs_dir / "confusion_matrix.png")
    mistakes = plot_confident_mistakes(images, y_true, probs, test_paths[:len(y_true)],
                                       docs_dir / "confident_mistakes.png")
    for m in mistakes:
        log(f"  wrong: {m['file']}  true={m['true']}  pred={m['pred']} ({m['confidence']:.1%})")

    log("\n=== SHORTCUT TEST: does the model read the leaf or the background? ===")
    shortcut = shortcut_test(model, img_size, quick)

    # ---- save ----------------------------------------------------------------
    model_path = models_dir / "corn_disease_model.keras"
    # WHY save -> reload -> save: model.save() also stores the Adam optimizer's
    # internal state (two extra numbers per trainable weight, ~15 MB), which
    # only matters for resuming training. Reloading with compile=False drops
    # the optimizer, and saving again gives a ~10 MB predict-only file.
    model.save(model_path)
    export = keras.models.load_model(model_path, compile=False)
    export.save(model_path)
    # WHY check: proves the file on disk reloads and predicts exactly like the
    # model we just evaluated, before anyone builds an app on it.
    reloaded = keras.models.load_model(model_path, compile=False)
    assert np.allclose(reloaded.predict(images[:16], verbose=0), probs[:16], atol=1e-4), \
        "saved model predicts differently from the trained model!"
    size_mb = model_path.stat().st_size / 1e6
    # WHY save class names: index i of the model's output means class_names[i].
    # The app reads this file instead of guessing, so the order can't drift.
    with open(models_dir / "class_names.json", "w") as f:
        json.dump(config.CLASS_NAMES, f, indent=2)
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "per_class": {n: {k: round(v, 4) for k, v in report[n].items()} for n in names},
        "confusion_matrix": {"order": names, "rows_true_cols_pred": cm.tolist()},
        "n_test": int(len(y_true)),
        "shortcut_test_nlb": shortcut,
        "confident_mistakes": mistakes,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "img_size": img_size,
        "quick_mode": quick,
        "epochs_run": {"stage1": len(h1["loss"]), "stage2": len(h2["loss"])},
        "params": {"total": int(total_params), "trainable_stage2": int(n_trainable2)},
        "model_file_mb": round(size_mb, 2),
        "augmentation_step_time_s": aug_cost,
        "wall_clock_s": {"stage1": round(stage1_s), "stage2": round(stage2_s),
                         "training_total": round(train_wall),
                         "script_total": round(time.perf_counter() - run_start)},
        "background_fix": "black backgrounds replaced: random colour (train), grey (val/test)",
        "split": "grouped by original photo (see src/data.py), ~70/15/15",
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": keras.__version__, "numpy": np.__version__},
        "cpu_threads": os.cpu_count(),
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log("\n" + "=" * 72)
    log(f"TEST ACCURACY: {test_acc:.2%}  (n={len(y_true)})")
    log("  ! READ THIS NEXT TO THE NUMBER: PlantVillage is lab imagery - one detached")
    log("    leaf, uniform background, even lighting. A near-perfect score here is")
    log("    normal and does NOT mean the model works on real field photos. Expect a")
    log("    large drop on photos taken in a field (tested in Phase 5).")
    log(f"Training wall-clock: {fmt_min(train_wall)} (stage 1 {fmt_min(stage1_s)}, "
        f"stage 2 {fmt_min(stage2_s)}); whole script {fmt_min(time.perf_counter() - run_start)}")
    log(f"Saved {model_path.relative_to(config.PROJECT_ROOT)} ({size_mb:.1f} MB), "
        f"class_names.json, metrics.json")
    log("=" * 72)


if __name__ == "__main__":
    main()
