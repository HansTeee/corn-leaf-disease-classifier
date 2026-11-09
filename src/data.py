"""
Loading and splitting the corn leaf images.

THE KEY IDEA IN THIS FILE: we do NOT split images randomly.

PlantVillage (at least the copy in data/raw/) contains files like:
    ...___R.S_HL 8189 copy.jpg
    ...___R.S_HL 8189 copy 2.jpg
    ...___R.S_HL 8189 copy 3.jpg
These are different crops of the SAME original photograph of the SAME leaf.
If one crop lands in train and another in test, the "test" is no longer unseen
data - the model has already seen that exact leaf, and the test score is inflated.

So we split by GROUP: every file that comes from the same original photo (and
its near neighbours in camera numbering) is assigned to train, val or test as a
unit. Then a test image's leaf has never been seen during training.

Once the groups are assigned, we materialise data/splits/{train,val,test}/<class>/
using hard links (no extra disk space) and load each folder with
tf.keras.utils.image_dataset_from_directory - the standard Keras loader.
"""
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from src import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# 1. Work out which original photo each file came from
# ---------------------------------------------------------------------------
def parse_photo_id(filename: str) -> tuple[str, int]:
    """
    Turn a filename into (camera_series, photo_number).

    Examples:
        'RS_Rust 1563.JPG'                                  -> ('RS_Rust', 1563)
        '0079c731-...___RS_NLB 4121.JPG'                    -> ('RS_NLB', 4121)
        '00031d74-...___R.S_HL 8315 copy 2.jpg'             -> ('R.S_HL', 8315)

    WHY: the long UUID prefix is unique per FILE, so it can't tell us which files
    are related. The part after '___' is the original camera filename; stripping
    ' copy' / ' copy 2' from it gives the photo the crop was cut from.
    """
    stem = Path(filename).stem
    stem = stem.split("___")[-1]                               # drop UUID prefix
    stem = re.sub(r"\s+copy(\s+\d+)?$", "", stem, flags=re.I)  # drop ' copy N'
    match = re.match(r"^(.*?)\s*(\d+)$", stem.strip())
    if match is None:
        # Unknown naming scheme: treat the file as its own group (safe fallback).
        return stem, -1
    return match.group(1).strip(), int(match.group(2))


def group_key(filename: str) -> str:
    """
    The unit we split on. Files with the same key always go to the same split.

    WHY the block: photo numbers 3700-3704 become one group, 3705-3709 the next,
    so back-to-back shots of the same leaf can't straddle train and test.
    """
    series, number = parse_photo_id(filename)
    if number < 0:
        return f"{series}"
    return f"{series}#{number // config.PHOTO_BLOCK}"


def list_images(raw_dir: Path = config.RAW_DIR) -> dict[str, list[Path]]:
    """Return {class_name: [image paths]} for the three classes we use."""
    images = {}
    for class_name in config.CLASS_NAMES:
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {class_dir}. Put the PlantVillage corn folders in data/raw/."
            )
        # sorted() so the file order (and therefore the split) doesn't depend on
        # the operating system's directory listing order.
        images[class_name] = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


# ---------------------------------------------------------------------------
# 2. Assign whole groups to train / val / test
# ---------------------------------------------------------------------------
def group_split(
    images: dict[str, list[Path]], seed: int = config.SEED
) -> dict[str, dict[str, list[Path]]]:
    """
    Split each class 70/15/15 BY GROUP. Returns {split: {class: [paths]}}.

    WHY per class (stratified): doing it class by class keeps the class balance
    roughly the same in train, val and test, so the test set isn't accidentally
    short of one disease.

    WHY the ratios are approximate: groups have different sizes (1 to ~10
    files), so we can't hit 70.0% exactly. We fill train until it reaches 70% of
    the images, then val until 85%, and the rest is test. It lands within a
    percent or so, which doesn't matter; leakage does.
    """
    rng = random.Random(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for class_name, paths in images.items():
        groups = defaultdict(list)
        for p in paths:
            groups[group_key(p.name)].append(p)

        keys = sorted(groups)        # deterministic order before shuffling
        rng.shuffle(keys)

        total = len(paths)
        train_cut = config.TRAIN_FRAC * total
        val_cut = (config.TRAIN_FRAC + config.VAL_FRAC) * total

        for s in splits:
            splits[s][class_name] = []
        seen = 0
        for k in keys:
            if seen < train_cut:
                target = "train"
            elif seen < val_cut:
                target = "val"
            else:
                target = "test"
            splits[target][class_name].extend(groups[k])
            seen += len(groups[k])

    return splits


def check_no_group_overlap(splits: dict[str, dict[str, list[Path]]]) -> None:
    """
    Fail loudly if any group appears in more than one split.

    WHY: this is the whole point of the module. A cheap assertion now is far
    better than discovering leakage after reporting a 99% test score.
    """
    owner = {}
    for split_name, by_class in splits.items():
        for class_name, paths in by_class.items():
            for p in paths:
                key = (class_name, group_key(p.name))
                if owner.setdefault(key, split_name) != split_name:
                    raise AssertionError(f"Leak: group {key} in {owner[key]} and {split_name}")


# ---------------------------------------------------------------------------
# 3. Write the split to disk and load it with Keras
# ---------------------------------------------------------------------------
def materialise_splits(
    splits: dict[str, dict[str, list[Path]]], split_dir: Path = config.SPLIT_DIR
) -> None:
    """
    Create data/splits/{train,val,test}/<class>/ containing links to the images.

    WHY on disk: image_dataset_from_directory reads a folder-per-class layout.
    Its own validation_split option splits FILES randomly, which is exactly the
    leaky behaviour we're avoiding - so we do the split ourselves and give it
    one folder per split.

    WHY hard links: they look like normal files to TensorFlow but take no extra
    disk space. If the filesystem doesn't support them, we fall back to copying.

    WHY delete first: if an old split folder were left behind, files from a
    previous split could mix into the new one.
    """
    if split_dir.exists():
        shutil.rmtree(split_dir)
    rng = random.Random(config.SEED)
    for split_name, by_class in splits.items():
        for class_name, paths in by_class.items():
            out = split_dir / split_name / class_name
            out.mkdir(parents=True, exist_ok=True)
            # Which files get swapped for their cut-out version (see config).
            swap = set()
            if class_name == config.CUTOUT_CLASS and split_name in config.CUTOUT_SPLITS:
                swap = set(rng.sample(range(len(paths)), int(config.CUTOUT_FRACTION * len(paths))))
            for i, p in enumerate(paths):
                src = p
                if i in swap:
                    cutout = cutout_path(p, class_name)
                    if cutout.exists():
                        src = cutout
                try:
                    os.link(src, out / src.name)
                except OSError:
                    shutil.copy2(src, out / src.name)


def cutout_path(p: Path, class_name: str) -> Path:
    """PlantVillage names the cut-out version '<original stem>_final_masked.jpg'."""
    return config.SEGMENTED_DIR / class_name / f"{p.stem}_final_masked.jpg"


# ---------------------------------------------------------------------------
# 4. Background fix: stop "black background" from meaning "rust"
# ---------------------------------------------------------------------------
# Phase 1 found that 100% of PlantVillage's Common rust images (in the official
# color/ release) have their background removed and filled with pure black,
# while the other classes don't. A model can score ~100% by detecting black
# corners. These functions replace that black background before the model sees
# it, so the only remaining way to tell rust apart is the leaf itself.

def black_background_mask(images):
    """
    Return a [B, H, W, 1] float mask: 1.0 where an image's removed background is.

    WHY each step:
    - max over RGB < threshold: a pixel is "black" only if ALL channels are dark.
      Rust pustules are brown (red channel well above 25), so they survive.
    - dilate (max-pool 5x5): JPEG compression and resizing smear the leaf edge
      into a ring of dark-grey pixels. Left alone, that dark outline would be a
      new shortcut. Growing the mask by 2 px covers it.
    - only if >5% of the image is black: so a dark shadow on a healthy or blight
      leaf is never touched. Phase 1 measured this gate selects 1,189/1,192 rust
      images, 2/985 blight and 1/1,162 healthy.
    """
    import tensorflow as tf

    dark = tf.cast(tf.reduce_max(images, axis=-1, keepdims=True) < config.BLACK_THRESHOLD, tf.float32)
    dilated = tf.nn.max_pool2d(dark, ksize=5, strides=1, padding="SAME")
    has_black_bg = tf.reduce_mean(dark, axis=[1, 2, 3], keepdims=True) > config.BLACK_MIN_FRACTION
    return dilated * tf.cast(has_black_bg, tf.float32)


def fill_background(images, training: bool):
    """
    Replace black background pixels with a fill colour.

    training=True: each image gets a RANDOM colour plus pixel noise, freshly
      drawn every epoch. WHY random: if we always used one grey, the model could
      learn "flat grey = rust" instead. Random colours make the background carry
      no information at all.
    training=False (val/test): a fixed grey close to the board in the blight
      photos, with mild noise. WHY fixed: evaluation must be repeatable, and
      using the board colour means the test can't be passed via the background.
    """
    import tensorflow as tf

    mask = black_background_mask(images)
    shape = tf.shape(images)
    if training:
        colour = tf.random.uniform([shape[0], 1, 1, 3], 40.0, 220.0)
        noise = tf.random.normal(shape, stddev=12.0)
    else:
        colour = tf.constant(config.BOARD_RGB, tf.float32)[None, None, None, :]
        # Stateless noise with a fixed seed: same "random" texture every run.
        noise = tf.random.stateless_normal(shape, seed=[config.SEED, 0], stddev=12.0)
    fill = tf.clip_by_value(colour + noise, 0.0, 255.0)
    return images * (1.0 - mask) + fill * mask


def build_datasets(img_size: int = config.IMG_SIZE, batch_size: int = config.BATCH_SIZE):
    """
    Build the leakage-safe train/val/test tf.data datasets.

    Returns (train_ds, val_ds, test_ds) of RAW pixels (0-255 floats, black
    backgrounds NOT yet replaced - see prepare_for_training in train.py, which
    decides where caching and the background fill go). Labels are one-hot
    (label_mode='categorical') in the order of config.CLASS_NAMES.
    """
    import tensorflow as tf  # imported here so the EDA can run without loading TF

    splits = group_split(list_images())
    check_no_group_overlap(splits)
    materialise_splits(splits)

    def load(split_name: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            config.SPLIT_DIR / split_name,
            # WHY pass class_names: by default Keras sorts folder names
            # alphabetically. Passing our list guarantees label index i means
            # config.CLASS_NAMES[i], whatever the folder order on disk.
            class_names=config.CLASS_NAMES,
            label_mode="categorical",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            # WHY shuffle only train: training needs a new random order each
            # epoch; val/test order doesn't matter, and a fixed order makes it
            # easy to match predictions back to file paths later.
            shuffle=shuffle,
            seed=config.SEED,
        )

    return load("train", True), load("val", False), load("test", False)


if __name__ == "__main__":
    # Quick self-check: print split sizes per class.
    splits = group_split(list_images())
    check_no_group_overlap(splits)
    for split_name, by_class in splits.items():
        sizes = {config.DISPLAY_NAMES[c]: len(v) for c, v in by_class.items()}
        print(f"{split_name:5s} total={sum(sizes.values()):5d}  {sizes}")
    print("No group appears in more than one split.")
