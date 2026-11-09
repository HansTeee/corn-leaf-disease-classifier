"""
Phase 1 exploratory data analysis. Run with:   python -m src.eda

Prints a report and saves charts to docs/. Nothing here trains a model; the goal
is to find problems in the data BEFORE they quietly ruin the results.
"""
import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # WHY: render straight to PNG files; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config
from src.data import check_no_group_overlap, group_key, group_split, list_images, parse_photo_id

# Palette: the first three slots of a colour-blind-validated categorical palette,
# always in the same order so a class keeps its colour across every chart.
CLASS_COLORS = dict(zip(config.CLASS_NAMES, ["#2a78d6", "#eb6834", "#1baf7a"]))
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax):
    """Quiet axes so the data, not the chart furniture, draws the eye."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def header(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 1. Class counts and imbalance
# ---------------------------------------------------------------------------
def class_counts(images):
    header("1. IMAGE COUNT PER CLASS")
    counts = {c: len(p) for c, p in images.items()}
    for c, n in counts.items():
        print(f"  {config.DISPLAY_NAMES[c]:22s} {n:5d}")
    ratio = max(counts.values()) / min(counts.values())
    print(f"  Total: {sum(counts.values())}")
    # WHY the ratio matters: a big imbalance (say 10:1) would let a model score
    # well by mostly guessing the big class, and we'd need class weights. Under
    # ~1.5:1 it's usually fine to train as-is.
    print(f"  Imbalance ratio (largest / smallest): {ratio:.2f} : 1")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [config.DISPLAY_NAMES[c] for c in counts]
    bars = ax.bar(names, counts.values(), color=[CLASS_COLORS[c] for c in counts], width=0.55)
    ax.bar_label(bars, padding=3, color=INK)
    ax.set_ylabel("Images", color=INK_MUTED)
    ax.set_title(f"Images per class (imbalance {ratio:.2f} : 1)", color=INK, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(config.DOCS_DIR / "eda_class_counts.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Sample grid
# ---------------------------------------------------------------------------
def sample_grid(images, per_class=3):
    header("2. SAMPLE GRID")
    rng = random.Random(config.SEED)
    fig, axes = plt.subplots(len(images), per_class, figsize=(per_class * 2.6, len(images) * 2.8))
    for row, (c, paths) in enumerate(images.items()):
        for col, p in enumerate(rng.sample(paths, per_class)):
            ax = axes[row, col]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if col == 0:
                ax.set_ylabel(config.DISPLAY_NAMES[c], color=INK, fontsize=11)
    fig.suptitle("Random samples per class", color=INK)
    fig.tight_layout()
    out = config.DOCS_DIR / "eda_sample_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Saved {out.relative_to(config.PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 3 + 5. Dimensions and corrupt files (one pass over every image)
# ---------------------------------------------------------------------------
def dimensions_and_corruption(images):
    header("3. IMAGE DIMENSIONS   +   5. CORRUPT / UNREADABLE FILES")
    import tensorflow as tf

    sizes, modes, bad = Counter(), Counter(), []
    odd_sizes = []
    for c, paths in images.items():
        for p in paths:
            try:
                # WHY two checks: PIL's verify() catches truncated/garbled files;
                # decoding with TensorFlow catches files PIL tolerates but the
                # training pipeline would choke on (TF is what actually reads them).
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
                    if im.size != (256, 256):
                        odd_sizes.append((p.name, im.size))
                tf.io.decode_image(tf.io.read_file(str(p)), channels=3)
            except Exception as e:  # noqa: BLE001 - we want to record any failure
                bad.append((str(p.relative_to(config.PROJECT_ROOT)), repr(e)))

    print("  Sizes (width x height): ", {f"{w}x{h}": n for (w, h), n in sizes.items()})
    print("  Colour modes:           ", dict(modes))
    if odd_sizes:
        print(f"  {len(odd_sizes)} images are NOT 256x256, e.g. {odd_sizes[:5]}")
    else:
        print("  All images are 256x256.")
    print(f"  Corrupt/unreadable files: {len(bad)}")
    for b in bad[:10]:
        print("    ", b)


# ---------------------------------------------------------------------------
# 4. Brightness per class (and the background problem it uncovers)
# ---------------------------------------------------------------------------
def brightness(images):
    header("4. MEAN PIXEL BRIGHTNESS PER CLASS")
    # WHY: if one class is systematically brighter or darker, a model can learn
    # "dark image = class X" instead of the actual disease. That shortcut gives
    # a great test score on this dataset and fails in the real world.
    whole, leaf_only, black_bg = defaultdict(list), defaultdict(list), Counter()
    for c, paths in images.items():
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            gray = a.mean(axis=2)
            whole[c].append(gray.mean())
            # Near-black pixels (all channels < 10) are background, not leaf.
            # Measuring brightness with and without them separates "the leaf is
            # darker" from "there's a black background".
            is_black = a.max(axis=2) < 10
            leaf_only[c].append(gray[~is_black].mean())
            if is_black.mean() > 0.05:
                black_bg[c] += 1

    print(f"  {'class':22s} {'whole image':>12s} {'excl. black px':>15s} {'black background':>18s}")
    for c in images:
        n = len(images[c])
        print(f"  {config.DISPLAY_NAMES[c]:22s} {np.mean(whole[c]):12.1f} "
              f"{np.mean(leaf_only[c]):15.1f} {black_bg[c]:8d} / {n} ({black_bg[c] / n:4.0%})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    bins = np.linspace(0, 255, 52)
    for ax, data, title in [
        (axes[0], whole, "Whole image"),
        (axes[1], leaf_only, "Excluding pure-black background pixels"),
    ]:
        for c in images:
            ax.hist(data[c], bins=bins, histtype="step", linewidth=2,
                    color=CLASS_COLORS[c], label=config.DISPLAY_NAMES[c])
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("Mean grey level per image (0-255)", color=INK_MUTED)
        style_axes(ax)
    axes[0].set_ylabel("Images", color=INK_MUTED)
    axes[1].legend(frameon=False)
    fig.suptitle("Brightness distribution per class", color=INK)
    fig.tight_layout()
    fig.savefig(config.DOCS_DIR / "eda_brightness.png", dpi=120)
    plt.close(fig)
    return black_bg


# ---------------------------------------------------------------------------
# 6. Leakage audit
# ---------------------------------------------------------------------------
def dhash_bits(img, size=8):
    """
    Difference hash: shrink to 9x8 grey, record whether each pixel is brighter
    than its right-hand neighbour -> 64 bits. Near-identical images give hashes
    that differ in only a few bits. WHY dHash: cheap, and robust to brightness,
    contrast and JPEG re-compression. Its blind spot: CROPS and shifts.
    """
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0]


def leakage_audit(images):
    header("6. LEAKAGE AUDIT")

    # --- 6a. filename evidence -------------------------------------------------
    print("  6a. Filenames: files cut from the same original photo ('X', 'X copy', 'X copy 2')")
    for c, paths in images.items():
        photos = Counter(parse_photo_id(p.name) for p in paths)
        n_copy = sum("copy" in p.name.lower() for p in paths)
        in_multi = sum(n for n in photos.values() if n > 1)
        print(f"    {config.DISPLAY_NAMES[c]:22s} files={len(paths):5d}  distinct photos={len(photos):5d}  "
              f"'copy' in name={n_copy:5d}  files sharing a photo={in_multi:5d} ({in_multi / len(paths):.0%})")

    # --- 6b. what a naive random split would leak -----------------------------
    # WHY simulate: "N duplicates exist" is abstract. "X% of test images have a
    # sibling crop of the same photo in train" is what actually inflates scores.
    rng = random.Random(config.SEED)
    all_files = [(c, p) for c, ps in images.items() for p in ps]
    n_train = int(config.TRAIN_FRAC * len(all_files))
    n_test = int(config.TEST_FRAC * len(all_files))
    leaked = defaultdict(list)  # class -> list of True/False, one per test image
    for _ in range(20):
        rng.shuffle(all_files)
        train_photos = {(c, parse_photo_id(p.name)) for c, p in all_files[:n_train]}
        for c, p in all_files[-n_test:]:
            leaked[c].append((c, parse_photo_id(p.name)) in train_photos)
    print("\n  6b. Naive random 70/15/15 split (20 repeats): test images whose original photo is ALSO in train")
    print(f"    overall: {np.mean([x for v in leaked.values() for x in v]):.1%}")
    for c in images:
        print(f"    {config.DISPLAY_NAMES[c]:22s} {np.mean(leaked[c]):.1%}")

    # --- 6c. perceptual hash over ALL images ----------------------------------
    # The brief said "a sample"; with 3,339 small images, all of them takes
    # seconds, so there's no reason to sample. For each image we also hash its
    # flips/rotations, so a flipped copy still counts as a near-duplicate.
    print("\n  6c. Perceptual hash (dHash, 64 bits) over all images, flip/rotation-aware")
    entries = [(c, p) for c, ps in images.items() for p in ps]
    base = np.zeros(len(entries), dtype=np.uint64)
    variants = np.zeros((len(entries), 6), dtype=np.uint64)
    for i, (_, p) in enumerate(entries):
        im = Image.open(p)
        ts = [im, im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM),
              im.transpose(Image.ROTATE_90), im.transpose(Image.ROTATE_180), im.transpose(Image.ROTATE_270)]
        hs = [dhash_bits(t) for t in ts]
        base[i] = hs[0]
        variants[i] = hs
    # Hamming distance between image i and the closest transform of image j.
    dist = np.full((len(entries), len(entries)), 64, dtype=np.int16)
    for k in range(6):
        d = np.bitwise_count(base[:, None] ^ variants[None, :, k]).astype(np.int16)
        dist = np.minimum(dist, d)
    dist = np.minimum(dist, dist.T)
    iu = np.triu_indices(len(entries), k=1)
    d_pairs = dist[iu]

    for thr in (4, 8, 12):
        close = np.nonzero(d_pairs <= thr)[0]
        n_same_photo = sum(
            entries[iu[0][k]][0] == entries[iu[1][k]][0]
            and parse_photo_id(entries[iu[0][k]][1].name) == parse_photo_id(entries[iu[1][k]][1].name)
            for k in close
        )
        n_cross_class = sum(entries[iu[0][k]][0] != entries[iu[1][k]][0] for k in close)
        print(f"    pairs within {thr:2d}/64 bits: {len(close):5d}   "
              f"(same original photo: {n_same_photo}, different classes: {n_cross_class})")

    # How similar are the filename-identified sibling crops, according to dHash?
    sib = []
    index = defaultdict(list)
    for i, (c, p) in enumerate(entries):
        index[(c, parse_photo_id(p.name))].append(i)
    for members in index.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                sib.append(dist[members[a], members[b]])
    print(f"    Hash distance between sibling crops of the SAME photo (n={len(sib)} pairs): "
          f"median {np.median(sib):.0f}/64, only {np.mean(np.array(sib) <= 8):.0%} within 8 bits")
    print("    -> dHash misses most sibling crops (a crop shifts every gradient), so the")
    print("       filename grouping, not the hash, is what the split must rely on.")

    # --- 6d. the group split actually used ------------------------------------
    splits = group_split(images)
    check_no_group_overlap(splits)
    split_of = {p: s for s, by_c in splits.items() for ps in by_c.values() for p in ps}
    print("\n  6d. Group split used by src/data.py")
    for s, by_c in splits.items():
        n = sum(len(v) for v in by_c.values())
        print(f"    {s:5s} {n:5d} images ({n / len(entries):.1%})")
    print("    Original photos shared between splits: 0 (asserted)")
    # WHY <=4 bits: at 8 bits the hash already matches 169 pairs from DIFFERENT
    # classes - impossible as true duplicates - because plain green leaf
    # close-ups all look alike to a 64-bit hash. <=4 bits is where matches are
    # dominated by real duplicates (checked by eye: docs/eda_close_pairs.png).
    close = np.nonzero(d_pairs <= 4)[0]
    same_split = [k for k in close if split_of[entries[iu[0][k]][1]] == split_of[entries[iu[1][k]][1]]]
    same_photo = [k for k in close if entries[iu[0][k]][0] == entries[iu[1][k]][0]
                  and group_key(entries[iu[0][k]][1].name) == group_key(entries[iu[1][k]][1].name)]
    print(f"    Hash pairs <=4 bits: {len(close)}; same photo group: {len(same_photo)} "
          f"(all kept in one split); in the same split overall: {len(same_split)}; "
          f"crossing splits: {len(close) - len(same_split)}")
    save_close_pairs(entries, iu, d_pairs, close, split_of)


def save_close_pairs(entries, iu, d_pairs, close, split_of, cols=4, w=128):
    """Save every <=4-bit pair side by side, so a human can judge them."""
    from PIL import ImageDraw
    rows = (len(close) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (2 * w + 24), rows * (w + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, k in enumerate(close):
        (ca, pa), (cb, pb) = entries[iu[0][k]], entries[iu[1][k]]
        x, y = (n % cols) * (2 * w + 24), (n // cols) * (w + 16)
        sheet.paste(Image.open(pa).resize((w, w)), (x, y))
        sheet.paste(Image.open(pb).resize((w, w)), (x + w, y))
        label = (f"{config.DISPLAY_NAMES[ca][:7]}/{config.DISPLAY_NAMES[cb][:7]} "
                 f"{d_pairs[k]}b {split_of[pa]}/{split_of[pb]}")
        draw.text((x + 2, y + w + 2), label, fill=INK)
    sheet.save(config.DOCS_DIR / "eda_close_pairs.png")


def main():
    config.DOCS_DIR.mkdir(exist_ok=True)
    images = list_images()
    class_counts(images)
    sample_grid(images)
    dimensions_and_corruption(images)
    brightness(images)
    leakage_audit(images)


if __name__ == "__main__":
    main()
