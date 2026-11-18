"""
Copy the held-out TEST split to data/test/<class_name>/ so evaluate_model.py can read it.

    python -m src.export_test_split

WHY reuse src/data.py instead of re-splitting: the split must be the exact one
training used, or the "test" score could include training leaves. The split in
data.py is a pure function of (sorted file list in data/raw, config.SEED,
config.TRAIN_FRAC/VAL_FRAC, config.PHOTO_BLOCK), with no tf.data randomness, so
calling the same functions reproduces it exactly. As an extra check we compare
against data/splits/test/, which train.py materialised during the real run.

The test split never gets cut-out swaps (config.CUTOUT_SPLITS = train, val), so
the test files are exactly the raw paths returned by group_split.
"""
import shutil
from pathlib import Path

from src import config
from src.data import check_no_group_overlap, cutout_path, group_key, group_split, list_images

OUT_DIR = config.DATA_DIR / "test"


def main():
    splits = group_split(list_images(), seed=config.SEED)
    check_no_group_overlap(splits)

    train_paths = {p.resolve() for paths in splits["train"].values() for p in paths}
    val_paths = {p.resolve() for paths in splits["val"].values() for p in paths}
    test = splits["test"]
    test_paths = {p.resolve() for paths in test.values() for p in paths}

    # 1. File-path leakage check against train (and val).
    assert not (test_paths & train_paths), f"{len(test_paths & train_paths)} test files are in train!"
    assert not (test_paths & val_paths), f"{len(test_paths & val_paths)} test files are in val!"
    # Cut-out swaps in train use '<stem>_final_masked.jpg' of TRAIN files only; make sure
    # no test leaf's cut-out version is among them (it can't be, but check).
    train_cutouts = {cutout_path(p, c).name for c, ps in splits["train"].items() for p in ps}
    test_cutouts = {cutout_path(p, c).name for c, ps in test.items() for p in ps}
    assert not (train_cutouts & test_cutouts), "a test leaf's cut-out version is in train!"
    # Group-level check (same original photo / camera-number block).
    train_groups = {(c, group_key(p.name)) for c, ps in splits["train"].items() for p in ps}
    test_groups = {(c, group_key(p.name)) for c, ps in test.items() for p in ps}
    assert not (train_groups & test_groups), "a photo group appears in both train and test!"
    print(f"[ok] 0 of {len(test_paths)} test files are in train ({len(train_paths)} files) or val "
          f"({len(val_paths)} files); 0 shared photo groups; 0 shared cut-out versions.")

    # 2. Cross-check against the split train.py wrote to disk during the real run.
    for split_name, by_class in splits.items():
        for c, ps in by_class.items():
            on_disk_dir = config.SPLIT_DIR / split_name / c
            if not on_disk_dir.is_dir():
                print(f"[warn] {on_disk_dir} missing: cannot cross-check {split_name}/{c}")
                continue
            on_disk = {f.name for f in on_disk_dir.iterdir()}
            ours = {p.name for p in ps}
            if split_name == "test":
                assert ours == on_disk, f"test/{c}: differs from data/splits ({len(ours ^ on_disk)} files)"
            else:
                # train/val on disk contain some cut-out swaps; compare original stems.
                stem = lambda n: n.replace("_final_masked", "").rsplit(".", 1)[0]  # noqa: E731
                assert {stem(n) for n in ours} == {stem(n) for n in on_disk}, \
                    f"{split_name}/{c}: differs from data/splits"
    print("[ok] Recomputed train/val/test split is identical to data/splits/ written by train.py.")

    # 3. Copy test images.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    total = 0
    for c, ps in test.items():
        out = OUT_DIR / c
        out.mkdir(parents=True)
        for p in ps:
            shutil.copy2(p, out / p.name)
        n = len(list(out.iterdir()))
        total += n
        print(f"  {c:<40} {n}")
    print(f"  {'TOTAL':<40} {total}")
    print(f"[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()
