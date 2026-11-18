"""The grouped split is the project's defence against leakage. These tests use
synthetic file names (no dataset needed), so they run in CI."""
from pathlib import Path

import pytest

from src import config
from src.data import check_no_group_overlap, group_key, group_split, parse_photo_id


def fake_images(n_photos=200):
    """PlantVillage-style names, including ' copy N' crops of the same photo."""
    images = {}
    for c, series in zip(config.CLASS_NAMES, ["RS_Rust", "RS_NLB", "R.S_HL"]):
        paths = []
        for i in range(1000, 1000 + n_photos):
            paths.append(Path(f"uuid{i}___{series} {i}.JPG"))
            if i % 3 == 0:
                paths.append(Path(f"uuidb{i}___{series} {i} copy.jpg"))
                paths.append(Path(f"uuidc{i}___{series} {i} copy 2.jpg"))
        images[c] = sorted(paths)
    return images


@pytest.mark.parametrize("name, expected", [
    ("RS_Rust 1563.JPG", ("RS_Rust", 1563)),
    ("0079c731-aaaa___RS_NLB 4121.JPG", ("RS_NLB", 4121)),
    ("00031d74-bbbb___R.S_HL 8315 copy 2.jpg", ("R.S_HL", 8315)),
    ("00031d74-cccc___R.S_HL 8315 copy.jpg", ("R.S_HL", 8315)),
])
def test_parse_photo_id(name, expected):
    assert parse_photo_id(name) == expected


def test_crops_and_neighbouring_photos_share_a_group():
    assert group_key("a___RS_NLB 3700.JPG") == group_key("b___RS_NLB 3704 copy 2.jpg")
    assert group_key("a___RS_NLB 3704.JPG") != group_key("a___RS_NLB 3705.JPG")


def test_no_file_or_group_in_two_splits():
    splits = group_split(fake_images())
    check_no_group_overlap(splits)  # raises on leakage
    seen = {}
    for split_name, by_class in splits.items():
        for c, paths in by_class.items():
            for p in paths:
                assert seen.setdefault((c, p.name), split_name) == split_name
                seen_group = (c, group_key(p.name))
                assert seen.setdefault(seen_group, split_name) == split_name


def test_every_file_assigned_exactly_once():
    images = fake_images()
    splits = group_split(images)
    for c, paths in images.items():
        assigned = [p for s in splits.values() for p in s[c]]
        assert sorted(assigned) == paths


def test_split_is_deterministic_and_seed_dependent():
    a, b = group_split(fake_images()), group_split(fake_images())
    assert a == b
    assert group_split(fake_images(), seed=config.SEED + 1) != a


def test_split_ratios_roughly_70_15_15():
    splits = group_split(fake_images())
    for c in config.CLASS_NAMES:
        total = sum(len(splits[s][c]) for s in splits)
        assert abs(len(splits["train"][c]) / total - 0.70) < 0.05
        assert abs(len(splits["test"][c]) / total - 0.15) < 0.05


def test_leak_detector_catches_a_leak():
    leaky = {"train": {"x": [Path("a___RS_NLB 3700.JPG")]}, "val": {"x": []},
             "test": {"x": [Path("b___RS_NLB 3701.JPG")]}}
    with pytest.raises(AssertionError, match="Leak"):
        check_no_group_overlap(leaky)
