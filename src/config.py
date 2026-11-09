"""
Central configuration for the corn leaf disease classifier.

WHY a config file: every other module (data loading, training, the app) needs the
same class names, image size and paths. If each file hard-coded its own copy,
changing one value (e.g. image size) would silently break the others. One source
of truth avoids that whole class of bug.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# WHY: splitting and shuffling are random. Fixing the seed means you (and anyone
# reading your README) get the exact same train/val/test split every run, so
# results are comparable between experiments.
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# WHY Path(__file__): makes paths relative to the project, not to wherever you
# happened to launch Python from. No absolute "C:\Users\..." paths, which would
# break on any other machine (including Hugging Face Spaces).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # original images, one folder per class
SPLIT_DIR = DATA_DIR / "splits"     # train/val/test folders built by data.py
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# WHY an explicit ordered list: the model outputs a vector of 3 probabilities,
# and index i of that vector means CLASS_NAMES[i]. If the order ever differed
# between training and the app, every prediction would be silently mislabelled.
# These must match the folder names inside data/raw/ exactly.
CLASS_NAMES = [
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]

# Human-friendly names for charts and the app UI.
DISPLAY_NAMES = {
    "Corn_(maize)___Common_rust_": "Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy",
}

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# WHY 224: MobileNetV2's ImageNet weights were trained at 224x224. Feeding the
# size the weights were learned at gives the best transfer. (160 would roughly
# halve CPU training time at a small accuracy cost; 96/128/160/192/224 are the
# officially supported sizes.)
IMG_SIZE = 224
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Split ratios
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# WHY a block size: consecutive camera numbers (e.g. "RS_NLB 3700" and
# "RS_NLB 3701") are photos taken seconds apart and are sometimes the same leaf
# from a slightly different angle. Grouping every 5 consecutive photo numbers
# keeps those neighbours on the same side of the split. It costs nothing and
# closes a leakage path we can't fully rule out by looking.
PHOTO_BLOCK = 5

# ---------------------------------------------------------------------------
# Background fix (see data.fill_background)
# ---------------------------------------------------------------------------
BLACK_THRESHOLD = 25        # a pixel is "black" if every RGB channel is below this
BLACK_MIN_FRACTION = 0.05   # only treat an image as black-background above 5% black
BOARD_RGB = [135.0, 130.0, 136.0]  # median colour of the grey board in blight photos
SEGMENTED_DIR = DATA_DIR / "kaggle" / "segmented"  # PlantVillage's cut-out versions

# Run 1 showed that replacing the black background was not enough: rust images
# are still the only "leaf cut out onto a flat background" images, and the model
# learned that look as "rust" (62% of blight leaves shown that way were called
# rust). Fix: in train and val, swap this fraction of blight images for their
# cut-out version, so the cut-out look appears in BOTH disease classes and stops
# being a clue. The test set keeps the original photos.
CUTOUT_CLASS = "Corn_(maize)___Northern_Leaf_Blight"
CUTOUT_FRACTION = 0.5
CUTOUT_SPLITS = ("train", "val")

# ---------------------------------------------------------------------------
# Training (see train.py)
# ---------------------------------------------------------------------------
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5     # 100x smaller: nudge the pretrained weights, don't overwrite them
FINE_TUNE_FROM = 100 # layers[:100] stay frozen in stage 2
EARLY_STOP_PATIENCE = 3
