
from pathlib import Path

import torch


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# NAIM repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CKD dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "tabular"
    / "classification"
    / "ckd"
    / "ckd.csv"
)

# All experiment outputs will be saved here
OUTPUT_DIR = PROJECT_ROOT / "outputs_ckd_modular"


# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

# Random seeds used for repeated experiments
SEED_LIST = [42, 123, 2026]

# Additional MCAR missingness rates
MISSING_RATES = [0.10, 0.30, 0.50]

# Target column
TARGET = "class"


# --------------------------------------------------
# Dataset columns
# --------------------------------------------------

NUMERIC_COLUMNS = [
    "age",
    "bp",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
]

CATEGORICAL_COLUMNS = [
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]


# --------------------------------------------------
# NAIM training settings
# --------------------------------------------------

DEVICE = torch.device("cpu")

NAIM_MAX_EPOCHS = 50
NAIM_PATIENCE = 10
NAIM_BATCH_SIZE = 32
NAIM_LEARNING_RATE = 1e-3

NAIM_D_TOKEN = 6
NAIM_NUM_HEADS = 3
NAIM_FEEDFORWARD_DIM = 128
NAIM_DROPOUT_RATE = 0.1
NAIM_NUM_LAYERS = 2
