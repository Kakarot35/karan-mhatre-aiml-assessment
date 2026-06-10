"""
Configuration module for the Lead Conversion Prediction System.

Centralizes all project-wide constants including file paths, model
hyperparameters, feature definitions, API settings, and logging
configuration. Also loads the Gemini API key from a local .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file (if present) at the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# Base directory for raw and processed lead data.
DATA_DIR = "data/"

# Raw input datasets.
LEADS_CSV = "data/leads.csv"
INTERACTIONS_CSV = "data/interactions.csv"

# Trained model artifact location.
MODEL_PATH = "model.pkl"

# Output artifacts produced during training and evaluation.
METRICS_PATH = "outputs/model_metrics.json"
FEATURE_IMPORTANCE_PATH = "outputs/feature_importance.png"

# ------------------------------------------------------------------
# Model settings
# ------------------------------------------------------------------
# Fraction of the dataset reserved for evaluation.
TEST_SIZE = 0.2

# Seed for all stochastic operations to ensure reproducibility.
RANDOM_STATE = 42

# Number of folds used in cross-validation.
CV_FOLDS = 5

# ------------------------------------------------------------------
# Feature lists
# ------------------------------------------------------------------
# Categorical columns that require encoding (one-hot or ordinal).
CATEGORICAL_FEATURES = [
    "source",
    "company_size",
    "segment",
    "device",
]

# Numerical features used directly by the model.
NUMERICAL_FEATURES = [
    "total_sessions",
    "total_duration",
    "avg_scroll_depth",
    "total_pages_visited",
    "demo_requests",
    "pricing_views",
    "days_since_first_visit",
    "days_since_last_visit",
    "session_count",
    "funnel_stage_max",
]

# Target column for the supervised learning task.
TARGET_COLUMN = "converted"

# ------------------------------------------------------------------
# API settings
# ------------------------------------------------------------------
# FastAPI metadata exposed through the OpenAPI schema.
API_TITLE = "Lead Conversion Prediction API"
API_VERSION = "1.0.0"

# Probability thresholds used to label prediction confidence.
CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.4

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
# Standard structured log format with timestamp, logger, level, and message.
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# ------------------------------------------------------------------
# External services
# ------------------------------------------------------------------
# Gemini API key loaded from the environment (e.g. via .env).
# Falls back to None if not set so the rest of the system can
# fail gracefully on startup rather than at first use.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
