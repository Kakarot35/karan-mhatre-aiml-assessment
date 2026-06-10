"""
Utility module for the Lead Conversion Prediction System.

Provides reusable helpers for logging setup, data loading, missing
value handling, categorical encoding, confidence/risk classification,
JSON I/O, and model deserialization.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config


# Logging
def setup_logging(log_level: str = config.LOG_LEVEL) -> logging.Logger:
    """
    Configure application-wide logging and return a project logger.

    Parameters
    ----------
    log_level : str, optional
        Logging level (e.g. "DEBUG", "INFO", "WARNING"). Defaults to
        the value defined in ``config.LOG_LEVEL``.

    Returns
    -------
    logging.Logger
        Configured logger named after the project.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=config.LOG_FORMAT,
    )
    logger = logging.getLogger("lead_conversion")
    logger.debug("Logging configured at level %s", log_level.upper())
    return logger


# Module-level logger for internal use by utility functions.
logger = setup_logging()


# Data loading
def load_data(
    leads_path: str = config.LEADS_CSV,
    interactions_path: str = config.INTERACTIONS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the leads and interactions CSV files and validate required columns.

    Parameters
    ----------
    leads_path : str
        Filesystem path to the leads CSV.
    interactions_path : str
        Filesystem path to the interactions CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The loaded leads and interactions DataFrames.

    Raises
    ------
    FileNotFoundError
        If either CSV file cannot be located.
    ValueError
        If required columns are missing from either dataset.
    """
    try:
        leads_df = pd.read_csv(leads_path)
        interactions_df = pd.read_csv(interactions_path)
    except FileNotFoundError as exc:
        logger.error("CSV file not found: %s", exc.filename)
        raise

    required_lead_cols = {"lead_id"}
    required_interaction_cols = {"lead_id"}
    missing_lead_cols = required_lead_cols - set(leads_df.columns)
    missing_interaction_cols = required_interaction_cols - set(
        interactions_df.columns
    )

    if missing_lead_cols or missing_interaction_cols:
        missing = missing_lead_cols | missing_interaction_cols
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(
        "Loaded leads=%d rows, interactions=%d rows",
        len(leads_df),
        len(interactions_df),
    )
    return leads_df, interactions_df



# Missing value handling

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame.

    Numerical columns are filled with their median; categorical
    columns are filled with their mode. The cleaned DataFrame is
    returned as a copy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential nulls.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing values imputed.
    """
    cleaned_df = df.copy()
    total_missing = int(cleaned_df.isnull().sum().sum())
    logger.info("Total missing values before cleaning: %d", total_missing)

    if total_missing == 0:
        logger.info("No missing values detected; returning input unchanged.")
        return cleaned_df

    categorical_cols = cleaned_df.select_dtypes(include=["object", "category"]).columns
    numerical_cols = cleaned_df.select_dtypes(include=["number"]).columns

    for col in numerical_cols:
        if cleaned_df[col].isnull().any():
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)
            logger.debug("Filled %d nulls in '%s' with median %.4f",
                          int(df[col].isnull().sum()), col, median_value)

    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            mode_series = cleaned_df[col].mode(dropna=True)
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                logger.debug("Filled %d nulls in '%s' with mode '%s'",
                              int(df[col].isnull().sum()), col, mode_value)

    remaining = int(cleaned_df.isnull().sum().sum())
    logger.info("Missing values after cleaning: %d", remaining)
    return cleaned_df



# Categorical encoding

def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str] = config.CATEGORICAL_FEATURES,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode the specified categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the categorical columns.
    categorical_cols : list[str], optional
        Columns to encode. Defaults to ``config.CATEGORICAL_FEATURES``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        The encoded DataFrame (as a copy) and a mapping of column name
        to the fitted :class:`~sklearn.preprocessing.LabelEncoder`.
    """
    encoded_df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        if col not in encoded_df.columns:
            logger.warning("Column '%s' not in DataFrame; skipping.", col)
            continue
        try:
            encoder = LabelEncoder()
            encoded_df[col] = encoder.fit_transform(encoded_df[col].astype(str))
            encoders[col] = encoder
            logger.debug("Encoded categorical column '%s' (%d classes).",
                         col, len(encoder.classes_))
        except Exception as exc:
            logger.exception("Failed to encode column '%s': %s", col, exc)
            raise

    logger.info("Encoded %d categorical columns.", len(encoders))
    return encoded_df, encoders



# Confidence and risk classification

def get_confidence_level(probability: float) -> str:
    """
    Map a conversion probability to a confidence bucket.

    Parameters
    ----------
    probability : float
        Predicted probability in the range [0.0, 1.0].

    Returns
    -------
    str
        ``"high"`` if ``probability > 0.7``,
        ``"medium"`` if ``0.4 <= probability <= 0.7``,
        ``"low"`` otherwise.
    """
    if probability > config.CONFIDENCE_HIGH:
        return "high"
    if probability >= config.CONFIDENCE_LOW:
        return "medium"
    return "low"


def get_risk_level(probability: float) -> str:
    """
    Map a conversion probability to a non-conversion risk bucket.

    Parameters
    ----------
    probability : float
        Predicted probability of conversion in [0.0, 1.0].

    Returns
    -------
    str
        ``"low"`` if ``probability > 0.7`` (likely to convert),
        ``"medium"`` if ``0.4 <= probability <= 0.7``,
        ``"high"`` if ``probability < 0.4`` (unlikely to convert).
    """
    if probability > config.CONFIDENCE_HIGH:
        return "low"
    if probability >= config.CONFIDENCE_LOW:
        return "medium"
    return "high"



# JSON I/O

def save_json(data: dict[str, Any], path: str) -> None:
    """
    Persist a dictionary as a JSON file, creating parent directories as needed.

    Parameters
    ----------
    data : dict[str, Any]
        Serialisable Python dictionary.
    path : str
        Destination file path.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        logger.info("Saved JSON to %s", target)
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("Failed to save JSON to %s: %s", path, exc)
        raise



# Model deserialization

def load_model(model_path: str = config.MODEL_PATH) -> Any:
    """
    Load a pickled model from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to the pickled model. Defaults to ``config.MODEL_PATH``.

    Returns
    -------
    Any
        The deserialised model object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    RuntimeError
        If the file exists but cannot be unpickled.
    """
    if not os.path.exists(model_path):
        logger.error("Model file not found at %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)
        logger.info("Loaded model from %s", model_path)
        return model
    except (pickle.UnpicklingError, EOFError, ImportError) as exc:
        logger.exception("Failed to unpickle model at %s: %s", model_path, exc)
        raise RuntimeError(f"Could not load model from {model_path}") from exc
