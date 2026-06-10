"""
Train lead conversion models and persist the best performing pipeline.

The script expects the assessment datasets at:
    data/leads.csv
    data/interactions.csv

It builds lead-level behavioral features, evaluates multiple classifiers,
stores the best model in model.pkl, writes metrics to outputs/model_metrics.json,
and saves a feature importance chart when the selected estimator exposes
feature importances.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from utils import save_json, setup_logging

logger = setup_logging()


NUMERIC_FEATURES = [
    "pages_visited",
    "time_spent_minutes",
    "pricing_views",
    "session_count",
    "days_since_first_visit",
    "days_since_last_visit",
    "avg_scroll_depth",
    "unique_pages",
    "active_days",
]

# FIX #4: device lives in leads.csv as "device_type"; added "industry" and "job_role"
# for richer lead-profile signal available without leakage
CATEGORICAL_FEATURES = ["source", "company_size", "lead_segment", "device_type"]
TARGET = "converted"

# Funnel stage ordinal mapping (mirrors EDA cell 18)
_FUNNEL_ORDER = {"Awareness": 1, "Consideration": 2, "Evaluation": 3, "Decision": 4}


def _require_data_files() -> None:
    """Fail early with a clear message when raw CSV files are absent."""
    missing = [
        path
        for path in (Path(config.LEADS_CSV), Path(config.INTERACTIONS_CSV))
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required data file(s): {missing_text}. "
            "Place the assessment CSVs under data/ before running training."
        )


def _contains_value(series: pd.Series, pattern: str) -> pd.Series:
    """Case-insensitive containment helper that handles missing text."""
    return series.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def _safe_day_delta(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Return non-negative day differences with nulls filled as zero."""
    delta = (later - earlier).dt.total_seconds().div(86400)
    return delta.clip(lower=0).fillna(0)


def _derive_converted(leads: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    FIX #1: leads.csv has no 'converted' column — derive it from interactions.

    A lead is converted if any of their interactions contain:
      - a high-intent event name  (demo_request, free_trial_start, contact_form_submit)
      - OR a completed form       (form_completed == True)
    """
    high_intent_events = {"demo_request", "free_trial_start", "contact_form_submit"}
    converted_event = interactions["event_name"].isin(high_intent_events)
    converted_form = interactions["form_completed"] == True  # noqa: E712
    converted_lead_ids = set(
        interactions.loc[converted_event | converted_form, "lead_id"]
    )
    leads = leads.copy()
    leads[TARGET] = leads["lead_id"].isin(converted_lead_ids).astype(int)
    logger.info(
        "Derived conversion labels — rate: %.1f%%",
        leads[TARGET].mean() * 100,
    )
    return leads


def build_features(leads: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per lead using only attributes available before conversion.

    Interaction-level converted events are used only to build the target label
    (via _derive_converted); they are not used as input features.
    """
    # FIX #1: derive the target before any feature work
    leads = _derive_converted(leads, interactions)
    interactions = interactions.copy()

    if "lead_id" not in leads or "lead_id" not in interactions:
        raise ValueError("Both datasets must include a lead_id column.")

    leads["created_at"] = pd.to_datetime(leads.get("created_at"), errors="coerce")
    interactions["timestamp"] = pd.to_datetime(
        interactions.get("timestamp"), errors="coerce"
    )

    # --- event flag columns (text matching on available columns) ---
    page_text = interactions.get("page_name", pd.Series("", index=interactions.index))
    event_name = interactions.get("event_name", pd.Series("", index=interactions.index))
    event_type = interactions.get("event_type", pd.Series("", index=interactions.index))
    combined_event_text = (
        page_text.fillna("").astype(str)
        + " "
        + event_name.fillna("").astype(str)
        + " "
        + event_type.fillna("").astype(str)
    )

    interactions["is_demo_request"] = _contains_value(combined_event_text, "demo")
    interactions["is_pricing_view"] = _contains_value(combined_event_text, "pricing")

    # FIX #2: actual column is "session_duration_seconds", not "duration_seconds"
    duration_col = "session_duration_seconds"
    if duration_col not in interactions.columns:
        interactions[duration_col] = 0
    interactions[duration_col] = pd.to_numeric(
        interactions[duration_col], errors="coerce"
    ).clip(lower=0)

    # FIX #3: actual column is "scroll_depth_percent", not "scroll_depth"
    scroll_col = "scroll_depth_percent"
    if scroll_col not in interactions.columns:
        interactions[scroll_col] = 0
    interactions[scroll_col] = pd.to_numeric(
        interactions[scroll_col], errors="coerce"
    ).clip(lower=0, upper=100)

    if "session_id" not in interactions:
        interactions["session_id"] = interactions["lead_id"].astype(str) + "_session"

    # FIX #8: encode funnel_stage as an ordinal and take the max per lead
    interactions["funnel_order"] = (
        interactions["funnel_stage"].map(_FUNNEL_ORDER).fillna(0).astype(int)
    )

    # FIX #6: removed whatsapp_clicks and email_opens (columns don't exist).
    #         Replaced with form_completions, which is directly available.
    interactions["is_form_completed"] = interactions["form_completed"].fillna(False).astype(int)

    aggregations: dict[str, tuple[str, str]] = {
        "pages_visited": ("interaction_id", "count")
        if "interaction_id" in interactions
        else ("lead_id", "size"),
        "time_spent_minutes": (duration_col, "sum"),
        "avg_scroll_depth": (scroll_col, "mean"),        # FIX #3
        "unique_pages": ("page_name", "nunique")
        if "page_name" in interactions
        else ("lead_id", "size"),
        "demo_requests": ("is_demo_request", "sum"),
        "pricing_views": ("is_pricing_view", "sum"),
        "form_completions": ("is_form_completed", "sum"),  # FIX #6
        "session_count": ("session_id", "nunique"),
        "max_funnel_stage": ("funnel_order", "max"),       # FIX #8
        "first_interaction_at": ("timestamp", "min"),
        "last_interaction_at": ("timestamp", "max"),
    }

    features = interactions.groupby("lead_id", dropna=False).agg(**aggregations)
    features["time_spent_minutes"] = features["time_spent_minutes"].div(60)
    features["active_days"] = _safe_day_delta(
        features["last_interaction_at"], features["first_interaction_at"]
    )

    # FIX #4 & #5: correct column names are "device_type" and "lead_segment"
    lead_columns = [
        col
        for col in [
            "lead_id",
            "source",
            "company_size",
            "lead_segment",     # FIX #5: was "segment"
            "created_at",
            TARGET,
        ]
        if col in leads.columns
    ]
    dataset = leads[lead_columns].merge(features, on="lead_id", how="left")
    dataset["days_since_first_visit"] = _safe_day_delta(
        dataset["first_interaction_at"], dataset["created_at"]
    )
    dataset["days_since_last_visit"] = _safe_day_delta(
        dataset["last_interaction_at"], dataset["created_at"]
    )

    # FIX #4: device lives in leads.csv as "device_type", not in interactions
    if "device_type" not in dataset.columns:
        if "device_type" in interactions.columns:
            primary_device = (
                interactions.groupby("lead_id")["device_type"]
                .agg(
                    lambda values: values.mode(dropna=True).iloc[0]
                    if not values.mode(dropna=True).empty
                    else "unknown"
                )
                .rename("device_type")
            )
            dataset = dataset.merge(primary_device, on="lead_id", how="left")
        else:
            dataset["device_type"] = "unknown"

    for column in NUMERIC_FEATURES:
        if column not in dataset:
            dataset[column] = 0
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").fillna(0)

    for column in CATEGORICAL_FEATURES:
        if column not in dataset:
            dataset[column] = "unknown"
        dataset[column] = dataset[column].fillna("unknown").astype(str)

    dataset[TARGET] = pd.to_numeric(dataset[TARGET], errors="coerce").fillna(0).astype(int)
    return dataset[["lead_id", TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _make_preprocessor() -> ColumnTransformer:
    """Create the preprocessing graph shared by all candidate models."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def _candidate_models() -> dict[str, Any]:
    """Return model candidates; XGBoost is included when installed."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        )
    except ImportError:
        logger.warning("xgboost is not installed; skipping XGBoost candidate.")

    return models


def _evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate classification metrics for a fitted pipeline."""
    predictions = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
    else:
        probabilities = predictions

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def _save_feature_importance(best_pipeline: Pipeline) -> None:
    """Persist a feature importance chart for tree-based models."""
    estimator = best_pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        logger.info("Selected model has no feature_importances_; skipping chart.")
        return

    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    Path(config.FEATURE_IMPORTANCE_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#2563eb")
    plt.title("Top Lead Conversion Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_PATH, dpi=160)
    plt.close()


def train() -> dict[str, Any]:
    """Run the full training workflow and return selected model metrics."""
    _require_data_files()
    leads = pd.read_csv(config.LEADS_CSV)
    interactions = pd.read_csv(config.INTERACTIONS_CSV)
    dataset = build_features(leads, interactions)

    x = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = dataset[TARGET]
    stratify = y if y.nunique() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )

    all_results: dict[str, dict[str, float]] = {}
    best_name = ""
    best_score = -np.inf
    best_pipeline: Pipeline | None = None

    for name, estimator in _candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor()),
                ("model", estimator),
            ]
        )
        logger.info("Training %s", name)
        pipeline.fit(x_train, y_train)
        metrics = _evaluate_model(pipeline, x_test, y_test)
        all_results[name] = metrics
        selection_score = metrics["f1"] + metrics["auc_roc"]
        if selection_score > best_score:
            best_score = selection_score
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model candidates were trained successfully.")

    final_metrics = {"best_model": best_name, **all_results[best_name], "all_models": all_results}

    with open(config.MODEL_PATH, "wb") as model_file:
        pickle.dump(
            {
                "pipeline": best_pipeline,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "metrics": final_metrics,
            },
            model_file,
        )

    save_json(final_metrics, config.METRICS_PATH)
    _save_feature_importance(best_pipeline)
    logger.info("Best model: %s", best_name)
    logger.info(json.dumps(final_metrics, indent=2))
    return final_metrics


if __name__ == "__main__":
    train()