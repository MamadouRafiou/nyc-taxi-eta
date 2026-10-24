"""Prédiction de la durée des courses de taxi, avec inférence distribuée."""

from .data import (
    SyntheticSpec,
    add_duration,
    clean_trips,
    generate_synthetic,
    generate_synthetic_with_truth,
    irreducible_mae,
    load_parquet,
)
from .features import derive_features, FeatureBuilder
from .inference import (
    InferenceReport,
    predict_dask,
    predict_parquet_dask,
    predict_partition,
    predict_sequential,
)
from .model import Metrics, build_pipeline, evaluate, load_model, save_model, train

__version__ = "1.0.0"

__all__ = [
    "SyntheticSpec",
    "generate_synthetic",
    "generate_synthetic_with_truth",
    "irreducible_mae",
    "add_duration",
    "clean_trips",
    "load_parquet",
    "derive_features",
    "FeatureBuilder",
    "build_pipeline",
    "train",
    "evaluate",
    "Metrics",
    "save_model",
    "load_model",
    "predict_partition",
    "predict_sequential",
    "predict_dask",
    "predict_parquet_dask",
    "InferenceReport",
]
