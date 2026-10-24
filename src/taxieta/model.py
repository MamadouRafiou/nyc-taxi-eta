"""Entraînement, évaluation et persistance du modèle de durée."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureBuilder


TARGET = "trip_duration"


@dataclass(frozen=True, slots=True)
class Metrics:
    """Qualité d'un modèle sur un jeu de test."""

    mae: float
    rmse: float
    median_ae: float
    p90_ae: float
    n: int

    def summary(self) -> str:
        return (
            f"MAE {self.mae:.1f}s ({self.mae / 60:.2f} min) | "
            f"RMSE {self.rmse:.1f}s | "
            f"médiane {self.median_ae:.1f}s | "
            f"p90 {self.p90_ae:.1f}s | "
            f"{self.n:,} courses"
        )

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Calcule les indicateurs d'erreur.

    La MAE est l'indicateur principal : elle s'exprime en secondes, donc se
    lit sans conversion, et elle est moins sensible que la RMSE aux courses
    extrêmes — nombreuses dans ce jeu de données. La médiane et le 90e
    centile complètent le tableau : une MAE correcte peut cacher une queue
    de prédictions très mauvaises, ce qu'un service de réservation ressent
    immédiatement.
    """
    errors = np.abs(np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64"))
    squared = (np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")) ** 2
    return Metrics(
        mae=float(errors.mean()),
        rmse=float(np.sqrt(squared.mean())),
        median_ae=float(np.median(errors)),
        p90_ae=float(np.percentile(errors, 90)),
        n=int(len(errors)),
    )


def build_pipeline(
    estimator: str = "hgb",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Assemble le pipeline complet, de la course brute à la prédiction.

    Le pipeline contient la dérivation des attributs, leur mise à l'échelle,
    l'encodage des catégories et le modèle. Un seul objet est donc
    sérialisé, et `predict` accepte directement des lignes au schéma TLC.

    `handle_unknown="ignore"` sur l'encodage one-hot n'est pas un détail :
    si un nouveau `VendorID` apparaît en production, l'encodeur produit un
    vecteur de zéros au lieu de lever une exception. La prédiction perd un
    peu en qualité, le service continue de répondre.
    """
    if estimator == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.1,
            max_depth=None,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=random_state,
        )
    elif estimator == "rf":
        model = RandomForestRegressor(
            n_estimators=60,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(
            f"estimateur inconnu : {estimator!r}. Valeurs acceptées : 'hgb', 'rf'."
        )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def train(
    frame: pd.DataFrame,
    estimator: str = "hgb",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, Metrics, Metrics]:
    """Entraîne le modèle et renvoie ses performances.

    Retourne le pipeline ajusté, les métriques du modèle, et celles d'une
    référence naïve. Sans référence, une MAE n'a aucune signification : il
    faut savoir ce qu'on obtiendrait sans modèle du tout.
    """
    from sklearn.model_selection import train_test_split

    if TARGET not in frame.columns:
        raise ValueError(f"colonne cible {TARGET!r} absente")

    y = frame[TARGET].to_numpy(dtype="float64")
    X = frame.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(estimator=estimator, random_state=random_state)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(y_test, pipeline.predict(X_test))
    # Référence : prédire systématiquement la durée médiane d'entraînement.
    baseline = evaluate(y_test, np.full_like(y_test, np.median(y_train)))

    return pipeline, metrics, baseline


def save_model(pipeline: Pipeline, path: Path) -> Path:
    """Sérialise le pipeline sur disque."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path) -> Pipeline:
    """Relit un pipeline sérialisé."""
    import joblib

    return joblib.load(Path(path))
