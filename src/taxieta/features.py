"""Construction des attributs à partir des courses brutes.

Le projet d'origine n'utilisait que deux attributs : la distance et
l'identifiant du fournisseur. C'est insuffisant, parce que la durée d'un
trajet urbain dépend surtout de *quand* il a lieu : cinq kilomètres à
8 h un mardi et les mêmes cinq kilomètres à 3 h un dimanche n'ont rien à
voir.

Ce module ajoute donc le temps, et le code de façon à ce qu'un modèle
puisse l'exploiter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NUMERIC_FEATURES = [
    "trip_distance",
    "log_distance",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "passenger_count",
]
CATEGORICAL_FEATURES = ["VendorID"]


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Dérive les attributs temporels et dérivés à partir des colonnes brutes.

    Deux choix méritent justification.

    **L'heure est encodée en sinus et cosinus.** Utilisée telle quelle,
    l'heure 23 et l'heure 0 sont séparées de 23 unités alors qu'une minute
    les sépare. Projetées sur un cercle, elles redeviennent voisines, ce qui
    correspond à la réalité de la circulation.

    **La distance est doublée d'un logarithme.** Sa distribution est très
    asymétrique : une majorité de courses courtes et une longue traîne. Le
    logarithme donne au modèle une variable mieux répartie, sans retirer la
    distance brute qui reste directement proportionnelle à la durée.
    """
    out = pd.DataFrame(index=frame.index)

    distance = frame["trip_distance"].astype("float64")
    out["trip_distance"] = distance
    out["log_distance"] = np.log1p(distance)

    pickup = pd.to_datetime(frame["tpep_pickup_datetime"])
    hour = pickup.dt.hour + pickup.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    day_of_week = pickup.dt.dayofweek
    out["day_of_week"] = day_of_week.astype("float64")
    out["is_weekend"] = (day_of_week >= 5).astype("float64")

    passengers = frame.get("passenger_count")
    if passengers is None:
        out["passenger_count"] = 1.0
    else:
        out["passenger_count"] = (
            pd.to_numeric(passengers, errors="coerce").fillna(1.0).astype("float64")
        )

    out["VendorID"] = frame["VendorID"].astype("float64").fillna(-1).astype(int).astype(str)

    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Enveloppe `derive_features` pour l'insérer dans un pipeline scikit-learn.

    Mettre la dérivation des attributs *dans* le pipeline, plutôt que de
    l'appliquer à part, supprime une classe entière de bogues : celle où
    l'entraînement et l'inférence ne calculent pas tout à fait les mêmes
    colonnes. Un seul objet sérialisé contient la recette complète.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureBuilder":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return derive_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)
