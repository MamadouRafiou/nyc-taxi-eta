"""Chargement des courses et génération d'un jeu de données synthétique.

Deux sources coexistent, pour la même raison que dans tout projet qui
dépend de données externes : les tests et l'intégration continue ne doivent
pas télécharger 50 Mo à chaque exécution, ni échouer quand le serveur
distant est indisponible.

- `download_tlc` récupère les vraies courses publiées par la NYC Taxi &
  Limousine Commission.
- `generate_synthetic` fabrique un jeu de données de même forme, à partir
  d'un processus génératif documenté. Comme ce processus est connu, on
  connaît aussi le plancher d'erreur théorique : c'est ce qui permet de
  juger un modèle sans disposer des vraies données.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Colonnes du schéma TLC que le projet utilise réellement.
REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "VendorID",
    "passenger_count",
]


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Paramètres du processus génératif.

    Les valeurs par défaut reproduisent l'ordre de grandeur des courses
    new-yorkaises : distance médiane d'environ 1,7 mille, vitesse moyenne
    d'environ 12 milles à l'heure, plus lente aux heures de pointe.
    """

    n_rows: int = 100_000
    seed: int = 42
    base_speed_mph: float = 12.0
    # Multiplicateur de vitesse par heure de la journée (0 h à 23 h).
    # Creux marqués à 8-9 h et 17-18 h, circulation fluide la nuit.
    hourly_speed: tuple[float, ...] = (
        1.55, 1.65, 1.70, 1.70, 1.65, 1.50, 1.20, 0.90,
        0.70, 0.72, 0.85, 0.90, 0.88, 0.85, 0.80, 0.72,
        0.65, 0.60, 0.62, 0.75, 0.95, 1.15, 1.35, 1.50,
    )
    weekend_speed_bonus: float = 1.18
    fixed_overhead_s: float = 55.0
    # Bruit multiplicatif lognormal : la variabilité d'une course est
    # proportionnelle à sa durée, pas additive.
    noise_sigma: float = 0.28


def generate_synthetic_with_truth(
    spec: SyntheticSpec | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Génère des courses suivant un processus connu, et rend aussi la vérité.

    Le processus est le suivant :

        vitesse  = base * facteur_horaire * bonus_weekend
        durée    = overhead + 3600 * distance / vitesse       <- « vérité »
        observée = durée * lognormal(0, sigma)

    Toute l'information prédictive tient donc dans la distance, l'heure et
    le type de jour. Le bruit lognormal est irréductible : aucun modèle ne
    peut descendre en dessous de l'erreur qu'il impose.

    Retourne la table observée et le tableau des durées propres, ce qui
    permet de calculer exactement le plancher d'erreur.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n_rows

    # Distances : lognormale, écrêtée au domaine plausible.
    distance = np.clip(rng.lognormal(mean=0.55, sigma=0.85, size=n), 0.1, 60.0)

    start = np.datetime64("2024-01-01T00:00:00")
    offsets = rng.integers(0, 31 * 24 * 3600, size=n)
    pickup = start + offsets.astype("timedelta64[s]")

    hours = pickup.astype("datetime64[h]").astype(int) % 24
    weekday = (pickup.astype("datetime64[D]").astype(int) + 4) % 7  # 1970-01-01 = jeudi
    is_weekend = weekday >= 5

    speed = spec.base_speed_mph * np.asarray(spec.hourly_speed)[hours]
    speed = np.where(is_weekend, speed * spec.weekend_speed_bonus, speed)

    clean = spec.fixed_overhead_s + 3600.0 * distance / speed
    observed = clean * rng.lognormal(mean=0.0, sigma=spec.noise_sigma, size=n)
    observed = np.clip(observed, 30.0, 86_400.0)

    # L'arrondi à la seconde du couple (pickup, dropoff) est ce que verra le
    # modèle ; la vérité doit donc être arrondie de la même façon, sinon le
    # plancher calculé serait légèrement faux.
    observed_s = np.round(observed)
    dropoff = pickup + observed_s.astype("timedelta64[s]")

    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "trip_distance": distance.astype(np.float32),
            "VendorID": rng.choice([1, 2, 6], size=n, p=[0.45, 0.52, 0.03]),
            "passenger_count": rng.choice(
                [1, 2, 3, 4, 5], size=n, p=[0.70, 0.15, 0.07, 0.05, 0.03]
            ).astype(np.float32),
        }
    )
    return frame, clean


def generate_synthetic(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """Génère des courses synthétiques, sans exposer la vérité sous-jacente."""
    frame, _ = generate_synthetic_with_truth(spec)
    return frame


def irreducible_mae(spec: SyntheticSpec | None = None) -> float:
    """Erreur absolue moyenne minimale atteignable sur les données synthétiques.

    Le bruit appliqué étant multiplicatif et indépendant des attributs,
    aucun modèle ne peut le prédire. Le meilleur prédicteur possible rend
    la durée propre ; son erreur est donc la moyenne des écarts entre durée
    observée et durée propre. On la calcule directement, puisque le
    générateur connaît les deux.
    """
    spec = spec or SyntheticSpec()
    frame, clean = generate_synthetic_with_truth(spec)
    observed = (
        frame["tpep_dropoff_datetime"] - frame["tpep_pickup_datetime"]
    ).dt.total_seconds().to_numpy()
    return float(np.mean(np.abs(observed - clean)))


def download_tlc(year: int, month: int, destination: Path) -> Path:
    """Télécharge un mois de courses depuis le site de la NYC TLC."""
    import urllib.request

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    url = f"{TLC_BASE_URL}/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    urllib.request.urlretrieve(url, destination)  # noqa: S310
    return destination


def load_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Lit un fichier Parquet en ne chargeant que les colonnes utiles."""
    return pd.read_parquet(path, columns=columns or REQUIRED_COLUMNS)


def add_duration(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne cible `trip_duration`, en secondes."""
    frame = frame.copy()
    frame["trip_duration"] = (
        frame["tpep_dropoff_datetime"] - frame["tpep_pickup_datetime"]
    ).dt.total_seconds()
    return frame


def clean_trips(
    frame: pd.DataFrame,
    min_duration_s: float = 30.0,
    max_duration_s: float = 10_800.0,
    max_distance_mi: float = 100.0,
) -> pd.DataFrame:
    """Retire les courses aberrantes.

    Les bornes ne sont pas arbitraires. Une course de moins de 30 secondes
    est presque toujours une annulation enregistrée par erreur ; au-delà de
    trois heures, il s'agit de compteurs oubliés plutôt que de trajets
    réels. Garder ces lignes revient à entraîner le modèle sur du bruit de
    saisie, et à mesurer une erreur dominée par quelques valeurs absurdes.
    """
    if "trip_duration" not in frame.columns:
        frame = add_duration(frame)

    mask = (
        frame["trip_duration"].between(min_duration_s, max_duration_s)
        & frame["trip_distance"].between(0.05, max_distance_mi)
        & frame["trip_distance"].notna()
        & frame["VendorID"].notna()
    )
    return frame.loc[mask].reset_index(drop=True)
