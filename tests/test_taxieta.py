"""Suite de tests hermétique : aucun téléchargement, aucun accès réseau.

Les données proviennent du générateur synthétique, dont le processus est
connu. Cela permet de tester non seulement que le code s'exécute, mais
qu'il apprend réellement quelque chose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxieta import (
    SyntheticSpec,
    add_duration,
    build_pipeline,
    clean_trips,
    derive_features,
    evaluate,
    generate_synthetic,
    generate_synthetic_with_truth,
    irreducible_mae,
    load_model,
    predict_dask,
    predict_partition,
    predict_sequential,
    save_model,
    train,
)
from taxieta.inference import PREDICTION_COLUMN


SMALL = SyntheticSpec(n_rows=4_000, seed=7)


@pytest.fixture(scope="module")
def trips() -> pd.DataFrame:
    return clean_trips(add_duration(generate_synthetic(SMALL)))


@pytest.fixture(scope="module")
def fitted(trips):
    pipeline, metrics, baseline = train(trips, estimator="hgb")
    return pipeline, metrics, baseline


# --------------------------------------------------------------------------
# Génération de données
# --------------------------------------------------------------------------


def test_schema_attendu():
    frame = generate_synthetic(SMALL)
    for column in (
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "VendorID",
        "passenger_count",
    ):
        assert column in frame.columns
    assert len(frame) == SMALL.n_rows


def test_generation_deterministe():
    a = generate_synthetic(SMALL)
    b = generate_synthetic(SMALL)
    pd.testing.assert_frame_equal(a, b)


def test_les_graines_differentes_donnent_des_donnees_differentes():
    a = generate_synthetic(SyntheticSpec(n_rows=500, seed=1))
    b = generate_synthetic(SyntheticSpec(n_rows=500, seed=2))
    assert not a["trip_distance"].equals(b["trip_distance"])


def test_dropoff_toujours_apres_pickup():
    frame = generate_synthetic(SMALL)
    assert (frame["tpep_dropoff_datetime"] > frame["tpep_pickup_datetime"]).all()


def test_les_heures_de_pointe_sont_plus_lentes():
    """Le processus génératif doit bien produire le signal qu'on veut apprendre."""
    frame = add_duration(generate_synthetic(SyntheticSpec(n_rows=40_000, seed=3)))
    frame["hour"] = frame["tpep_pickup_datetime"].dt.hour
    frame["speed"] = frame["trip_distance"] / (frame["trip_duration"] / 3600.0)
    pointe = frame.loc[frame["hour"].isin([8, 17, 18]), "speed"].median()
    nuit = frame.loc[frame["hour"].isin([2, 3, 4]), "speed"].median()
    assert nuit > pointe * 1.5


def test_plancher_derreur_coherent():
    floor = irreducible_mae(SMALL)
    assert 0 < floor < 2000


def test_la_verite_est_plus_lisse_que_lobserve():
    frame, clean = generate_synthetic_with_truth(SMALL)
    observed = (
        frame["tpep_dropoff_datetime"] - frame["tpep_pickup_datetime"]
    ).dt.total_seconds().to_numpy()
    assert len(clean) == len(observed)
    assert np.std(clean) < np.std(observed)


# --------------------------------------------------------------------------
# Nettoyage
# --------------------------------------------------------------------------


def test_le_nettoyage_retire_les_aberrations():
    base = generate_synthetic(SyntheticSpec(n_rows=100, seed=11))
    frame = add_duration(base)
    # Une course de 10 heures et une distance négative.
    frame.loc[0, "trip_duration"] = 36_000.0
    frame.loc[1, "trip_distance"] = -5.0
    cleaned = clean_trips(frame)
    assert len(cleaned) == len(frame) - 2
    assert cleaned["trip_duration"].between(30, 10_800).all()


def test_le_nettoyage_calcule_la_duree_si_absente():
    frame = generate_synthetic(SyntheticSpec(n_rows=50, seed=12))
    assert "trip_duration" not in frame.columns
    assert "trip_duration" in clean_trips(frame).columns


def test_le_nettoyage_retire_les_vendorid_manquants():
    frame = add_duration(generate_synthetic(SyntheticSpec(n_rows=50, seed=13)))
    frame = frame.astype({"VendorID": "float64"})
    frame.loc[0, "VendorID"] = np.nan
    assert len(clean_trips(frame)) == len(frame) - 1


# --------------------------------------------------------------------------
# Attributs
# --------------------------------------------------------------------------


def test_lheure_est_encodee_de_facon_cyclique():
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01 23:59:00", "2024-01-02 00:01:00", "2024-01-02 12:00:00"]
            ),
            "trip_distance": [1.0, 1.0, 1.0],
            "VendorID": [1, 1, 1],
            "passenger_count": [1, 1, 1],
        }
    )
    f = derive_features(frame)
    minuit_moins = f.loc[0, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    minuit_plus = f.loc[1, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    midi = f.loc[2, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)

    # 23h59 et 00h01 doivent être voisins ; midi doit être loin des deux.
    assert np.linalg.norm(minuit_moins - minuit_plus) < 0.05
    assert np.linalg.norm(minuit_moins - midi) > 1.5


def test_le_weekend_est_marque():
    frame = pd.DataFrame(
        {   # 2024-01-06 est un samedi, 2024-01-08 un lundi
            "tpep_pickup_datetime": pd.to_datetime(["2024-01-06", "2024-01-08"]),
            "trip_distance": [1.0, 1.0],
            "VendorID": [1, 1],
            "passenger_count": [1, 1],
        }
    )
    assert derive_features(frame)["is_weekend"].tolist() == [1.0, 0.0]


def test_passenger_count_absent_ou_manquant():
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2024-01-01 08:00:00"]),
            "trip_distance": [2.0],
            "VendorID": [1],
        }
    )
    assert derive_features(frame)["passenger_count"].tolist() == [1.0]


def test_log_distance_monotone():
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2024-01-01"] * 3),
            "trip_distance": [0.5, 5.0, 50.0],
            "VendorID": [1, 1, 1],
            "passenger_count": [1, 1, 1],
        }
    )
    values = derive_features(frame)["log_distance"].to_numpy()
    assert (np.diff(values) > 0).all()


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------


def test_le_modele_bat_largement_la_reference(fitted):
    _, metrics, baseline = fitted
    assert metrics.mae < baseline.mae * 0.5


def test_le_modele_approche_le_plancher_theorique(fitted):
    """Avec un processus génératif connu, on sait ce qui est atteignable."""
    _, metrics, _ = fitted
    floor = irreducible_mae(SMALL)
    # Sur 4 000 courses seulement, on tolère un écart large ; l'important
    # est que le modèle soit du bon ordre de grandeur et non deux fois pire.
    assert metrics.mae < floor * 1.8


def test_metriques_coherentes(fitted):
    _, metrics, _ = fitted
    assert metrics.median_ae <= metrics.mae <= metrics.p90_ae
    assert metrics.rmse >= metrics.mae
    assert metrics.n > 0


def test_evaluate_sur_prediction_parfaite():
    y = np.array([100.0, 200.0, 300.0])
    m = evaluate(y, y)
    assert m.mae == 0.0 and m.rmse == 0.0 and m.p90_ae == 0.0


@pytest.mark.parametrize("estimator", ["hgb", "rf"])
def test_les_deux_estimateurs_fonctionnent(trips, estimator):
    _, metrics, baseline = train(trips.head(1500), estimator=estimator)
    assert metrics.mae < baseline.mae


def test_estimateur_inconnu_rejete():
    with pytest.raises(ValueError, match="estimateur inconnu"):
        build_pipeline(estimator="xgboost")


def test_cible_absente_rejetee():
    with pytest.raises(ValueError, match="trip_duration"):
        train(generate_synthetic(SyntheticSpec(n_rows=10)))


def test_vendorid_inconnu_ne_fait_pas_planter(fitted, trips):
    """handle_unknown='ignore' doit absorber une catégorie jamais vue."""
    pipeline, _, _ = fitted
    unseen = trips.head(5).copy()
    unseen["VendorID"] = 999
    predictions = pipeline.predict(unseen)
    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_aller_retour_du_modele_sur_disque(fitted, trips, tmp_path):
    pipeline, _, _ = fitted
    path = save_model(pipeline, tmp_path / "m.joblib")
    reloaded = load_model(path)
    sample = trips.head(50).drop(columns=["trip_duration"])
    np.testing.assert_allclose(
        pipeline.predict(sample), reloaded.predict(sample), rtol=1e-9
    )


# --------------------------------------------------------------------------
# Inférence
# --------------------------------------------------------------------------


def test_predict_partition_ajoute_la_colonne(fitted, trips):
    pipeline, _, _ = fitted
    out = predict_partition(trips.head(100), pipeline)
    assert PREDICTION_COLUMN in out.columns
    assert out[PREDICTION_COLUMN].notna().all()


def test_partition_vide(fitted, trips):
    pipeline, _, _ = fitted
    out = predict_partition(trips.head(0), pipeline)
    assert PREDICTION_COLUMN in out.columns
    assert len(out) == 0


def test_la_notification_est_emise_pendant_le_traitement(fitted, trips):
    pipeline, _, _ = fitted
    recus: list[tuple[int, float]] = []
    predict_partition(
        trips.head(250), pipeline, notify=lambda i, d: recus.append((i, d)), notify_every=100
    )
    assert len(recus) == 3
    assert all(d > 0 for _, d in recus)


def test_sequentiel_et_dask_donnent_les_memes_predictions(fitted, trips):
    """Propriété centrale : distribuer ne change pas le résultat."""
    pipeline, _, _ = fitted
    sample = trips.head(2000)

    seq, seq_report = predict_sequential(sample, pipeline, chunk_size=500)
    par, par_report = predict_dask(sample, pipeline, npartitions=4, scheduler="threads")

    assert seq_report.rows == par_report.rows == len(sample)
    np.testing.assert_allclose(
        seq[PREDICTION_COLUMN].to_numpy(),
        par[PREDICTION_COLUMN].to_numpy(),
        rtol=1e-9,
    )


def test_le_nombre_de_partitions_ne_change_pas_le_resultat(fitted, trips):
    pipeline, _, _ = fitted
    sample = trips.head(1200)
    a, _ = predict_dask(sample, pipeline, npartitions=2, scheduler="threads")
    b, _ = predict_dask(sample, pipeline, npartitions=6, scheduler="threads")
    np.testing.assert_allclose(
        a[PREDICTION_COLUMN].to_numpy(), b[PREDICTION_COLUMN].to_numpy(), rtol=1e-9
    )


def test_rapport_dinference_coherent(fitted, trips):
    pipeline, _, _ = fitted
    _, report = predict_sequential(trips.head(500), pipeline, chunk_size=200)
    assert report.rows == 500
    assert report.partitions == 3
    assert report.seconds > 0
    assert report.rows_per_second > 0


def test_ecriture_parquet_des_predictions(fitted, trips, tmp_path):
    pipeline, _, _ = fitted
    out = tmp_path / "preds"
    result, report = predict_dask(
        trips.head(800), pipeline, npartitions=4, scheduler="threads", output_path=out
    )
    assert result is None
    assert report.rows == 800
    relu = pd.read_parquet(out)
    assert len(relu) == 800
    assert PREDICTION_COLUMN in relu.columns


def test_mode_parquet_hors_memoire(fitted, trips, tmp_path):
    """Le chemin Parquet ne charge jamais la table entière en mémoire."""
    from taxieta import predict_parquet_dask

    pipeline, _, _ = fitted
    source = tmp_path / "in.parquet"
    trips.head(1500).drop(columns=["trip_duration"]).to_parquet(source, index=False)

    out = tmp_path / "out"
    report = predict_parquet_dask(source, pipeline, out, scheduler="threads")

    assert report.rows == 1500
    relu = pd.read_parquet(out)
    assert PREDICTION_COLUMN in relu.columns
    assert relu[PREDICTION_COLUMN].notna().all()
