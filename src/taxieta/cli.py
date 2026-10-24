"""Ligne de commande : make-data, train, predict, bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import (
    SyntheticSpec,
    add_duration,
    clean_trips,
    download_tlc,
    generate_synthetic,
    irreducible_mae,
    load_parquet,
)
from .inference import (
    predict_dask,
    predict_parquet_dask,
    predict_sequential,
    print_eta,
)
from .model import load_model, save_model, train


def _read_trips(args: argparse.Namespace):
    """Charge les courses depuis un fichier, ou en génère."""
    if args.data:
        frame = load_parquet(Path(args.data))
    elif args.tlc:
        year, month = (int(p) for p in args.tlc.split("-"))
        path = Path(args.cache) / f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
        print(f"téléchargement des courses TLC {year}-{month:02d}…", file=sys.stderr)
        frame = load_parquet(download_tlc(year, month, path))
    else:
        frame = generate_synthetic(SyntheticSpec(n_rows=args.rows, seed=args.seed))
    return clean_trips(add_duration(frame))


def cmd_make_data(args: argparse.Namespace) -> int:
    frame = generate_synthetic(SyntheticSpec(n_rows=args.rows, seed=args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # La taille des row groups n'est pas un détail de stockage : c'est elle
    # qui borne la finesse du découpage de Dask à la lecture. Un fichier
    # écrit en quelques row groups géants ne peut pas être traité par
    # petits blocs, quel que soit le `blocksize` demandé.
    frame.to_parquet(out, index=False, row_group_size=args.row_group_size)
    print(f"{len(frame):,} courses synthétiques écrites dans {out}")
    print(f"plancher d'erreur théorique : {irreducible_mae(SyntheticSpec(n_rows=min(args.rows, 100_000), seed=args.seed)):.1f}s")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    frame = _read_trips(args)
    print(f"{len(frame):,} courses après nettoyage\n")

    pipeline, metrics, baseline = train(
        frame, estimator=args.estimator, test_size=args.test_size
    )

    print(f"modèle     : {metrics.summary()}")
    print(f"référence  : {baseline.summary()}   (prédit toujours la médiane)")
    gain = (1 - metrics.mae / baseline.mae) * 100
    print(f"\nréduction de la MAE face à la référence : {gain:.1f} %")

    if not args.data and not args.tlc:
        floor = irreducible_mae(SyntheticSpec(n_rows=min(args.rows, 100_000), seed=args.seed))
        print(f"plancher théorique du bruit synthétique : {floor:.1f}s")
        print(f"écart au plancher : {metrics.mae - floor:+.1f}s")

    save_model(pipeline, Path(args.out))
    print(f"\nmodèle écrit dans {args.out}")

    if args.metrics_out:
        Path(args.metrics_out).write_text(
            json.dumps(
                {"model": metrics.as_dict(), "baseline": baseline.as_dict()}, indent=2
            ),
            encoding="utf-8",
        )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    notify = print_eta if args.notify else None

    # Mode hors mémoire : on ne charge jamais le fichier entier.
    if args.mode == "parquet":
        if not args.data or not args.out:
            print("le mode parquet exige --data et --out", file=sys.stderr)
            return 1
        report = predict_parquet_dask(
            Path(args.data), model, Path(args.out),
            scheduler=args.scheduler, blocksize=args.blocksize, notify=notify,
        )
        print(report.summary())
        print(f"prédictions écrites dans {args.out}")
        return 0

    frame = _read_trips(args)

    if args.mode == "sequential":
        result, report = predict_sequential(frame, model, notify=notify)
    else:
        result, report = predict_dask(
            frame,
            model,
            npartitions=args.partitions,
            scheduler=args.scheduler,
            notify=notify,
            output_path=Path(args.out) if args.out else None,
        )

    print(report.summary())
    if args.out and result is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.out, index=False)
        print(f"prédictions écrites dans {args.out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    frame = _read_trips(args)
    model = load_model(Path(args.model))

    print(f"{len(frame):,} courses, {args.partitions} partitions\n")

    _, seq = predict_sequential(frame, model)
    print(seq.summary())

    reports = [seq]
    for scheduler in ("threads", "processes"):
        try:
            _, rep = predict_dask(
                frame, model, npartitions=args.partitions, scheduler=scheduler
            )
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            print(f"[dask {scheduler}] indisponible : {type(exc).__name__}: {exc}")
            continue
        print(rep.summary())
        reports.append(rep)

    print("\naccélération face au séquentiel :")
    for rep in reports[1:]:
        print(f"  {rep.mode:18s} ×{seq.seconds / rep.seconds:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taxieta",
        description="Prédiction de la durée des courses de taxi, inférence distribuée.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", help="fichier Parquet de courses")
        p.add_argument("--tlc", help="mois TLC à télécharger, au format AAAA-MM")
        p.add_argument("--cache", default="data", help="dossier de cache des téléchargements")
        p.add_argument("--rows", type=int, default=200_000, help="courses synthétiques si aucune source")
        p.add_argument("--seed", type=int, default=42)

    p_data = sub.add_parser("make-data", help="générer un jeu de données synthétique")
    p_data.add_argument("--rows", type=int, default=200_000)
    p_data.add_argument("--seed", type=int, default=42)
    p_data.add_argument("--out", default="data/synthetic.parquet")
    p_data.add_argument(
        "--row-group-size", type=int, default=50_000,
        help="lignes par row group ; borne la finesse du découpage de Dask",
    )
    p_data.set_defaults(func=cmd_make_data)

    p_train = sub.add_parser("train", help="entraîner le modèle")
    add_source(p_train)
    p_train.add_argument("--estimator", choices=["hgb", "rf"], default="hgb")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--out", default="artifacts/model.joblib")
    p_train.add_argument("--metrics-out")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="prédire sur un jeu de courses")
    add_source(p_pred)
    p_pred.add_argument("--model", default="artifacts/model.joblib")
    p_pred.add_argument(
        "--mode", choices=["sequential", "dask", "parquet"], default="dask",
        help="parquet : lecture et écriture par blocs, sans tout charger en mémoire",
    )
    p_pred.add_argument("--blocksize", default="64MB")
    p_pred.add_argument("--scheduler", choices=["threads", "processes"], default="threads")
    p_pred.add_argument("--partitions", type=int, default=8)
    p_pred.add_argument("--out")
    p_pred.add_argument("--notify", action="store_true", help="émettre les estimations au fil de l'eau")
    p_pred.set_defaults(func=cmd_predict)

    p_bench = sub.add_parser("bench", help="comparer séquentiel, threads et processus")
    add_source(p_bench)
    p_bench.add_argument("--model", default="artifacts/model.joblib")
    p_bench.add_argument("--partitions", type=int, default=8)
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
