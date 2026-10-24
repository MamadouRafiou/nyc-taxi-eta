"""Mesure du pic mémoire : table chargée en entier contre lecture par blocs.

Chaque mode doit tourner dans un processus neuf, sinon le pic mesuré est
celui du mode le plus gourmand exécuté avant. Ce script relance donc
lui-même des sous-processus.

Usage :
    python bench/memory.py --rows 3000000 --model artifacts/model.joblib
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_child(mode: str, data: Path, model: Path, row_group_size: int) -> dict:
    from taxieta import load_model, load_parquet, predict_dask, predict_parquet_dask

    pipeline = load_model(model)
    out = Path(tempfile.mkdtemp(prefix=f"pred_{mode}_"))
    start = time.perf_counter()

    if mode == "memory":
        frame = load_parquet(data)
        _, report = predict_dask(
            frame, pipeline, npartitions=8, scheduler="threads", output_path=out
        )
    else:
        report = predict_parquet_dask(
            data, pipeline, out, scheduler="threads", blocksize="8MB"
        )

    return {
        "mode": mode,
        "rows": report.rows,
        "partitions": report.partitions,
        "seconds": round(time.perf_counter() - start, 2),
        "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "row_group_size": row_group_size,
    }


def baseline_mb(model: Path) -> float:
    from taxieta import load_model

    load_model(model)
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3_000_000)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--row-group-sizes", type=int, nargs="+", default=[1_048_576, 50_000])
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    parser.add_argument("--_data", help=argparse.SUPPRESS)
    parser.add_argument("--_rgs", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._child:
        print(json.dumps(run_child(args._child, Path(args._data), Path(args.model), args._rgs)))
        return 0

    if args._data == "baseline":
        print(json.dumps({"baseline_mb": baseline_mb(Path(args.model))}))
        return 0

    if args._data and args._data.startswith("generate:"):
        # Génération dans un processus dédié : le parent ne doit jamais
        # détenir la table, sinon les enfants qu'il engendre héritent de son
        # empreinte et le pic mesuré n'est plus celui du mode testé.
        from taxieta import SyntheticSpec, generate_synthetic

        target = args._data.split(":", 1)[1]
        generate_synthetic(SyntheticSpec(n_rows=args.rows)).to_parquet(
            target, index=False, row_group_size=args._rgs
        )
        print(json.dumps({"written": target}))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="taxibench_"))

    base = json.loads(
        subprocess.run(
            [sys.executable, __file__, "--model", args.model, "--_data", "baseline"],
            capture_output=True, text=True, check=True,
        ).stdout
    )["baseline_mb"]
    print(f"empreinte de base (imports + modèle) : {base} Mo\n")
    print("| mode | lignes/row group | partitions | temps | pic mémoire | données seules |")
    print("|---|---|---|---|---|---|")

    for rgs in args.row_group_sizes:
        path = tmp / f"trips_{rgs}.parquet"
        subprocess.run(
            [sys.executable, __file__, "--model", args.model, "--rows", str(args.rows),
             "--_data", f"generate:{path}", "--_rgs", str(rgs)],
            capture_output=True, text=True, check=True,
        )
        for mode in ("memory", "stream"):
            out = subprocess.run(
                [sys.executable, __file__, "--model", args.model,
                 "--_child", mode, "--_data", str(path), "--_rgs", str(rgs)],
                capture_output=True, text=True, check=True,
            )
            r = json.loads(out.stdout.strip().splitlines()[-1])
            label = "pandas complet" if mode == "memory" else "Parquet par blocs"
            print(
                f"| {label} | {rgs:,} | {r['partitions']} | {r['seconds']}s | "
                f"{r['peak_mb']} Mo | {round(r['peak_mb'] - base, 1)} Mo |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
