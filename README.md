# taxieta — prédiction de la durée des courses de taxi

> Modèle de durée de trajet, inférence distribuée avec Dask, et une conclusion mesurée qui va à l'encontre de l'intuition de départ.
> Fonctionne hors ligne, sans télécharger un seul octet de données.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![Dask](https://img.shields.io/badge/Dask-distributed-FDA061?logo=dask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn)
[![CI](https://github.com/MamadouRafiou/nyc-taxi-eta/actions/workflows/ci.yml/badge.svg)](https://github.com/MamadouRafiou/nyc-taxi-eta/actions/workflows/ci.yml)

---

## En une minute

```bash
git clone https://github.com/MamadouRafiou/nyc-taxi-eta.git
cd nyc-taxi-eta
pip install -e ".[dev]"

taxieta make-data --rows 400000 --out data/trips.parquet
taxieta train     --data data/trips.parquet --out artifacts/model.joblib
taxieta predict   --data data/trips.parquet --model artifacts/model.joblib \
                  --mode parquet --out predictions/
```

Aucune donnée à télécharger : un générateur synthétique produit des courses au schéma de la NYC TLC. Pour travailler sur les vraies données, `--tlc 2024-01` les récupère.

---

## Le problème

Un service de taxi veut annoncer au chauffeur la durée estimée **dès que la course est enregistrée**, pas après le traitement d'un lot complet. Deux exigences en découlent : un modèle correct, et une inférence capable d'absorber un mois de courses en émettant ses résultats au fil de l'eau.

---

## Le modèle

### Attributs

| Attribut | Pourquoi |
|---|---|
| `trip_distance` | Premier déterminant de la durée |
| `log_distance` | La distribution est très asymétrique ; le logarithme la rééquilibre |
| `hour_sin`, `hour_cos` | L'heure projetée sur un cercle |
| `day_of_week`, `is_weekend` | La circulation du dimanche n'est pas celle du mardi |
| `passenger_count` | Marginal, mais disponible |
| `VendorID` | Encodé en one-hot, avec `handle_unknown="ignore"` |

**L'encodage cyclique de l'heure mérite un mot.** Utilisée brute, l'heure 23 et l'heure 0 sont distantes de 23 unités alors qu'une minute les sépare. Projetées en sinus et cosinus, elles redeviennent voisines — ce qui correspond à la circulation réelle. Un test le vérifie explicitement.

### Un seul pipeline, de la ligne brute à la prédiction

La dérivation des attributs, la mise à l'échelle, l'encodage et le modèle sont assemblés dans un unique objet scikit-learn. `predict` accepte donc directement des lignes au schéma TLC.

Ce n'est pas de la cosmétique : cela supprime la classe de bogues où l'entraînement et l'inférence ne calculent pas tout à fait les mêmes colonnes. Un seul fichier sérialisé contient la recette complète.

### Résultats

Sur 400 000 courses synthétiques, 80 000 en test :

| | MAE | RMSE | médiane | p90 |
|---|---|---|---|---|
| **Modèle** (HistGradientBoosting) | **187,3 s** | 340,0 s | 98,3 s | 433,8 s |
| Référence (prédit toujours la médiane) | 521,3 s | 946,3 s | 290,0 s | 1 229,0 s |

**Réduction de la MAE face à la référence : 64,1 %.**

Le chiffre qui compte davantage est ailleurs. Le générateur synthétique applique un bruit multiplicatif lognormal que rien ne peut prédire : il impose un **plancher d'erreur de 185,0 s**. Le modèle atteint 187,3 s, soit **2,4 secondes au-dessus de l'optimum théorique**. Il a donc appris pratiquement tout ce qui était apprenable.

C'est l'intérêt d'un générateur dont on connaît le processus : on sait ce que « bon » veut dire. Sur des données réelles, une MAE isolée ne dit rien sans point de comparaison.

> Ces chiffres portent sur des données synthétiques. Pour les reproduire sur les vraies courses new-yorkaises : `taxieta train --tlc 2024-01`.

---

## L'inférence distribuée : ce que la mesure a démenti

L'hypothèse de départ était simple : distribuer l'inférence sur plusieurs unités d'exécution devrait l'accélérer. **C'est faux pour ce pipeline**, et la raison est instructive.

400 000 courses, 8 partitions, 2 cœurs logiques :

| configuration | temps | débit | face au séquentiel |
|---|---|---|---|
| pandas séquentiel | **1,24 s** | 323 572 courses/s | référence |
| Dask, fils d'exécution | 1,62 s | 246 378 courses/s | ×0,76 |
| Dask, processus | 5,72 s | 69 879 courses/s | ×0,22 |

Le mode « séquentiel » n'en est pas un. `HistGradientBoostingRegressor.predict` est parallélisé en interne via OpenMP : il occupe déjà les deux cœurs. Ajouter Dask par-dessus ne crée pas de parallélisme, cela crée de la **sur-souscription** — plus de fils que de cœurs, et un ordonnanceur de plus à nourrir.

Pour le vérifier, il suffit de confiner le modèle à un seul cœur :

| configuration (`OMP_NUM_THREADS=1`) | temps | débit | face au séquentiel |
|---|---|---|---|
| pandas séquentiel | 2,04 s | 195 731 courses/s | référence |
| Dask, fils d'exécution | **1,59 s** | 252 145 courses/s | **×1,29** |
| Dask, processus | 4,33 s | 92 380 courses/s | ×0,47 |

Dask redevient utile — il récupère le parallélisme qu'on vient de retirer au modèle. Mais il ne rattrape jamais la configuration la plus simple : 1,59 s contre 1,24 s.

Quant aux processus, ils perdent dans tous les cas. Chaque worker doit recevoir une copie sérialisée du modèle et de sa partition ; à cette taille, ce transport coûte davantage que le calcul.

**Conclusion : sur une machine unique, pour un modèle déjà vectorisé et déjà parallèle, Dask ne sert pas à aller plus vite.**

---

## Alors à quoi sert Dask ici ?

À traiter un fichier plus gros que la mémoire disponible. C'est le mode `--mode parquet` : Dask lit par blocs, prédit, et écrit chaque partition au fur et à mesure, sans jamais charger la table entière.

3 millions de courses, fichier Parquet de 52 Mo, empreinte de base du processus (imports + modèle) : **181,4 Mo**.

| mode | lignes par row group | partitions | temps | pic mémoire | dont données |
|---|---|---|---|---|---|
| pandas complet | 1 048 576 | 8 | 9,55 s | 953,8 Mo | 772,4 Mo |
| Parquet par blocs | 1 048 576 | 3 | 8,40 s | 1 000,7 Mo | 819,3 Mo |
| pandas complet | 50 000 | 8 | 9,01 s | 1 011,6 Mo | 830,2 Mo |
| **Parquet par blocs** | **50 000** | 10 | 9,00 s | **571,2 Mo** | **389,8 Mo** |

Reproductible avec `python bench/memory.py --rows 3000000`.

### Le piège que cette mesure a révélé

Lire par blocs ne réduit la mémoire que si le fichier le permet. Sur la deuxième ligne du tableau, le mode « par blocs » ne gagne **rien** — il consomme même un peu plus.

La raison est que **la finesse du découpage de Dask est bornée par la taille des row groups du Parquet**. Un fichier écrit en trois row groups d'un million de lignes ne peut pas être traité en petits morceaux : Dask doit charger un row group entier à la fois, quel que soit le `blocksize` demandé. Trois partitions d'un million de lignes, ce n'est pas du traitement par blocs.

Avec des row groups de 50 000 lignes, le même code descend à 389,8 Mo de données en mémoire — **moitié moins**.

La conséquence pratique est dans le code : `taxieta make-data` écrit avec `row_group_size=50_000` par défaut. La façon dont un fichier est écrit décide de la façon dont il pourra être lu.

---

## Architecture

```
src/taxieta/
├── data.py         Générateur synthétique, téléchargement TLC, nettoyage
├── features.py     Dérivation des attributs, encodage cyclique de l'heure
├── model.py        Pipeline scikit-learn, entraînement, métriques
├── inference.py    Séquentiel, Dask en mémoire, Dask hors mémoire
└── cli.py          make-data, train, predict, bench
```

### Nettoyage des courses

Les bornes retenues — durée entre 30 secondes et 3 heures, distance entre 0,05 et 100 milles — ne sont pas arbitraires. Sous 30 secondes il s'agit presque toujours d'une annulation enregistrée par erreur ; au-delà de trois heures, de compteurs oubliés. Garder ces lignes revient à entraîner le modèle sur du bruit de saisie et à mesurer une erreur dominée par quelques valeurs absurdes.

### Notification au fil de l'eau

`predict_partition` émet l'estimation **à l'intérieur** de la partition, pas après un `compute()` global. Un chauffeur reçoit donc sa durée dès que sa partition est traitée, sans attendre que le mois entier soit calculé. C'est précisément l'exigence du cas d'usage.

---

## Utilisation

```bash
# Données
taxieta make-data --rows 400000 --out data/trips.parquet
taxieta make-data --rows 400000 --row-group-size 50000 --out data/trips.parquet

# Entraînement
taxieta train --data data/trips.parquet --estimator hgb --out artifacts/model.joblib
taxieta train --tlc 2024-01 --out artifacts/model.joblib     # vraies données TLC
taxieta train --data data/trips.parquet --estimator rf       # forêt aléatoire

# Inférence
taxieta predict --data data/trips.parquet --model artifacts/model.joblib --mode sequential
taxieta predict --data data/trips.parquet --model artifacts/model.joblib --mode dask --scheduler threads
taxieta predict --data data/trips.parquet --model artifacts/model.joblib --mode parquet --out predictions/
taxieta predict --data data/trips.parquet --model artifacts/model.joblib --notify

# Mesures
taxieta bench --data data/trips.parquet --model artifacts/model.joblib --partitions 8
python bench/memory.py --rows 3000000
```

---

## Tests

32 tests, aucun accès réseau, quelques secondes d'exécution.

```bash
pytest -v
```

Ils vérifient notamment que :

- le générateur produit bien le signal à apprendre (les heures de pointe sont mesurablement plus lentes) ;
- l'encodage cyclique rapproche 23 h 59 de 00 h 01 et les éloigne de midi ;
- le modèle bat la référence d'au moins un facteur deux et reste proche du plancher théorique ;
- un `VendorID` jamais vu ne fait pas tomber le service ;
- **séquentiel et distribué produisent des prédictions identiques**, et le nombre de partitions n'y change rien ;
- la notification est bien émise pendant le traitement, pas après ;
- le mode hors mémoire écrit un Parquet complet et relisible.

La CI exécute la suite sur Python 3.10, 3.11 et 3.12, puis déroule le pipeline complet en ligne de commande.

---

## Données réelles

Les courses des taxis jaunes de New York sont publiées par la [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). `--tlc AAAA-MM` télécharge le mois demandé et le met en cache.

---

## Contexte

Projet personnel écrit à partir de zéro, inspiré d'un travail de session en systèmes distribués (INF8810, UQAM). Le générateur synthétique, le pipeline scikit-learn, les modes d'inférence, la suite de tests, la ligne de commande et les mesures sont les miens.

**Mamadou Rafiou Barry** — [github.com/MamadouRafiou](https://github.com/MamadouRafiou)
