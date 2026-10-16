# Router source for the chintasamsonhruday submissions

This branch publishes the Python sources used for the intro and hard route files
in the submission PR. It is separate from the results PR because
`CONTRIBUTING.md` restricts that PR to `submissions/**` and the generated
leaderboard files. The challenge's `m3d` package and benchmark instances are
already in this repository. Python 3.9+ and the standard library are sufficient.

## Intro: staged seed plus corridor LNS

From the repository root:

```sh
python experiments/run_hard_comparison.py --suite benchmarks --script test_multi_target_astar.py --out-dir intro_reproduction
python -m m3d.cli score-suite --suite benchmarks --submission-dir intro_reproduction/lns
```

The `lns` directory contains the completed case routes and full end-to-end
runtime measurements. The comparison runner snapshots and hashes its sources.
The submitted intro run used a 60-second per-case LNS budget, 2-second group
budget, maximum 200 nodes, maximum group size 6, and 100 groups. These are
the runner defaults. The prototype script is executed through
`experiments/run_updated.py` and then refined by
`experiments/run_lns_experiment.py`.

## Hard: negotiated seed plus multidimensional refinement

From the repository root:

```sh
python experiments/run_guarded_hard_suite.py --out hard_reproduction --workers 6 --refine-stages bounds,neutral --refine-seconds 120 --seed 3
python -m m3d.cli score-suite --suite benchmarks_hard --submission-dir hard_reproduction
```

The hard runner saves source snapshots and SHA-256 hashes in
`hard_reproduction/source_snapshot` and `hard_reproduction/launch.json`.
It checks each completed route before keeping it. `run_hard_suite_parallel.py`
generates the seed, and `experiments/run_multidimensional.py` performs the
bounded refinement. The submitted hard suite was produced by this command.

Both runs were measured on an AMD Ryzen 7 8845HS laptop with 16 GB RAM.
Each case used one routing thread. The hard suite scheduled up to six
independent cases concurrently.
