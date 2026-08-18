# Historical experiments

The current portfolio path lives in `src/train_v13.py`, `src/evaluate_v13.py`,
and `src/v13_pipeline.py`. This directory preserves earlier controlled
experiments so their negative results and data decisions remain reproducible
without obscuring the current entry points.

Run legacy modules from the repository root:

```bash
python -m experiments.legacy.train_smoke
python -m experiments.legacy.train_baseline --help
python -m experiments.legacy.build_dent_dataset --help
python -m experiments.legacy.build_rim_dent_dataset --help
python -m experiments.legacy.evaluate_dent --help
```

See [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) for the configuration,
result, and interpretation of each run.
