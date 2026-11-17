# Evaluation scripts

Run from the repository root:

```bash
python scripts/issue39_random_forest_evaluation.py --repo-root . --fn-cost 10 --fp-cost 1 --calibration sigmoid
```

See outputs/issue_39/README.md for results and limitations. The script refits recorded optimized hyperparameters without loading a pickle.
