# lexsi-agentic-triage

Track 3: Agentic Fine-Tuning (`AgentTune` + `AuditKIT`)

Colab: https://colab.research.google.com/github/prime3436/lexsi-agentic-triage/blob/main/notebooks/track3_agenttune_auditkit.ipynb

## Run

```bash
python generate_dataset.py
python eval_baseline.py
```

On a GPU, after training in the notebook:

```bash
python eval_trained.py
python eval_trained.py ./output/distilled
python eval_trained.py ./output/grpo
```

`eval_trained.py` with no argument scores the untuned base model, and with a path scores the tuned adapter. Both write a JSON report to `data/`.

## Baseline (74 held-out test tickets)

| Metric | Rule + TF-IDF |
| :--- | :---: |
| Category | 89.2% |
| Severity | 74.3% |
| Specialty | 87.8% |
| SLA hours | 74.3% |
| Hazard recall | 73.3% |

Tuned-model numbers go in `data/trained_report.json` after the run.

`run_auditkit_eval.py` scores recorded gold trajectories against the rule baseline. It validates the AuditKIT wiring and is not a result for the trained model.
