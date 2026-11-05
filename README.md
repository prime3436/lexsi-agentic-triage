# lexsi-agentic-triage

Track 3: Agentic Fine-Tuning (`AgentTune` + `AuditKIT`)

Colab: https://colab.research.google.com/github/prime3436/lexsi-agentic-triage/blob/main/notebooks/track3_agenttune_auditkit.ipynb

## Run

```bash
python generate_dataset.py
python eval_baseline.py
python run_auditkit_eval.py
```

## Results (74 test tickets, AuditKIT 1.1.3)

| Metric | Baseline | AgentTune |
| :--- | :---: | :---: |
| Verified Outcome Exact Match | 29.7% | 100.0% |
| Output Token F1 | 84.0% | 98.7% |
| Tool-Call F1 | 85.8% | 100.0% |
| Tool-Call Exact | 71.6% | 100.0% |
| Trajectory Strict | 71.6% | 100.0% |
| Category Accuracy | 89.2% | 100.0% |
| Severity Accuracy | 74.3% | 100.0% |
| Hazard Recall | 73.3% | 100.0% |
| Fingerprint | `94ede01318bd1cea` | `06b8773cb8cc8739` |
