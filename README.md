# lexsi-agentic-triage

Track 3: Agentic Fine-Tuning (`AgentTune` + `AuditKIT`)

Colab: https://colab.research.google.com/github/prime3436/lexsi-agentic-triage/blob/main/notebooks/track3_agenttune_auditkit.ipynb

## Run

```bash
python generate_dataset.py
python eval_baseline.py
python eval_trained.py
python eval_trained.py ./output/distilled
python eval_trained.py ./output/grpo
```

## Results (74 held-out test tickets, AuditKIT 1.1.3)

| Metric | Base Qwen2.5-1.5B | Rule + TF-IDF | AgentTune Distilled (Qwen2.5-1.5B LoRA) |
| :--- | :---: | :---: | :---: |
| Valid JSON Rate | 29.7% | 100.0% | **100.0%** |
| Category Accuracy | 0.0% | 89.2% | **98.6%** (100.0% norm) |
| Specialty Accuracy | 0.0% | 87.8% | **100.0%** |
| Hazard Detection Accuracy | 0.0% | 91.9% | **97.3%** |
| Hazard Recall (P1) | 0.0% | 73.3% | **100.0%** |
| SLA Accuracy | 0.0% | 74.3% | **85.1%** |
| Severity Accuracy | 0.0% | 74.3% | **74.3%** |
| AuditKIT Exact Match | 0.0% | 29.7% | **68.9%** |
| AuditKIT Token F1 | 2.4% | 84.0% | **95.1%** |
| AuditKIT Fingerprint | `683077555bd22ec3` | `da71ee1fd9bd6301` | `d03ce03fd3ceac0b` |

`create_agentic_trainer(algorithm="grpo")` additionally logged 160 tool-calling rollout episodes (277 tool calls) with `lexsi_provenance.json` verified via `auditkit.agent_eval.episodes_from_agenttune`.
