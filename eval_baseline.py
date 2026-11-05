import json
import time
from pathlib import Path
from agent import TriageBaseline

DATA = Path(__file__).parent / "data" / "test_tickets.jsonl"


def main():
    ag = TriageBaseline()
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    cat_ok = sev_ok = spec_ok = sla_ok = 0
    tp = fn = fp = 0
    lat = []
    for r in rows:
        g = r["gold"]
        t0 = time.perf_counter()
        res = ag.run(r["description"])
        lat.append((time.perf_counter() - t0) * 1000)
        cat_ok += res["category"] == g["category"]
        sev_ok += res["severity"] == g["severity"]
        spec_ok += res["specialty"] == g["specialty"]
        sla_ok += res["sla_hours"] == g["sla_hours"]
        p, gt = res["hazard_detected"], g["hazard_detected"]
        tp += p and gt
        fn += (not p) and gt
        fp += p and (not gt)
    out = {
        "n_test": n,
        "category_accuracy": round(100 * cat_ok / n, 1),
        "severity_accuracy": round(100 * sev_ok / n, 1),
        "specialty_accuracy": round(100 * spec_ok / n, 1),
        "sla_accuracy": round(100 * sla_ok / n, 1),
        "hazard_recall": round(100 * tp / max(1, tp + fn), 1),
        "hazard_false_positives": fp,
        "avg_latency_ms": round(sum(lat) / n, 2),
    }
    print(json.dumps(out, indent=2))
    (Path(__file__).parent / "data" / "baseline_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
