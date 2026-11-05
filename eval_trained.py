import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import auditkit as ak
from generate_dataset import SYSTEM, gold_final_json

DIR = Path(__file__).parent / "data"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
FIELDS = ["category", "severity", "hazard_detected", "sla_hours", "specialty"]


def load(path):
    tok = AutoTokenizer.from_pretrained(BASE)
    mdl = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
    if path and (Path(path) / "adapter_config.json").exists():
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, path)
    elif path:
        mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
        tok = AutoTokenizer.from_pretrained(path)
    mdl.eval()
    return tok, mdl


def gen(tok, mdl, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(ids, max_new_tokens=160, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def parse(s):
    try:
        a, b = s.index("{"), s.rindex("}") + 1
        return json.loads(s[a:b])
    except Exception:
        return None


def score(preds, rows):
    n = len(rows)
    valid = sum(p is not None for p in preds)
    acc = {}
    for f in FIELDS:
        acc[f] = round(100 * sum(p is not None and p.get(f) == r["gold"][f] for p, r in zip(preds, rows)) / n, 1)
    tp = sum(p is not None and p.get("hazard_detected") and r["gold"]["hazard_detected"] for p, r in zip(preds, rows))
    pos = sum(r["gold"]["hazard_detected"] for r in rows)
    return {"n": n, "valid_json_pct": round(100 * valid / n, 1), "field_accuracy_pct": acc, "hazard_recall_pct": round(100 * tp / max(1, pos), 1)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rows = [json.loads(l) for l in (DIR / "test_tickets.jsonl").read_text(encoding="utf-8").splitlines()]
    tok, mdl = load(path)
    raw = [gen(tok, mdl, r["description"]) for r in rows]
    preds = [parse(x) for x in raw]
    out_map = {r["id"]: x for r, x in zip(rows, raw)}
    idx = {r["description"]: r["id"] for r in rows}
    samples = [ak.Sample(id=r["id"], input=r["description"], target=gold_final_json(r["gold"])) for r in rows]
    res = ak.evaluate(samples, lambda p: [out_map[idx[x]] for x in p], ["exact_match", "f1_score"])
    out = {
        "model": path or BASE,
        "scores": score(preds, rows),
        "auditkit": {"headline": res.headline, "fingerprint": res.fingerprint},
    }
    name = "trained_report.json" if path else "base_report.json"
    (DIR / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
