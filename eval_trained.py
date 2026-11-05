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


def find(root, name):
    hits = sorted(Path(root).rglob(name))
    return hits[0].parent if hits else None


def load(path):
    tok = AutoTokenizer.from_pretrained(BASE)
    mdl = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0})
    if path:
        ad = find(path, "adapter_config.json")
        full = find(path, "config.json")
        if ad is not None:
            from peft import PeftModel
            mdl = PeftModel.from_pretrained(mdl, str(ad))
        elif full is not None and "model_type" in json.loads((full / "config.json").read_text()):
            mdl = AutoModelForCausalLM.from_pretrained(str(full), dtype=torch.float16, device_map={"": 0})
        else:
            files = sorted(str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file())
            raise SystemExit(f"no adapter or model in {path}; found: {files[:40]}")
    mdl.eval()
    return tok, mdl


def gen(tok, mdl, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**enc, max_new_tokens=160, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


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
    name = (Path(path).name if path else "base") + "_report.json"
    out = {
        "model": path or BASE,
        "scores": score(preds, rows),
        "auditkit": {"headline": res.headline, "fingerprint": res.fingerprint},
        "sample_outputs": raw[:3],
    }
    (DIR / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
