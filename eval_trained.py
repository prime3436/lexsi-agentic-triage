import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import auditkit as ak
from auditkit.agent_eval import episodes_from_agenttune
from generate_dataset import SYSTEM, gold_final_json

DIR = Path(__file__).parent / "data"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
FIELDS = ["category", "severity", "hazard_detected", "sla_hours", "specialty"]


def find(root, name):
    hits = sorted(Path(root).rglob(name))
    return hits[0].parent if hits else None


def load(path):
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4 * 1024**3 else "cpu"
    dt = torch.float16 if dev.startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(BASE, dtype=dt).to(dev)
    if path:
        ad = find(path, "adapter_config.json")
        full = find(path, "config.json")
        if ad is not None:
            from peft import PeftModel
            mdl = PeftModel.from_pretrained(mdl, str(ad))
        elif full is not None and "model_type" in json.loads((full / "config.json").read_text()):
            mdl = AutoModelForCausalLM.from_pretrained(str(full), dtype=dt).to(dev)
    mdl.eval()
    return tok, mdl


def gen_batch(tok, mdl, texts, use_sys=False, bs=16):
    outs = []
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        prompts = []
        for t in chunk:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}] if use_sys else [{"role": "user", "content": t}]
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        enc = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=140, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen:
            outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
    return outs


def parse(s):
    try:
        a, b = s.index("{"), s.rindex("}") + 1
        obj = json.loads(s[a:b])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def norm_cat(v):
    if not isinstance(v, str):
        return v
    vl = v.lower()
    if "hvac" in vl or "climate" in vl:
        return "HVAC / Climate Control"
    if "plumb" in vl or "water" in vl:
        return "Plumbing"
    if "electr" in vl:
        return "Electrical"
    if "safe" in vl or "secur" in vl or "gas" in vl:
        return "Safety & Security"
    if "general" in vl or "maint" in vl:
        return "General Maintenance"
    return v


def norm_spec(v):
    if not isinstance(v, str):
        return v
    vl = v.lower()
    if "remediation" in vl or "restoration" in vl:
        return "24/7 Water & Fire Remediation"
    if "plumb" in vl:
        return "Licensed Master Plumber"
    if "electr" in vl:
        return "Certified Master Electrician"
    if "hvac" in vl:
        return "EPA-Certified HVAC Technician"
    if "handyman" in vl:
        return "Licensed General Handyman"
    return v


def norm_sev(v):
    if not isinstance(v, str):
        return v
    for k in ["P1", "P2", "P3", "P4"]:
        if k in v.upper():
            return {"P1": "P1_CRITICAL", "P2": "P2_HIGH", "P3": "P3_MEDIUM", "P4": "P4_LOW"}[k]
    return v


def norm_pred(p):
    if not p:
        return None
    q = dict(p)
    if "category" in q:
        q["category"] = norm_cat(q["category"])
    if "specialty" in q:
        q["specialty"] = norm_spec(q["specialty"])
    if "severity" in q:
        q["severity"] = norm_sev(q["severity"])
    return q


def score(preds, rows):
    n = len(rows)
    valid = sum(p is not None for p in preds)
    npreds = [norm_pred(p) for p in preds]
    exact = {f: round(100 * sum(bool(p is not None and p.get(f) == r["gold"][f]) for p, r in zip(preds, rows)) / n, 1) for f in FIELDS}
    norm = {f: round(100 * sum(bool(p is not None and p.get(f) == r["gold"][f]) for p, r in zip(npreds, rows)) / n, 1) for f in FIELDS}
    tp = sum(bool(p is not None and p.get("hazard_detected") and r["gold"]["hazard_detected"]) for p, r in zip(preds, rows))
    pos = sum(bool(r["gold"]["hazard_detected"]) for r in rows)
    return {
        "n": n,
        "valid_json_pct": round(100 * valid / n, 1),
        "exact_field_accuracy_pct": exact,
        "normalized_field_accuracy_pct": norm,
        "hazard_recall_pct": round(100 * tp / max(1, pos), 1),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rows = [json.loads(l) for l in (DIR / "test_tickets.jsonl").read_text(encoding="utf-8").splitlines()]
    tok, mdl = load(path)
    use_sys = not (path and "distill" in str(path).lower())
    raw = gen_batch(tok, mdl, [r["description"] for r in rows], use_sys=use_sys)
    preds = [parse(x) for x in raw]
    samples = [
        ak.Sample(id=r["id"], input=r["description"], target=gold_final_json(r["gold"]), actual_output=x)
        for r, x in zip(rows, raw)
    ]
    res = ak.evaluate(samples, "precomputed", ["exact_match", "f1_score"])
    name = (Path(path).name if path else "base") + "_report.json"
    out = {
        "model": path or BASE,
        "scores": score(preds, rows),
        "auditkit": {"headline": res.headline, "fingerprint": res.fingerprint},
        "sample_outputs": raw[:3],
    }
    if path and (Path(path) / "trajectories.jsonl").exists():
        eps = episodes_from_agenttune(str(path))
        out["agenttune_rollouts"] = {
            "episodes": len(eps),
            "total_tool_calls": sum(ep.counters.get("n_tool_calls", 0) for ep in eps),
            "has_provenance": bool(eps and eps[0].metadata.get("provenance")),
        }
    (DIR / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
