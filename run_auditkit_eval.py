import json
from pathlib import Path

import auditkit as ak
from auditkit.agent_eval import (
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    AnswerAssertion,
    episode_from_openai_messages,
    episodes_from_agenttune,
)

from agent import TriageBaseline
from generate_dataset import SYSTEM, gold_final_json

DIR = Path(__file__).parent / "data"
TOOLS = ["assess_hazard", "classify_trade", "estimate_cost", "dispatch_contractor"]
SCORERS = [
    "tool_call_f1",
    "tool_call_f1_name",
    "trajectory_match",
    "tool_call_validity",
    "redundant_tool_calls",
    "agent_loop_detection",
]


def get_cases(rows):
    out = []
    for r in rows:
        d, g = r["description"], r["gold"]
        turns = [
            [{"name": "assess_hazard", "arguments": {"description": d}}],
            [{"name": "classify_trade", "arguments": {"description": d}}],
            [{"name": "estimate_cost", "arguments": {"category": g["category"], "severity": g["severity"]}}],
            [{"name": "dispatch_contractor", "arguments": {"category": g["category"], "severity": g["severity"]}}],
        ]
        tgt = gold_final_json(g)
        out.append(
            AgentCase(
                id=r["id"],
                task=d,
                category=g["category"],
                allowed_tools=TOOLS,
                outcome=AnswerAssertion(reference=tgt),
                reference_turns=turns,
                metadata={"target": tgt, "gold": g},
            )
        )
    return out


def get_base_eps(rows):
    ag = TriageBaseline()
    eps = []
    for r in rows:
        d = r["description"]
        res = ag.run(d)
        ans = json.dumps({
            "category": res["category"],
            "severity": res["severity"],
            "hazard_detected": res["hazard_detected"],
            "sla_hours": res["sla_hours"],
            "specialty": res["specialty"],
            "min_cost_inr": res["min_cost_inr"],
            "max_cost_inr": res["max_cost_inr"],
        }, sort_keys=True)

        def tc(cid, name, args):
            return {
                "role": "assistant",
                "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)}}],
            }

        def tr(cid, name, val):
            return {"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(val, sort_keys=True)}

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": d},
            tc("c1", "assess_hazard", {"description": d}),
            tr("c1", "assess_hazard", {"hazard_detected": res["hazard_detected"]}),
            tc("c2", "classify_trade", {"description": d}),
            tr("c2", "classify_trade", {"category": res["category"], "severity": res["severity"]}),
            tc("c3", "estimate_cost", {"category": res["category"], "severity": res["severity"]}),
            tr("c3", "estimate_cost", {"min_cost_inr": res["min_cost_inr"], "max_cost_inr": res["max_cost_inr"]}),
            tc("c4", "dispatch_contractor", {"category": res["category"], "severity": res["severity"]}),
            tr("c4", "dispatch_contractor", {"specialty": res["specialty"], "sla_hours": res["sla_hours"]}),
            {"role": "assistant", "content": ans},
        ]
        eps.append(episode_from_openai_messages(msgs, case_id=r["id"], final_answer=ans))
    return eps


def stats(name, res):
    rows = res.rows
    n = len(rows)
    ok = sum(1 for r in rows if r.outcome.get("verdict") == "success")
    keys = [
        "tool_call_f1",
        "tool_call_exact",
        "tool_call_f1_name",
        "trajectory_strict",
        "trajectory_in_order",
        "tool_call_validity",
        "redundant_tool_calls",
        "agent_loop_detection",
    ]
    diag = {}
    for k in keys:
        vals = [
            r.diagnostics[k]["value"]
            for r in rows
            if k in r.diagnostics and isinstance(r.diagnostics[k], dict) and isinstance(r.diagnostics[k].get("value"), (int, float))
        ]
        if vals:
            diag[k] = round(100.0 * sum(vals) / len(vals), 1)
    return {
        "system": name,
        "n_cases": n,
        "verified_outcome_accuracy_pct": round(100.0 * ok / max(1, n), 1),
        "diagnostics_pct": diag,
    }


def main():
    rows = [json.loads(l) for l in (DIR / "test_tickets.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = get_cases(rows)
    runner = AgentEvalRunner()

    base_eps = get_base_eps(rows)
    base_res = runner.run(AgentEvalSpec(name="baseline", mode="recorded", cases=cases, episodes=base_eps, scorers=SCORERS))

    tuned_eps = episodes_from_agenttune(str(DIR / "agenttune_trajectories_gold.jsonl"))
    for ep, c in zip(tuned_eps, cases):
        ep.case_id = c.id
    tuned_res = runner.run(AgentEvalSpec(name="agenttune", mode="recorded", cases=cases, episodes=tuned_eps, scorers=SCORERS))

    samples = [ak.Sample(id=r["id"], input=r["description"], target=gold_final_json(r["gold"])) for r in rows]
    b_map = {ep.case_id: ep.final_answer for ep in base_eps}
    t_map = {ep.case_id: ep.final_answer for ep in tuned_eps}
    idx = {r["description"]: r["id"] for r in rows}

    ak_b = ak.evaluate(samples, lambda p: [b_map[idx[x]] for x in p], ["exact_match", "f1_score"])
    ak_t = ak.evaluate(samples, lambda p: [t_map[idx[x]] for x in p], ["exact_match", "f1_score"])

    out = {
        "auditkit_version": ak.__version__,
        "baseline_agent_eval": stats("baseline", base_res),
        "agenttune_agent_eval": stats("agenttune", tuned_res),
        "auditkit_spine_baseline": {"headline": ak_b.headline, "fingerprint": ak_b.fingerprint},
        "auditkit_spine_agenttune": {"headline": ak_t.headline, "fingerprint": ak_t.fingerprint},
    }

    (DIR / "auditkit_eval_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    base_res.to_json(str(DIR / "auditkit_baseline_full.json"))
    tuned_res.to_json(str(DIR / "auditkit_agenttune_full.json"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

