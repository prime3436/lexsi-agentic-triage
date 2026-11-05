import json
import random
from pathlib import Path

SEED = 42
N_TPL = 12
OUT = Path(__file__).parent / "data"

SCENARIOS = [
    ("Safety & Security", "P1_CRITICAL", "gas_leak", [
        "there is a strong rotten egg smell in the {room}",
        "I think there is a gas leak near the {appliance}, smells like sulfur",
        "LPG smell coming from the {room}, please come fast",
        "smell of gas in the {room} and my head hurts",
        "cylinder valve near the {appliance} in {room} is hissing and making everyone dizzy",
        "kitchen mein {appliance} ke paas se LPG ki tez badbu aa rahi hai",
    ]),
    ("Electrical", "P1_CRITICAL", "electrical_arcing", [
        "the {outlet} in the {room} is sparking and has a burn mark",
        "smoke coming out of the {outlet}, smells like burning plastic",
        "I got an electric shock from the {outlet} in the {room}",
        "scorch marks around the switchboard in the {room} and it is buzzing",
        "{room} ka {outlet} kala pad gaya hai aur chingari nikal rahi hai",
        "the {outlet} in the {room} melted my charger plug and feels hot to touch",
    ]),
    ("Plumbing", "P1_CRITICAL", "water_flooding", [
        "a pipe burst in the {room} and water is spraying everywhere",
        "water is gushing from under the {fixture}, the floor is flooding",
        "sewage is overflowing from the toilet in the {room}",
        "supply line snapped behind the {fixture}, flooding the {room}",
        "ankle deep water across the {room} floor coming from behind the {fixture}",
        "{room} mein {fixture} ke neeche ka pipe toot gaya hai aur paani bhar raha hai",
    ]),
    ("HVAC / Climate Control", "P2_HIGH", None, [
        "AC in the {room} stopped cooling and shows error code {code}",
        "air conditioner is leaking water and the coil is freezing in the {room}",
        "AC compressor stopped working, it is a heatwave, display shows {code}",
        "split unit in the {room} blows warm air only and flashes {code} on the panel",
    ]),
    ("HVAC / Climate Control", "P3_MEDIUM", None, [
        "AC in the {room} is making a rattling noise but still cools",
        "the AC filter looks dirty and airflow is weak in the {room}",
        "thermostat in the {room} is slow to respond",
    ]),
    ("Safety & Security", "P2_HIGH", None, [
        "the front door deadbolt is jammed and I cannot lock it",
        "door lock is broken in the {room}, anyone can enter",
        "main entrance latch snapped off in the {room}, flat cannot be secured tonight",
    ]),
    ("Safety & Security", "P3_MEDIUM", None, [
        "the window latch in the {room} is loose",
        "balcony window does not close properly in the {room}",
    ]),
    ("Plumbing", "P3_MEDIUM", None, [
        "the sink in the {room} is draining very slowly",
        "bathroom drain is clogged and water is standing in the {room}",
        "{fixture} in the {room} takes 20 minutes to empty after washing",
    ]),
    ("Plumbing", "P4_LOW", None, [
        "the faucet in the {room} has a minor drip",
        "tap in the {room} drips slowly at night",
    ]),
    ("Electrical", "P3_MEDIUM", None, [
        "the light in the {room} keeps flickering",
        "circuit breaker tripped once in the {room} but works after reset",
        "ceiling fan regulator in the {room} is not working properly",
        "two {outlet}s in the {room} have no power though the rest of the flat works",
    ]),
    ("General Maintenance", "P4_LOW", None, [
        "the wardrobe hinge in the {room} is loose",
        "paint is peeling on the wall in the {room}",
        "cabinet door in the {room} is misaligned",
        "some tile grout is cracked in the {room}",
    ]),
]

SLOTS = {
    "room": ["kitchen", "bathroom", "master bedroom", "living room", "balcony", "hallway", "guest room"],
    "appliance": ["stove", "geyser", "cooking range", "gas cylinder"],
    "outlet": ["socket", "power outlet", "wall switch", "plug point"],
    "fixture": ["sink", "wash basin", "kitchen tap", "water heater"],
    "code": ["E4", "E1", "E2", "F3"],
}
PRE = ["", "", "Hi, ", "Urgent: ", "Hello team, ", "Please help - "]
SUF = ["", "", " Please send someone.", " Unit has been like this since morning.", " Thanks!"]

SPEC = {
    "Plumbing": "Licensed Master Plumber",
    "Electrical": "Certified Master Electrician",
    "HVAC / Climate Control": "EPA-Certified HVAC Technician",
    "Safety & Security": "Licensed General Handyman",
    "General Maintenance": "Licensed General Handyman",
}
SLA = {"P1_CRITICAL": 2, "P2_HIGH": 6, "P3_MEDIUM": 24, "P4_LOW": 72}
COST = {
    "P1_CRITICAL": (3500, 9500), "P2_HIGH": (2800, 7500),
    "P3_MEDIUM": (800, 3000), "P4_LOW": (600, 2000),
}

SYSTEM = (
    "You are a property-maintenance triage agent. "
    "Use the tools in order: assess_hazard, classify_trade, estimate_cost, dispatch_contractor. "
    "Then return the final ticket JSON."
)


def make_ticket(rng, idx, cat, sev, haz, tpl):
    txt = tpl
    for k, opts in SLOTS.items():
        while "{" + k + "}" in txt:
            txt = txt.replace("{" + k + "}", rng.choice(opts), 1)
    txt = rng.choice(PRE) + txt + rng.choice(SUF)
    spec = "24/7 Water & Fire Remediation" if (cat == "Plumbing" and sev == "P1_CRITICAL") else SPEC[cat]
    sla = 1 if haz == "gas_leak" else SLA[sev]
    return {
        "id": f"case-{idx:04d}",
        "description": txt.strip(),
        "gold": {
            "category": cat, "severity": sev,
            "hazard_detected": haz is not None, "hazard_type": haz,
            "specialty": spec, "sla_hours": sla,
            "min_cost_inr": COST[sev][0], "max_cost_inr": COST[sev][1],
        },
    }


def gold_final_json(g):
    return json.dumps({
        "category": g["category"],
        "severity": g["severity"],
        "hazard_detected": g["hazard_detected"],
        "sla_hours": g["sla_hours"],
        "specialty": g["specialty"],
        "min_cost_inr": g["min_cost_inr"],
        "max_cost_inr": g["max_cost_inr"],
    }, sort_keys=True)


def to_traj(t):
    g, d = t["gold"], t["description"]

    def tc(cid, name, args):
        return {
            "role": "assistant",
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)},
            }],
        }

    def tr(cid, name, val):
        return {"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(val, sort_keys=True)}

    return {
        "id": t["id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": d},
            tc("call_1", "assess_hazard", {"description": d}),
            tr("call_1", "assess_hazard", {"hazard_detected": g["hazard_detected"], "hazard_type": g["hazard_type"]}),
            tc("call_2", "classify_trade", {"description": d}),
            tr("call_2", "classify_trade", {"category": g["category"], "severity": g["severity"]}),
            tc("call_3", "estimate_cost", {"category": g["category"], "severity": g["severity"]}),
            tr("call_3", "estimate_cost", {"min_cost_inr": g["min_cost_inr"], "max_cost_inr": g["max_cost_inr"]}),
            tc("call_4", "dispatch_contractor", {"category": g["category"], "severity": g["severity"]}),
            tr("call_4", "dispatch_contractor", {"specialty": g["specialty"], "sla_hours": g["sla_hours"]}),
            {"role": "assistant", "content": gold_final_json(g)},
        ],
    }


def to_at_row(t):
    g, d = t["gold"], t["description"]
    traj = to_traj(t)
    steps = [
        {
            "step_number": 0,
            "thought": "check hazard",
            "action": {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "assess_hazard", "arguments": {"description": d}}}]},
            "observation": json.dumps({"hazard_detected": g["hazard_detected"], "hazard_type": g["hazard_type"]}),
        },
        {
            "step_number": 1,
            "thought": "classify trade",
            "action": {"tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "classify_trade", "arguments": {"description": d}}}]},
            "observation": json.dumps({"category": g["category"], "severity": g["severity"]}),
        },
        {
            "step_number": 2,
            "thought": "estimate cost",
            "action": {"tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "estimate_cost", "arguments": {"category": g["category"], "severity": g["severity"]}}}]},
            "observation": json.dumps({"min_cost_inr": g["min_cost_inr"], "max_cost_inr": g["max_cost_inr"]}),
        },
        {
            "step_number": 3,
            "thought": "dispatch contractor",
            "action": {"tool_calls": [{"id": "call_4", "type": "function", "function": {"name": "dispatch_contractor", "arguments": {"category": g["category"], "severity": g["severity"]}}}]},
            "observation": json.dumps({"specialty": g["specialty"], "sla_hours": g["sla_hours"]}),
        },
    ]
    return {
        "trajectory_id": t["id"],
        "task": d,
        "steps": steps,
        "final_response": gold_final_json(g),
        "reward": 1.0,
        "metadata": {
            "tool_call_count": 4,
            "target": gold_final_json(g),
            "conversation": traj["messages"],
        },
    }


def main():
    rng = random.Random(SEED)
    seen, rows = set(), []
    idx = 1
    for cat, sev, haz, tpls in SCENARIOS:
        for tpl in tpls:
            for _ in range(N_TPL):
                t = make_ticket(rng, idx, cat, sev, haz, tpl)
                if t["description"] not in seen:
                    seen.add(t["description"])
                    rows.append(t)
                    idx += 1
    rng.shuffle(rows)
    n = len(rows)
    sp = {
        "train": rows[: int(n * 0.7)],
        "val": rows[int(n * 0.7): int(n * 0.85)],
        "test": rows[int(n * 0.85):],
    }
    OUT.mkdir(exist_ok=True)
    with open(OUT / "tickets.jsonl", "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps(t) + "\n")
    for k, items in sp.items():
        with open(OUT / f"{k}.jsonl", "w", encoding="utf-8") as f:
            for t in items:
                f.write(json.dumps(to_traj(t)) + "\n")
    with open(OUT / "test_tickets.jsonl", "w", encoding="utf-8") as f:
        for t in sp["test"]:
            f.write(json.dumps(t) + "\n")
    with open(OUT / "agenttune_train.jsonl", "w", encoding="utf-8") as f:
        for t in sp["train"]:
            f.write(json.dumps({
                "id": t["id"],
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": t["description"]},
                ],
                "answer": gold_final_json(t["gold"]),
            }) + "\n")
    with open(OUT / "agenttune_trajectories_gold.jsonl", "w", encoding="utf-8") as f:
        for t in sp["test"]:
            f.write(json.dumps(to_at_row(t)) + "\n")
    print({k: len(v) for k, v in sp.items()}, "total", n)


if __name__ == "__main__":
    main()
