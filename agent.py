from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TriageBaseline:
    CORPUS = {
        "Plumbing": "pipe burst copper supply line water spraying flooding kitchen bathroom sink drain backed up overflowing toilet leak sewage standing water faucet drip valve pressure pex",
        "Electrical": "outlet sparking scorch burn mark electric shock smoke circuit breaker tripped mcb buzzing short circuit switchboard burning plastic smell bare wire voltage flicker",
        "HVAC / Climate Control": "air conditioner ac cooling compressor warm air thermostat error code e4 e1 e2 refrigerant coil freezing leak fan motor condenser filter ventilation climate",
        "Safety & Security": "gas leak natural gas lpg smell rotten eggs sulfur odor explosion hazard door lock jammed deadbolt broken entry window latch security emergency",
        "General Maintenance": "wardrobe hinge loose cabinet door alignment wall paint peeling cosmetic crack tile grout furniture general routine maintenance",
    }
    HAZ = "gas leak rotten eggs sulfur spark fire burning smoke scorch burst pressurized flooding explosion"

    def __init__(self):
        self.cats = list(self.CORPUS.keys())
        docs = [self.CORPUS[c] for c in self.cats] + [self.HAZ]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        mat = self.vec.fit_transform(docs)
        self.t_mat = mat[:-1]
        self.h_vec = mat[-1]

    def run(self, text: str) -> dict:
        q = self.vec.transform([text.lower()])
        sims = cosine_similarity(q, self.t_mat)[0]
        h_sim = float(cosine_similarity(q, self.h_vec)[0][0])
        cat = self.cats[int(np.argmax(sims))]

        dl = text.lower()
        haz = False
        h_type = None
        sev = "P4_LOW"
        sla = 72

        if "gas" in dl or "rotten egg" in dl or "sulfur" in dl:
            haz, h_type, sev, cat, sla = True, "gas_leak", "P1_CRITICAL", "Safety & Security", 1
        elif cat == "Electrical" and (h_sim > 0.25 or any(k in dl for k in ["spark", "fire", "smoke", "scorch", "burn", "shock"])):
            haz, h_type, sev, sla = True, "electrical_arcing", "P1_CRITICAL", 2
        elif cat == "Plumbing" and (h_sim > 0.25 or any(k in dl for k in ["burst", "spray", "flood", "gush", "sewage", "overflow"])):
            haz, h_type, sev, sla = True, "water_flooding", "P1_CRITICAL", 2
        elif cat == "HVAC / Climate Control":
            if any(k in dl for k in ["code", "e4", "stopped", "leak", "freeze", "smoke", "heatwave"]):
                sev, sla = "P2_HIGH", 6
            else:
                sev, sla = "P3_MEDIUM", 24
        elif cat == "Plumbing":
            sev = "P3_MEDIUM" if any(k in dl for k in ["slow", "clog", "drain"]) else "P4_LOW"
            sla = 24 if sev == "P3_MEDIUM" else 72
        elif cat == "Safety & Security":
            sev = "P2_HIGH" if ("lock" in dl or "deadbolt" in dl) else "P3_MEDIUM"
            sla = 6 if sev == "P2_HIGH" else 24

        if cat == "Plumbing":
            if sev == "P1_CRITICAL":
                lo, hi, spec = 3500, 9500, "24/7 Water & Fire Remediation"
            else:
                lo, hi, spec = 800, 2200, "Licensed Master Plumber"
        elif cat == "Electrical":
            lo, hi, spec = 2200, 6000, "Certified Master Electrician"
        elif cat == "HVAC / Climate Control":
            lo, hi, spec = 2800, 7500, "EPA-Certified HVAC Technician"
        else:
            lo, hi, spec = 600, 2000, "Licensed General Handyman"

        return {
            "category": cat,
            "severity": sev,
            "hazard_detected": haz,
            "hazard_type": h_type,
            "sla_hours": sla,
            "specialty": spec,
            "min_cost_inr": lo,
            "max_cost_inr": hi,
        }
