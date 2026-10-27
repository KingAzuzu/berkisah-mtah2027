"""Apply the preregistered thresholds to results_a.json / results_b.json -> results.json (+ two plots)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import idsse

a = json.load(open(idsse.ROOT / "results_a.json"))
b = json.load(open(idsse.ROOT / "results_b.json"))

v = {}
v["A0"] = {"value": a["n_valid"], "threshold": ">= 50 corners", "verdict": "PASS" if a["n_valid"] >= 50 else "UNDERPOWERED"}
r = a["A1_rho"]
v["A1"] = {"value": r, "threshold": "pass >= 0.6, weak 0.4-0.6", "verdict": "PASS" if r >= 0.6 else "WEAK" if r >= 0.4 else "FAIL"}
v["A2"] = {"value": a.get("A2_p"), "threshold": "KW p < 0.05 (does not gate)", "verdict": "PASS" if a.get("A2_p", 1) < 0.05 else "FAIL"}
r3 = a["A3_rho_primary_rep0"]
v["A3"] = {"value": r3, "threshold": ">= 0.5", "verdict": "PASS" if r3 >= 0.5 else "FAIL"}
v["Test A"] = "PASS" if all(v[k]["verdict"] == "PASS" for k in ("A0", "A1", "A3")) else "FAIL"
v["B0"] = {"value": {"n_losses": b["n_losses"], "base_Y1": b["base_rate_Y1"], "base_Y2": b["base_rate_Y2"]}, "threshold": "report", "verdict": "REPORTED"}
lo, hi = b["Y1"]["ci95_S"]
v["B1"] = {"value": {"coef_S": b["Y1"]["coef_S"], "ci95": [lo, hi]}, "threshold": "coef < 0 and CI excludes 0", "verdict": "PASS" if b["Y1"]["coef_S"] < 0 and hi < 0 else "FAIL"}
g = b["Y1"]["auc_gain"]
v["B2"] = {"value": g, "threshold": "AUC gain >= 0.03", "verdict": "PASS" if g >= 0.03 else "FAIL"}
lo2, hi2 = b["Y2"]["ci95_S"]
v["B3"] = {"value": {"coef_S": b["Y2"]["coef_S"], "ci95": [lo2, hi2]}, "threshold": "as B1 on Y2 (does not gate)", "verdict": "PASS" if b["Y2"]["coef_S"] < 0 and hi2 < 0 else "FAIL"}
v["Test B"] = "PASS" if v["B1"]["verdict"] == v["B2"]["verdict"] == "PASS" else "FAIL"
ta, tb = v["Test A"] == "PASS", v["Test B"] == "PASS"
v["decision"] = ("A headline, B supporting" if ta and tb else "A headline" if ta else "B headline" if tb else
                 "neither passes: descriptive fallback, re-choose after 26 Nov")
json.dump({"verdicts": v, "test_a": a, "test_b": b}, open(idsse.ROOT / "results.json", "w"), indent=1)
print(json.dumps(v, indent=1))

# plots
p = np.load(idsse.ROOT / "a_points.npz")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(p["idx"], p["ref"], s=18)
ax[0].set(xlabel="identity-free index", ylabel="reference stickiness", title=f"A1: Spearman {a['A1_rho']:.2f} (n={a['n_valid']})")
d = np.load(idsse.ROOT / "b_losses.npy")  # columns: S, ball_x, Y1, Y2, in_zone
Sc = np.clip(d[:, 0], -3, 6)
levels = np.unique(Sc)
rate = [d[Sc == s, 2].mean() for s in levels]
n = [(Sc == s).sum() for s in levels]
ax[1].bar(levels, rate, width=0.8)
for s, rt, k in zip(levels, rate, n):
    ax[1].text(s, rt, f"n={k}", ha="center", va="bottom", fontsize=7)
ax[1].set(xlabel="rest-defence surplus S (clipped to [-3, 6])", ylabel="P(Y1)", title="B: raw P(Y1) by S (not adjusted for ball_x)")
fig.tight_layout()
fig.savefig(idsse.ROOT / "plots.png", dpi=130)
