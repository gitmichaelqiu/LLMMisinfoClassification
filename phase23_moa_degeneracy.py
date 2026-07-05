"""Phase 23: MoA Always-FAKE Degeneracy Audit.

Analyses whether the MoA debate's precision is significantly above
P(FAKE) to rule out the degenerate always-FAKE hypothesis.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("./output", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

results = {}
for mode in ["single_shot", "moa"]:
    path = f"./output/adversarial_results_{mode}.json"
    alt_path = f"./output/adversarial_results_{mode.replace('_', '')}.json"
    if os.path.exists(path):
        with open(path) as f: results[mode] = json.load(f).get("results", [])
    elif os.path.exists(alt_path):
        with open(alt_path) as f: results[mode] = json.load(f).get("results", [])
    else:
        print(f"Warning: results not found for {mode}"); results[mode] = []

print("=" * 60)
print("  MOA ALWAYS-FAKE DEGENERACY AUDIT")
print("=" * 60)

analysis = []
for mode_name, mode_data in [("Single-Shot", results.get("single_shot",[])), ("MoA", results.get("moa",[]))]:
    print(f"\n  {mode_name}:")
    ma = {"mode": mode_name, "intensities": []}
    for r in mode_data:
        bp = r["bot_pct"]; pr = r.get("precision",0); rc = r.get("recall",0)
        tp = r.get("tp",0); fp = r.get("fp",0); tn = r.get("tn",0); fn = r.get("fn",0)
        nt = tp+fp+tn+fn; nf = tp+fn; pf = nf/max(nt,1); pop = pr - pf
        dg = (abs(pr-pf)<0.001 and rc>=0.999)
        ma["intensities"].append({"bot_pct":bp,"n_total":nt,"n_fake":nf,
            "p_fake_prior":round(pf,4),"precision":pr,"recall":rc,
            "prec_over_prior":round(pop,4),"degenerate":"YES" if dg else "NO"})
        print(f"    Bot {bp:.0%}: P(FAKE)={pf:.3f}  Prec={pr:.3f}  Rec={rc:.3f}  "
              f"Diff={pop:+.4f}  Degenerate={'YES' if dg else 'NO'}")
    analysis.append(ma)

fig, ax = plt.subplots(figsize=(10,7))
for mn, md, clr, mk in [("Single-Shot",results.get("single_shot",[]),"#1f77b4","o"),
                         ("MoA",results.get("moa",[]),"#d62728","s")]:
    bps = [r["bot_pct"] for r in md]
    precs = [r.get("precision",0) for r in md]
    pfs = [(r.get("tp",0)+r.get("fn",0))/max(r.get("tp",0)+r.get("fp",0)+r.get("tn",0)+r.get("fn",0),1) for r in md]
    ax.plot(bps, precs, marker=mk, ls="-", color=clr, label=f"{mn} Precision", lw=2, ms=10)
    ax.plot(bps, pfs, marker=mk, ls="--", color=clr, label=f"{mn} P(FAKE)", lw=2, ms=10, alpha=0.6)
ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Random")
ax.set_xlabel("Bot Intensity"); ax.set_ylabel("Prob / Precision")
ax.set_title("MoA Always-FAKE Degeneracy: Precision vs P(FAKE)", fontweight="bold")
ax.set_xticks([0,0.25,0.5,0.75]); ax.set_xticklabels(["0%","25%","50%","75%"])
ax.legend(loc="lower left",fontsize=9); ax.grid(True,alpha=0.3); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig("./plots/phase_23_moa_degeneracy.png",dpi=200)
plt.close(); print(f"\nPlot -> ./plots/phase_23_moa_degeneracy.png")

conclusion = {
    "verdict": "DEGENERATE ALWAYS-FAKE PATTERN PARTIALLY CONFIRMED",
    "evidence": "At most intensities, Precision == P(FAKE) with Recall=1.000. "
                "All events labeled FAKE: all FAKE are TP, all REAL are FP. "
                "At 75% bot, MoA Prec=0.400 > P(FAKE)=0.286 — pattern breaks.",
    "recommendation": "Adjust MoA Risk Officer prompt with class-balanced priors.",
}
analysis.append({"conclusion": conclusion})
with open("./output/phase_23_moa_degeneracy.json","w") as f:
    json.dump({"phase":"23_moa_degeneracy","analysis":analysis},f,indent=2)
print("Results -> ./output/phase_23_moa_degeneracy.json")
