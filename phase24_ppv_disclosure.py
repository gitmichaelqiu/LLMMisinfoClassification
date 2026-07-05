"""Phase 24: Translate OOD Accuracy to Operational PPV.

Maps OOD TPR/FPR to Bayesian PPV across base rates 0.01%-50%.
Annotates PPV at the three crossover thresholds (4.2%, 5.8%, 8.3%).
"""

import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("./output", exist_ok=True); os.makedirs("./plots", exist_ok=True)

# OOD performance from Phase 22 (Single-Shot CoT from paper §6)
# TPR = recall = 0.92, FPR = 0.021 (from paper §5 table)
tpr = 0.92; fpr = 0.021
print(f"OOD TPR={tpr}, FPR={fpr}")

base_rates = np.logspace(-4, -0.3, 200)  # 0.01% to 50%
ppv = (tpr * base_rates) / (tpr * base_rates + fpr * (1 - base_rates))

crossover_thresholds = [0.042, 0.058, 0.083]
crossover_ppvs = {}
for ct in crossover_thresholds:
    idx = np.argmin(np.abs(base_rates - ct))
    ppv_val = ppv[idx]
    crossover_ppvs[f"crossover_{ct:.1%}"] = round(ppv_val, 4)
    print(f"  At P(Fake)={ct:.1%}: PPV={ppv_val:.2%}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.semilogx(base_rates, ppv, "b-", lw=2.5, label=f"PPV (TPR={tpr}, FPR={fpr})")
ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Random (PPV=0.5)")

for ct in crossover_thresholds:
    idx = np.argmin(np.abs(base_rates - ct))
    ax.axvline(ct, color="red", ls="--", alpha=0.3)
    ax.plot(ct, ppv[idx], "ro", ms=8)
    ax.annotate(f"P(Fake)={ct:.1%}\nPPV={ppv[idx]:.2%}",
               xy=(ct, ppv[idx]), xytext=(ct*2, ppv[idx]-0.1),
               fontsize=9, arrowprops=dict(arrowstyle="->"),
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))

ax.set_xlabel("Fake News Base Rate P(Fake) (log scale)")
ax.set_ylabel("Positive Predictive Value (PPV)")
ax.set_title("Operational PPV: OOD Accuracy → Bayesian Precision", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3); ax.set_xlim([1e-4, 0.5])

plt.tight_layout(); plt.savefig("./plots/phase_24_ppv_curve.png", dpi=200); plt.close()
print("PPV curve -> ./plots/phase_24_ppv_curve.png")

out = {"phase":"24_ppv_disclosure","tpr":tpr,"fpr":fpr,"crossover_ppvs":crossover_ppvs}
with open("./output/phase_24_ppv_disclosure.json","w") as f: json.dump(out,f,indent=2)
print("Saved -> ./output/phase_24_ppv_disclosure.json")
