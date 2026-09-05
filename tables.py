"""
tables.py -- reproduces Table II (controller performance) and Table III
(residual oscillation vs. reporting delay) of the paper, exactly.

    python tables.py

Every number printed here should match the corresponding table in
paper/main.tex to the last decimal place.
"""
import numpy as np
from model import run_v2, metrics_v2

print("TABLE II -- Controller performance (eta=1.4, tau_d=1 yr, tau_max=1, T=25 yr)")
print(f"{'Controller':28s} {'peak N':>7s} {'IAE':>6s} {'ISCE':>6s} {'Yf':>6s} {'S [%]':>6s} {'breach':>7s}")
runs = [
    ("Open loop",                 "open_loop",   0.0),
    ("Static tax 22%",            "static",      1.0),
    ("CLF, no lag",               "clf",         0.0),
    ("CLF + Taylor pred.",        "clf_taylor",  1.0),
    ("CLF + Smith pred.",         "clf_smith",   1.0),
    ("CBF filter + Smith pred.",  "cbf_min",     1.0),
]
for name, mode, delay in runs:
    m = metrics_v2(run_v2(mode, delay=delay))
    print(f"{name:28s} {m['peak_N']:7.3f} {m['IAE']:6.2f} {m['ISCE']:6.2f} "
          f"{m['Yf']:6.2f} {100*m['sat']:6.1f} {'yes' if m['breach'] else 'no':>7s}")

print()
print("TABLE III -- Residual oscillation of N vs. reporting delay (kappa=1, unsaturated)")
print(f"{'tau_d [yr]':>10s} {'kappa*tau_d':>11s} {'Taylor pred.':>12s} {'Smith pred.':>12s}")
for td in [0.50, 0.90, 1.00, 1.25, 1.50, 2.00]:
    row = []
    for mode in ["clf_taylor", "clf_smith"]:
        r = run_v2(mode, delay=td, v_max=50.0, t_max=30.0, n=1500)
        tail = r["N"][r["t"] > 20]
        row.append(float(tail.max() - tail.min()))
    print(f"{td:10.2f} {td:11.2f} {row[0]:12.4f} {row[1]:12.4f}")
