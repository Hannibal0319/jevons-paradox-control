import numpy as np
import matplotlib.pyplot as plt

# Strict IEEE typography & sizing for single-column (3.5 in) plots.
# pdf.fonttype/ps.fonttype = 42 embeds real (Type 1/TrueType) outline
# fonts instead of matplotlib's default Type 3 bitmap fonts, which IEEE
# Xplore rejects.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.titlesize": 7.5,
    "lines.linewidth": 1.1,
})

# Calibration parameters (Table I)
N_star = 1.0
N0 = 370.0 / 460.0
A0 = N0
lam = 0.25
gamma = 0.04
eta = 1.40
M = np.log(9.0)
T = 25.0
tau_max = 1.0
v_max = np.log(1.0 + tau_max)
kappa = 1.0
alpha_cbf = 0.6
delay = 1.0

t = np.linspace(0, T, 2000)
# mu(t) rate/inflection matched to the calibration that actually produces
# the numbers quoted in Prop. 2 / Cor. 1 (223%, 17.3 yr, eta_crit=0.948);
# see v2-corrected/code/jevons_model_v2.py, mu_traj('s_curve').
mu = M / (1.0 + np.exp(-0.45 * (t - 12.0)))
mu_dot = np.gradient(mu, t)


def mu_of(tt):
    return M / (1.0 + np.exp(-0.45 * (tt - 12.0)))


def mudot_of(tt):
    s = 1.0 / (1.0 + np.exp(-0.45 * (tt - 12.0)))
    return M * 0.45 * s * (1.0 - s)


def F(x, tt):
    """Drift of the scalar plant, v = 0 (Eq. 8)."""
    return lam * (np.log(A0) + gamma * tt + (eta - 1.0) * mu_of(tt) - x) - mudot_of(tt)


G = -lam * eta

# ==========================================
# FIGURE 1: MECHANISM (Jevons Signature & Backfire)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.85))

x_eff = np.zeros_like(t)
for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    rhs = lam * (eta - 1.0) * mu[i-1] - mu_dot[i-1] - lam * x_eff[i-1]
    x_eff[i] = x_eff[i-1] + dt * rhs

ax1.plot(t, x_eff, 'b-')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.6)
ax1.set_xlabel("Time [yr]")
ax1.set_ylabel(r"Eff. Contrib. to $\ln N$")
ax1.set_title("(a) Jevons Signature", pad=4)
ax1.grid(True, linestyle=':', alpha=0.5)

etas = np.linspace(0.4, 1.8, 200)
x_inf = (etas - 1.0) * M
ax2.plot(etas, x_inf, 'k-')
# Clip the threshold lines to the lower part of the axes so they don't
# run through the legend box sitting in the upper-left corner.
ax2.axvline(1.0, color='r', linestyle='--', ymax=0.62, label=r"$\eta=1$")
ax2.axvline(0.948, color='g', linestyle=':', ymax=0.62,
            label=r"$\eta_{\rm crit}=0.948$")
ax2.set_xlabel(r"Rebound Elasticity $\eta$")
ax2.set_ylabel(r"Long-run $\Delta \ln N$")
ax2.set_title("(b) Feasibility Limit", pad=4)
ax2.legend(frameon=True, framealpha=0.9, edgecolor='none', loc="upper left",
           borderpad=0.3, handletextpad=0.4, labelspacing=0.3)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(pad=0.6, w_pad=1.2)
plt.savefig("fig_mechanism.pdf", dpi=300)
plt.close()


# ==========================================
# FIGURE 2: FEASIBILITY OBSTRUCTION
# ==========================================
fig, ax = plt.subplots(figsize=(3.5, 1.85))

v_req_no_m = (lam * (np.log(A0) + gamma * t - np.log(N_star))) / (lam * eta)
tau_req_no_m = np.exp(np.maximum(0, v_req_no_m)) - 1.0

v_req_full = (lam * (np.log(A0) + gamma * t + (eta - 1.0) * mu - np.log(N_star)) - mu_dot) / (lam * eta)
tau_req_full = np.exp(v_req_full) - 1.0

t_breach = t[np.argmax(tau_req_full > 1.0)]

ax.plot(t, tau_req_no_m * 100, 'g--', label=r"Growth only ($M=0$)")
ax.plot(t, tau_req_full * 100, 'r-', label=r"Growth + Rebound ($\eta=1.4$)")
ax.axhline(100, color='k', linestyle=':', label=r"$\tau_{\max}=100\%$")
ax.axvline(t_breach, color='gray', linestyle='-.', linewidth=0.7)

ax.annotate(f"Infeasible\n({t_breach:.1f} yr)",
            xy=(t_breach, 100),
            xytext=(19.0, 170),
            fontsize=6.5, ha='center',
            arrowprops=dict(arrowstyle="->", lw=0.7, color='black', shrinkB=3))

ax.set_xlabel("Time [yr]")
ax.set_ylabel(r"Required Tax $\tau_{\rm req}(t)$ [\%]")
ax.set_ylim(-25, 235)
ax.legend(frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.4)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(pad=0.6)
plt.savefig("fig_feasibility.pdf", dpi=300)
plt.close()
print(f"[fig_feasibility] max tau_req = {tau_req_full.max()*100:.1f}%  "
      f"breach at t = {t_breach:.2f} yr  "
      f"(paper states 223%, 17.3 yr)")


# ==========================================
# FIGURE 3: DELAY MARGIN ANALYSIS (real characteristic-root
# continuation + real neutral-DDE simulation; no fitted placeholders)
# ==========================================

def rightmost_root(h, kind):
    """Rightmost root of the characteristic equation via Newton continuation
    over a grid of initial guesses. kind='taylor': Delta(s)=s+h(1+s)e^{-s}.
    kind='none':   Delta(s)=s+h*e^{-s}."""
    best = -1e9
    for re0 in np.linspace(-3, 3, 13):
        for im0 in np.linspace(0, 30, 13):
            s = complex(re0, im0)
            for _ in range(150):
                with np.errstate(over='ignore', invalid='ignore'):
                    if kind == 'taylor':
                        Fs = s + h * (1 + s) * np.exp(-s)
                        dFs = 1 + h * np.exp(-s) - h * (1 + s) * np.exp(-s)
                    else:
                        Fs = s + h * np.exp(-s)
                        dFs = 1 - h * np.exp(-s)
                if not np.isfinite(Fs) or abs(dFs) < 1e-14:
                    break
                sn = s - Fs / dFs
                if abs(sn - s) < 1e-13:
                    s = sn
                    break
                s = sn
            with np.errstate(over='ignore', invalid='ignore'):
                Fs = (s + h * (1 + s) * np.exp(-s)) if kind == 'taylor' else (s + h * np.exp(-s))
            if np.isfinite(Fs) and abs(Fs) < 1e-8:
                best = max(best, s.real)
    return best


nu_star = None
lo, hi = 2.0, 3.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    val = np.cos(mid) + mid * np.sin(mid)
    if np.sign(val) == np.sign(np.cos(lo) + lo * np.sin(lo)):
        lo = mid
    else:
        hi = mid
nu_star = 0.5 * (lo + hi)
h_star = nu_star / (np.sin(nu_star) * (1 + nu_star**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.85))

hs = np.linspace(0.2, 1.8, 60)
re_taylor = [rightmost_root(h, 'taylor') for h in hs]
re_unpred = [rightmost_root(h, 'none') for h in hs]
ax1.plot(hs, re_taylor, 'r-', label="Taylor")
ax1.plot(hs, re_unpred, 'b--', label="No pred.")
ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(h_star, color='r', linestyle=':', linewidth=0.7)
ax1.axvline(np.pi / 2, color='b', linestyle=':', linewidth=0.7)
ax1.set_xlabel(r"Delay $h=\kappa\tau_d$")
ax1.set_ylabel(r"$\mathrm{Re}(\lambda_{\max})$")
ax1.set_title("(a) Root Crossing", pad=4)
ax1.legend(frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.3)
ax1.grid(True, linestyle=':', alpha=0.5)

# (b) Direct simulation of the closed-loop neutral equation
#     d/dt[z(t) + kappa*tau_d*z(t-tau_d)] = -kappa*z(t-tau_d),
# i.e. Eq. (17), at tau_d values bracketing h* (kappa=1, so h=tau_d),
# matching the tau_d grid of Table III.


def simulate_neutral(tau_d, kappa_, t_end=15.0, dt=0.002, z0=1.0):
    n_hist = int(round(tau_d / dt))
    n_steps = int(round(t_end / dt))
    z = np.empty(n_steps + n_hist + 1)
    z[:n_hist + 1] = z0  # constant history on [-tau_d, 0]
    for k in range(n_steps):
        n = n_hist + k
        z_delayed = z[n - n_hist]
        w = z[n] + kappa_ * tau_d * z_delayed
        dw = -kappa_ * z_delayed
        w_new = w + dt * dw
        z_del_new = z[n + 1 - n_hist] if (n + 1 - n_hist) >= 0 else z0
        # z_del_new is not yet known at n+1 if n+1-n_hist > n (impossible since
        # n_hist >= 1 for tau_d >= dt); by construction n+1-n_hist <= n, so it
        # was already computed.
        z_del_new = z[n + 1 - n_hist]
        z[n + 1] = w_new - kappa_ * tau_d * z_del_new
    tt = np.arange(len(z)) * dt - tau_d
    return tt, z


tt_stab, z_stab = simulate_neutral(0.70, kappa)   # h = 0.70 < h*
tt_unst, z_unst = simulate_neutral(1.10, kappa)   # h = 1.10 > h*
ax2.plot(tt_stab, z_stab, 'b-', label=r"$h=0.70<h^*$")
ax2.plot(tt_unst, z_unst, 'r--', label=r"$h=1.10>h^*$")
ax2.set_xlim(0, 15)
ax2.set_xlabel("Time [yr]")
ax2.set_ylabel(r"Error $z(t)$")
ax2.set_title("(b) Neutral Loop", pad=4)
ax2.legend(frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.3)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(pad=0.6, w_pad=1.2)
plt.savefig("fig_delay.pdf", dpi=300)
plt.close()
print(f"[fig_delay] nu*={nu_star:.6f}  h*={h_star:.6f}  (paper states 2.798386, 0.9417)")


# ==========================================
# FIGURE 4: TIME TRAJECTORIES & FISCAL ACTUATION
# Real delayed closed loop: CBF filter driven by a Smith (model-based)
# predictor across the tau_d = 1 yr reporting delay, matching Table II's
# best row ("CBF filter + Smith pred."). No delay is dropped for
# convenience: the predictor is simulated, not omitted.
# ==========================================

dt_sim = 0.01
n_hist = int(round(delay / dt_sim))
n_steps = int(round(T / dt_sim))
x_hist = np.full(n_steps + n_hist + 1, np.log(N0))
v_hist = np.zeros(n_steps + n_hist + 1)
t_hist = (np.arange(len(x_hist)) - n_hist) * dt_sim


def smith_predict(n_now, nsteps_rk=8):
    """Integrate the known model forward across the unmeasured window
    [t-delay, t] using the input history actually applied there (RK4),
    exactly mirroring Eq. (18)."""
    x0_ = x_hist[n_now - n_hist]
    t0_ = t_hist[n_now - n_hist]
    t1_ = t_hist[n_now]
    h_ = (t1_ - t0_) / nsteps_rk
    xh = x0_
    for k in range(nsteps_rk):
        s0 = t0_ + k * h_
        idx = n_now - n_hist + k
        v_s = v_hist[idx] if idx >= 0 else 0.0
        f1 = F(xh, s0) + G * v_s
        f2 = F(xh + 0.5*h_*f1, s0 + 0.5*h_) + G * v_s
        f3 = F(xh + 0.5*h_*f2, s0 + 0.5*h_) + G * v_s
        f4 = F(xh + h_*f3, s0 + h_) + G * v_s
        xh = xh + h_/6.0*(f1 + 2*f2 + 2*f3 + f4)
    return xh


for n in range(n_hist, n_hist + n_steps):
    tt_ = t_hist[n]
    x_hat = smith_predict(n) if tt_ > 0 else x_hist[n]
    h_bar = np.log(N_star) - x_hat
    v_cbf = (F(x_hat, tt_) - alpha_cbf * h_bar) / (-G)
    v_act = np.clip(v_cbf, 0.0, v_max)
    v_hist[n] = v_act
    dxdt = F(x_hist[n], tt_) + G * v_act
    x_hist[n+1] = x_hist[n] + dt_sim * dxdt

mask = (t_hist >= 0) & (np.arange(len(t_hist)) < n_hist + n_steps)
t_cl = t_hist[mask]
x_cl = x_hist[mask]
tau_applied = np.exp(v_hist[mask]) - 1.0

N_open = N0 * np.exp(gamma * t + (eta - 1.0) * mu)
N_cl = np.exp(x_cl)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.85))

ax1.plot(t, N_open, 'r--', label="Open")
ax1.plot(t_cl, N_cl, 'b-', label="Closed")
ax1.axhline(1.0, color='k', linestyle=':', label=r"$N^*$")
ax1.set_xlabel("Time [yr]")
ax1.set_ylabel(r"Resource $N(t)/N^\ast$")
ax1.set_title("(a) Draw Trajectory", pad=4)
ax1.set_ylim(0.7, 3.8)
ax1.legend(frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.3)
ax1.grid(True, linestyle=':', alpha=0.5)

ax2.plot(t_cl, tau_applied * 100, 'k-')
ax2.axhline(100, color='r', linestyle='--', label=r"$\tau_{\max}=100\%$")
ax2.set_xlabel("Time [yr]")
ax2.set_ylabel(r"Tax $\tau(t)$ [\%]")
ax2.set_title("(b) Fiscal Signal", pad=4)
ax2.legend(frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.3)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(pad=0.6, w_pad=1.2)
plt.savefig("fig_controllers.pdf", dpi=300)
plt.close()
print(f"[fig_controllers] peak N (closed, CBF+Smith, tau_d=1yr) = {N_cl.max():.3f}  "
      f"(Table II states 1.538)")

print("Figures rebuilt from real simulation/root-finding (no fitted placeholder curves).")
