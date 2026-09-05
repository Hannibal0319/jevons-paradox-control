"""
jevons_model_v2.py — dimensionally consistent, scale-free Jevons plant.
=======================================================================
Fixes three defects of the v1 plant:
  (1) v1 declares e = Y/N yet prescribes e(t) exogenously AND gives N, Y
      independent dynamics -> over-determined and violated by up to 6.4x.
  (2) v1's additive disturbance D_N * edot is not invariant to the units of
      the efficiency index: rescaling e rescales the predicted rebound.
  (3) v1 has no efficiency-savings channel: D_N > 0 makes every efficiency
      gain increase N, so backfire is assumed rather than derived.

Corrected structure
-------------------
Exogenous technology:   mu(t) := ln e(t)          (scale-free; units of e
                                                   only shift mu by a constant)
Identity:               N = Y / e   =>   ln N = ln Y - mu
Service demand (partial adjustment to the effective service price
p_eff = (1+tau) p_0 / e, with demand elasticity eta > 0):

    d(ln Y)/dt = lambda ( ln Y*(t) - ln Y ),
    ln Y*(t)   = ln A_0 + gamma t + eta*mu(t) - eta*v(t),   v := ln(1+tau)

Substituting the identity with x := ln N gives the scalar plant

    xdot = lambda [ ln A_0 + gamma t + (eta-1) mu(t) - eta v - x ] - mudot(t)
         =: F(x,t) + G v,        G = -lambda*eta   (constant)

Properties this buys
--------------------
  * affine in the control v = ln(1+tau); actuation gain G is CONSTANT, so the
    g_N(x) = 0 singularity of v1 disappears entirely.
  * scale-free: rescaling e shifts mu by a constant, absorbed into ln A_0.
  * the long-run efficiency elasticity of consumption is exactly (eta - 1):
        d x_ss / d mu = eta - 1
    so BACKFIRE IFF eta > 1 emerges structurally instead of being assumed.
  * the -mudot term is a genuine short-run SAVINGS channel absent from v1:
    fast efficiency gains transiently reduce N, and the rebound arrives with
    the demand adjustment lag 1/lambda.
  * x = ln N is the geodesic coordinate of the metric g_NN = 1/(sigma N^2),
    so the "geometric" CLF is the canonical one here, not an ornament.
"""

import numpy as np
from scipy.integrate import solve_ivp

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------- calibration
LAMBDA   = 0.25      # demand adjustment rate [1/yr]  (half-life ~2.8 yr)
GAMMA    = 0.04      # autonomous income/activity growth of service demand [1/yr]
T_HORIZON = 25.0     # policy window 2025-2050 [yr]
ETA_NOM  = 1.40      # demand (rebound) elasticity; eta>1 <=> backfire
M_EFF    = np.log(9.0)   # total log-efficiency gain: 9x over the horizon
E_RATE_V2 = 0.45     # logistic rate of mu(t) [1/yr]
E_INFL   = 12.0      # inflection year
N_STAR   = 1.0       # ceiling, normalised (= 460 TWh/yr)
N_INIT   = 370.0/460.0
TAU_MAX  = 1.0       # 100% tax
V_MAX    = np.log(1.0 + TAU_MAX)
K_GAIN   = 2.0
DELAY    = 1.0


def mu_traj(t, kind='s_curve', M=M_EFF, rate=E_RATE_V2, infl=E_INFL, rng_shocks=None):
    """Log-efficiency level mu(t) and its velocity mudot(t)."""
    if kind == 's_curve':
        s = 1.0/(1.0 + np.exp(-rate*(t-infl)))
        return M*s, M*rate*s*(1.0-s)
    if kind == 'step':
        r = 8.0
        s = 1.0/(1.0 + np.exp(-r*(t-infl)))
        return M*s, M*r*s*(1.0-s)
    if kind == 'stochastic':
        # sum of logistic ramps; total gain still M (scale-free)
        tot_mu, tot_md = 0.0, 0.0
        for t_i, w_i, a_i in rng_shocks:
            s = 1.0/(1.0 + np.exp(-(t-t_i)/w_i))
            tot_mu += a_i*s
            tot_md += a_i*s*(1.0-s)/w_i
        return tot_mu, tot_md
    raise ValueError(kind)


def make_stoch(seed=42, M=M_EFF, n=6):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.uniform(4.0, 32.0, n))
    ws = rng.uniform(0.8, 2.5, n)
    as_ = rng.uniform(0.5, 1.5, n)
    as_ = as_ / as_.sum() * M          # normalise so total gain = M
    return list(zip(ts, ws, as_))


class JevonsV2:
    def __init__(self, eta=ETA_NOM, lam=LAMBDA, gamma=GAMMA, K=K_GAIN,
                 delay=0.0, v_max=V_MAX, shock='s_curve', seed=42,
                 x_star=np.log(N_STAR), cbf_alpha=0.6, M=M_EFF):
        self.eta, self.lam, self.gamma = eta, lam, gamma
        self.K, self.delay, self.v_max = K, delay, v_max
        self.shock, self.x_star, self.cbf_alpha = shock, x_star, cbf_alpha
        self.M = M
        self.stoch = make_stoch(seed, M=M) if shock == 'stochastic' else None
        self.x0 = np.log(N_INIT)
        # anchor lnA_0 so that x(0) is the initial long-run equilibrium at v=0
        mu0, _ = self.mu(0.0)
        self.lnA0 = self.x0 - self.gamma*0.0 - (self.eta-1.0)*mu0
        self.hist_t, self.hist_x, self.hist_v = [], [], []

    def mu(self, t):
        return mu_traj(t, self.shock, M=self.M, rng_shocks=self.stoch)

    # long-run target for ln N at control v
    def x_bar(self, t, v):
        mu, _ = self.mu(t)
        return self.lnA0 + self.gamma*t + (self.eta-1.0)*mu - self.eta*v

    def F(self, x, t):
        """Drift with v = 0."""
        mu, mud = self.mu(t)
        return self.lam*(self.lnA0 + self.gamma*t + (self.eta-1.0)*mu - x) - mud

    @property
    def G(self):
        return -self.lam*self.eta        # constant actuation gain

    # ----- control laws (all act on the log coordinate x = ln N) -----
    def v_clf(self, x_est, t):
        """CLF on V = 0.5 (x - x*)^2, contraction rate kappa = K/2."""
        kappa = 0.5*self.K
        v = (self.F(x_est, t) + kappa*(x_est - self.x_star))/(-self.G)
        return float(np.clip(v, 0.0, self.v_max))

    def v_cbf(self, x_est, t):
        """
        Barrier h = x* - x >= 0 (safe set {N <= N*}), enforce hdot >= -alpha h.

            hdot = -xdot = -(F + G v) >= -alpha h
        =>  F + G v <= alpha h,   and since G = -lambda*eta < 0 the division flips:
            v >= ( F - alpha h ) / (-G).

        Note the MINUS on the alpha term: when the state is far below the
        ceiling (h large) the constraint is slack and v_cbf goes negative, i.e.
        no tax is required.  (A plus sign here makes the filter demand a large
        tax while still far from the boundary.)
        """
        h = self.x_star - x_est
        v = (self.F(x_est, t) - self.cbf_alpha*h)/(-self.G)
        return float(v)

    def v_required(self, t, x=None):
        """Unclipped tax needed to hold x at the ceiling (feasibility probe)."""
        x = self.x_star if x is None else x
        return self.F(x, t)/(-self.G)

    # ----- delay handling -----
    def x_delayed(self, t):
        if self.delay <= 0 or len(self.hist_t) < 2:
            return self.hist_x[-1] if self.hist_x else self.x0
        tt = t - self.delay
        if tt <= self.hist_t[0]:
            return self.hist_x[0]
        return float(np.interp(tt, self.hist_t, self.hist_x))

    def v_delayed(self, t):
        if self.delay <= 0 or len(self.hist_t) < 2:
            return 0.0
        tt = t - self.delay
        n = min(len(self.hist_t), len(self.hist_v))
        if n < 2 or tt <= self.hist_t[0]:
            return self.hist_v[0] if self.hist_v else 0.0
        return float(np.interp(tt, self.hist_t[:n], self.hist_v[:n]))

    def predict_taylor(self, t):
        xd = self.x_delayed(t)
        vd = self.v_delayed(t)
        return xd + self.delay*(self.F(xd, t-self.delay) + self.G*vd)

    def predict_smith(self, t, nsteps=12):
        """Integrate the model across the unmeasured window (open loop, L=0)."""
        t0 = max(t - self.delay, self.hist_t[0] if self.hist_t else 0.0)
        if t <= t0 + 1e-12:
            return self.x_delayed(t)
        xh = self.x_delayed(t)
        dt = (t - t0)/nsteps
        for k in range(nsteps):
            s = t0 + k*dt
            vs = self.v_delayed(s + self.delay)   # the input actually applied at s
            f1 = self.F(xh, s) + self.G*vs
            f2 = self.F(xh + 0.5*dt*f1, s + 0.5*dt) + self.G*vs
            f3 = self.F(xh + 0.5*dt*f2, s + 0.5*dt) + self.G*vs
            f4 = self.F(xh + dt*f3, s + dt) + self.G*vs
            xh = xh + dt/6.0*(f1 + 2*f2 + 2*f3 + f4)
        return xh

    # ----- plant ODE -----
    def rhs(self, t, xv, mode):
        x = xv[0]
        if self.hist_t and t <= self.hist_t[-1]:
            i = next((j for j, tj in enumerate(self.hist_t) if tj >= t), None)
            if i is not None:
                self.hist_t, self.hist_x = self.hist_t[:i], self.hist_x[:i]
                self.hist_v = self.hist_v[:i]
        self.hist_t.append(t); self.hist_x.append(x)

        if mode == 'open_loop':
            v = 0.0
        elif mode == 'static':
            v = float(np.clip(np.log(1.22), 0.0, self.v_max))
        else:
            if self.delay > 0 and len(self.hist_t) > 6:
                xe = self.predict_taylor(t) if mode.endswith('taylor') else self.predict_smith(t)
            else:
                xe = x
            if mode == 'cbf_min' or mode == 'cbf_min_taylor':
                # minimum-intervention safety filter: nominal policy is NO tax,
                # the barrier is the only thing that ever raises tau.
                v = float(np.clip(self.v_cbf(xe, t), 0.0, self.v_max))
            elif mode.startswith('cbf'):
                v = float(np.clip(max(self.v_clf(xe, t), self.v_cbf(xe, t)), 0.0, self.v_max))
            else:
                v = self.v_clf(xe, t)
        self.hist_v.append(v)
        return [self.F(x, t) + self.G*v]


def run_v2(mode='open_loop', eta=ETA_NOM, delay=0.0, shock='s_curve', K=K_GAIN,
           v_max=V_MAX, t_max=T_HORIZON, n=1000, lam=LAMBDA, gamma=GAMMA, seed=42,
           M=M_EFF):
    sim = JevonsV2(eta=eta, lam=lam, gamma=gamma, K=K, delay=delay,
                   v_max=v_max, shock=shock, seed=seed, M=M)
    te = np.linspace(0, t_max, n)
    sol = solve_ivp(lambda t, y: sim.rhs(t, y, mode), (0, t_max), [sim.x0],
                    t_eval=te, method='RK45', rtol=1e-7, atol=1e-9)
    x = sol.y[0]
    nn = min(len(sim.hist_t), len(sim.hist_v))
    v = np.interp(sol.t, sim.hist_t[:nn], sim.hist_v[:nn]) if nn > 1 else np.zeros_like(sol.t)
    mu = np.array([sim.mu(tt)[0] for tt in sol.t])
    return dict(t=sol.t, x=x, N=np.exp(x), v=v, tau=np.exp(v)-1.0,
                mu=mu, e=np.exp(mu), Y=np.exp(x+mu), sim=sim)


def metrics_v2(r, x_star=np.log(N_STAR)):
    t, x, N, tau = r['t'], r['x'], r['N'], r['tau']
    err = N - np.exp(x_star)
    return dict(
        peak_N=float(N.max()),
        Mp=float(max(0.0, err.max())),
        IAE=float(_trapz(np.abs(err), t)),
        ISCE=float(_trapz(tau**2, t)),
        Yf=float(r['Y'][-1]),
        breach=bool(N.max() > 1.05*np.exp(x_star)),
        sat=float(_trapz((r['v'] >= 0.999*r['sim'].v_max).astype(float), t)/(t[-1]-t[0])),
    )
