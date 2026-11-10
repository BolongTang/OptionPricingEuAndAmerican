# Extensions that build on the hand-written code (fast trees, Greeks, PDE solver, LSM, BAW, hedging experiments, ...)

def rng_for(k):                                     # independent reproducible numpy stream per experiment
    return np.random.default_rng([SEED, k])

def macro(name, text):
    # \newcommand{\name}{text}; the name may contain letters only
    assert name.isalpha(), name
    MACROS[name] = str(text)

def save_fig(fig, name, show=True):
    fig.savefig(f"figures/{name}.pdf")
    if show: plt.show()
    else: plt.close(fig)

def write_table(name, header, rows, align):
    # writes only the tabular environment (booktabs); caption and label live in main.tex
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(f"tables/{name}.tex").write_text("\n".join(lines) + "\n")

def timeit(f, *a, **k):
    # one timed call; calls that finish in under 50 ms are repeated five more times and the fastest run is reported
    # (the first call of a function pays import and cache overheads that say nothing about the method)
    t0 = time.perf_counter(); out = f(*a, **k); secs = time.perf_counter() - t0
    if secs < 0.05:
        for _ in range(5):
            t0 = time.perf_counter(); f(*a, **k); secs = min(secs, time.perf_counter() - t0)
    return out, secs

def brownian_grid_paths(N, n_paths, T=1.0, rng=None):
    dW = math.sqrt(T / N) * rng.standard_normal((n_paths, N))
    return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)

def scaled_walk_grid_paths(N, n_paths, T=1.0, rng=None):
    xi = rng.choice([-1.0, 1.0], size=(n_paths, N))
    return np.concatenate([np.zeros((n_paths, 1)), math.sqrt(T / N) * np.cumsum(xi, axis=1)], axis=1)

def ks_distance_to_normal(N):
    # sup_x |P(W^(N)_1 <= x) - Phi(x)| : the CDF is a step function, so check both sides of every jump
    k = np.arange(N + 1)
    x = (2 * k - N) / np.sqrt(N)
    right = binom.cdf(k, N, 0.5)
    left = np.concatenate([[0.0], right[:-1]])
    return max(np.abs(right - norm.cdf(x)).max(), np.abs(left - norm.cdf(x)).max())

def walk_max_survival(N, a):
    # P(max_{n<=N} W^(N)_{t_n} >= a) for the scaled walk, exact via reflection on the lattice
    k = np.ceil(a * math.sqrt(N) - 1e-12).astype(int)           # smallest integer level >= a sqrt(N)
    SN = lambda j: binom.sf(np.ceil((N + j) / 2) - 1, N, 0.5)   # P(S_N >= j) with S_N = 2B - N
    return SN(k) + binom.sf(np.floor((N + k) / 2), N, 0.5)

def bs_d1d2(S, K, r, sigma, T, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bs_price(S, K, r, sigma, T, kind="call", q=0.0):
    d1, d2 = bs_d1d2(S, K, r, sigma, T, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_greeks(S, K, r, sigma, T, kind="call", q=0.0):
    d1, d2 = bs_d1d2(S, K, r, sigma, T, q)
    pdf, eq, er = norm.pdf(d1), np.exp(-q * T), np.exp(-r * T)
    gamma = eq * pdf / (S * sigma * np.sqrt(T)); vega = S * eq * pdf * np.sqrt(T)
    if kind == "call":
        delta = eq * norm.cdf(d1); rho = K * T * er * norm.cdf(d2)
        theta = -S * eq * pdf * sigma / (2 * np.sqrt(T)) - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
    else:
        delta = eq * (norm.cdf(d1) - 1); rho = -K * T * er * norm.cdf(-d2)
        theta = -S * eq * pdf * sigma / (2 * np.sqrt(T)) + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def implied_vol(price_obs, S, K, r, T, kind="call", q=0.0, lo=1e-4, hi=5.0):
    # Brent root of bs_price(sigma) - price (bs_price is strictly increasing in sigma, so the root is unique)
    f = lambda s: bs_price(S, K, r, s, T, kind, q) - price_obs
    if f(lo) > 0 or f(hi) < 0: return np.nan
    return brentq(f, lo, hi, xtol=1e-12)

def _h_pp(z, n):
    # Peizer-Pratt inversion method 2
    return 0.5 + np.sign(z) * np.sqrt(0.25 - 0.25 * np.exp(-(z / (n + 1/3 + 0.1/(n + 1)))**2 * (n + 1/6)))

def tree_params(S0, K, r, sigma, T, N, method="crr", q=0.0):
    dt = T / N; a = math.exp((r - q) * dt)
    if method == "crr":
        u = math.exp(sigma * math.sqrt(dt)); d = 1 / u; p = (a - d) / (u - d)
    elif method == "jr":
        u = math.exp((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt)); d = math.exp((r - q - 0.5 * sigma**2) * dt - sigma * math.sqrt(dt)); p = 0.5
    elif method == "tian":
        nu = math.exp(sigma**2 * dt); w = math.sqrt(nu**2 + 2 * nu - 3)
        u = a * nu / 2 * (nu + 1 + w); d = a * nu / 2 * (nu + 1 - w); p = (a - d) / (u - d)
    elif method == "lr":
        assert N % 2 == 1, "Leisen-Reimer needs an odd number of steps"
        d1, d2 = bs_d1d2(S0, K, r, sigma, T, q)
        p = float(_h_pp(d2, N)); ps = float(_h_pp(d1, N)); u = a * ps / p; d = (a - p * u) / (1 - p)
    else:
        raise ValueError(method)
    return u, d, p

def tree_price(S0, K, r, sigma, T, N, kind="call", american=False, method="crr", q=0.0, smooth=False, nodes=False,
               root_flag=False, ex_every=None):
    # backward induction on the recombining lattice S_{j,i} = S0 u^i d^(j-i); O(N) memory.
    # american: exercise allowed at every layer (or only when j is a multiple of ex_every: Bermudan);
    # root_flag: also return whether exercise is strictly optimal at the root (used to find b(0;T)).
    u, d, p = tree_params(S0, K, r, sigma, T, N, method, q); dt = T / N; disc = math.exp(-r * dt)
    pay = (lambda s: np.maximum(s - K, 0.0)) if kind == "call" else (lambda s: np.maximum(K - s, 0.0))
    node = lambda j: S0 * u ** np.arange(j + 1) * d ** (j - np.arange(j + 1))
    if smooth and not american:                        # Broadie-Detemple: last layer = BS value over one step
        Sj = node(N - 1); V = bs_price(Sj, K, r, sigma, dt, kind, q); start = N - 2
    else:
        V = pay(node(N)); start = N - 1
    keep, ex0 = {}, False
    for j in range(start, -1, -1):
        cont = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american and (ex_every is None or j % ex_every == 0):
            pj = pay(node(j))
            if j == 0: ex0 = bool(pj[0] > cont[0] + 1e-12)
            V = np.maximum(pj, cont)
        else:
            V = cont
        if nodes and j <= 2: keep[j] = (node(j).copy(), V.copy())
    if nodes: return float(V[0]), keep
    if root_flag: return float(V[0]), ex0
    return float(V[0])

def gbm_euler_terminal(S0, mu, sigma, T, N, M, rng, Z=None):
    # Z (M x N) may be supplied so that the noise is identical to a run of gbm_euler_maruyama
    dt = T / N; S = np.full(M, S0, dtype=float)
    for n in range(N):
        S += mu * S * dt + sigma * S * math.sqrt(dt) * (rng.standard_normal(M) if Z is None else Z[:, n])
    return S

def terminal_exact(Z, S0=S0, r=r, sigma=sigma, T=T):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

def mc_estimators(M, rng, K=K):
    # returns {method: (estimate, standard error)} for the discounted call payoff, sampling cost M payoffs (antithetic: M/2 pairs)
    e = math.exp(-r * T); out = {}
    Z = rng.standard_normal(M); Y = e * np.maximum(terminal_exact(Z) - K, 0)
    out["plain"] = (Y.mean(), Y.std(ddof=1) / math.sqrt(M))
    Zh = Z[: M // 2]; Ya = 0.5 * (e * np.maximum(terminal_exact(Zh) - K, 0) + e * np.maximum(terminal_exact(-Zh) - K, 0))
    out["antithetic"] = (Ya.mean(), Ya.std(ddof=1) / math.sqrt(M // 2))
    X = e * terminal_exact(Z); b = np.cov(Y, X)[0, 1] / X.var(ddof=1)            # control variate: discounted S_T, E = S0
    Yc = Y - b * (X - S0); out["control variate"] = (Yc.mean(), Yc.std(ddof=1) / math.sqrt(M))
    return out

def euler_and_exact_terminal(S0, mu, sigma, T, N, M, rng):
    dt = T / N; Sx = np.full(M, float(S0)); sumZ = np.zeros(M)
    for _ in range(N):
        Z = rng.standard_normal(M); sumZ += Z; Sx *= 1.0 + mu * dt + sigma * math.sqrt(dt) * Z
    return Sx, S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * math.sqrt(dt) * sumZ)

def tree_greeks(S0, K, r, sigma, T, N, kind="call", american=False, q=0.0, method="crr"):
    V0, nd = tree_price(S0, K, r, sigma, T, N, kind, american, method, q, nodes=True)
    (S1, V1), (S2, V2) = nd[1], nd[2]; dt = T / N
    d_up = (V2[2] - V2[1]) / (S2[2] - S2[1]); d_dn = (V2[1] - V2[0]) / (S2[1] - S2[0])
    delta = (V1[1] - V1[0]) / (S1[1] - S1[0]); gamma = (d_up - d_dn) / (0.5 * (S2[2] - S2[0]))
    theta = (V2[1] - V0) / (2 * dt)
    hv, hr = 1e-4, 1e-5
    vega = (tree_price(S0, K, r, sigma + hv, T, N, kind, american, method, q) - tree_price(S0, K, r, sigma - hv, T, N, kind, american, method, q)) / (2 * hv)
    rho = (tree_price(S0, K, r + hr, sigma, T, N, kind, american, method, q) - tree_price(S0, K, r - hr, sigma, T, N, kind, american, method, q)) / (2 * hr)
    return dict(price=V0, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def mc_greeks(S0, K, r, sigma, T, M, rng):
    Z = rng.standard_normal(M); ST = terminal_exact(Z, S0, r, sigma, T); e = math.exp(-r * T); ind = ST > K; out = {}
    se = lambda y: (y.mean(), y.std(ddof=1) / math.sqrt(M))
    out["delta pathwise"] = se(e * ind * ST / S0)
    out["delta LR"] = se(e * np.maximum(ST - K, 0) * Z / (S0 * sigma * math.sqrt(T)))
    h = 0.01 * S0; up = e * np.maximum(terminal_exact(Z, S0 + h, r, sigma, T) - K, 0); dn = e * np.maximum(terminal_exact(Z, S0 - h, r, sigma, T) - K, 0)
    out["delta bump (CRN)"] = se((up - dn) / (2 * h))
    out["gamma LR x pathwise"] = se(e * ind * ST / S0**2 * (Z / (sigma * math.sqrt(T)) - 1))
    out["gamma bump (CRN)"] = se((up - 2 * e * np.maximum(ST - K, 0) + dn) / h**2)
    out["vega pathwise"] = se(e * ind * ST * (math.sqrt(T) * Z - sigma * T))
    out["vega LR"] = se(e * np.maximum(ST - K, 0) * ((Z**2 - 1) / sigma - math.sqrt(T) * Z))
    return out

def hedge_experiment(S0, K, r, sig_real, sig_hedge, T, n, M, rng, drift=None):
    drift = r if drift is None else drift; dt = T / n
    Z = rng.standard_normal((M, n)); logS = np.cumsum((drift - 0.5 * sig_real**2) * dt + sig_real * math.sqrt(dt) * Z, axis=1)
    Sp = S0 * np.concatenate([np.ones((M, 1)), np.exp(logS)], axis=1)
    tk = np.arange(n + 1) * dt; D = np.exp(-r * tk)
    tau = T - tk[:-1]; d1 = (np.log(Sp[:, :-1] / K) + (r + 0.5 * sig_hedge**2) * tau) / (sig_hedge * np.sqrt(tau))
    delta = norm.cdf(d1); gains = (delta * (D[1:] * Sp[:, 1:] - D[:-1] * Sp[:, :-1])).sum(axis=1)
    C0 = float(bs_price(S0, K, r, sig_hedge, T)); eps = C0 + gains - D[-1] * np.maximum(Sp[:, -1] - K, 0)
    gamma_term = (D[:-1] * norm.pdf(d1) / (sig_hedge * np.sqrt(tau)) * Sp[:, :-1]).sum(axis=1) * dt   # integral of D * Gamma * S^2 dt, for the theory line
    return eps, 0.5 * (sig_hedge**2 - sig_real**2) * gamma_term

def pde_put(S0, K, r, sigma, T, Ny=801, Nt=200, american=True, q=0.0, rho=1e7, half_width=None):
    L = (5 * sigma * math.sqrt(T) + abs(r - q) * T + 0.5) if half_width is None else half_width
    y = math.log(K) + np.linspace(-L, L, Ny); Sg = np.exp(y); dy = y[1] - y[0]; dt = T / Nt
    mu_ = r - q - 0.5 * sigma**2
    a = 0.5 * sigma**2 / dy**2 - mu_ / (2 * dy); b = -sigma**2 / dy**2 - r; c = 0.5 * sigma**2 / dy**2 + mu_ / (2 * dy)
    n = Ny - 2; A = diags([a * np.ones(n - 1), b * np.ones(n), c * np.ones(n - 1)], [-1, 0, 1], format="csc"); I = diags([np.ones(n)], [0], format="csc")
    g = np.maximum(K - Sg, 0.0); gi = g[1:-1]
    left = lambda tau: (K - Sg[0]) if american else (K * math.exp(-r * tau) - Sg[0] * math.exp(-q * tau)); right = lambda tau: 0.0
    V = g[1:-1].copy(); tau = 0.0; bnd = [np.nan]; tgrid = [0.0]
    sched = [(dt / 2, 1.0)] * 4 + [(dt, 0.5)] * (Nt - 2)         # Rannacher start, then Crank-Nicolson
    for h, th in sched:
        bc_old = np.zeros(n); bc_new = np.zeros(n); bc_old[0] = a * left(tau); bc_old[-1] = c * right(tau)
        bc_new[0] = a * left(tau + h); bc_new[-1] = c * right(tau + h)
        rhs = V + (1 - th) * h * (A @ V + bc_old) + th * h * bc_new; M0 = (I - th * h * A).tocsc()
        if american:
            act = V <= gi
            for _ in range(40):
                Vn = splu((M0 + diags([rho * act], [0], format="csc")).tocsc()).solve(rhs + rho * act * gi)
                new = Vn < gi
                if (new == act).all(): break
                act = new
            V = np.maximum(Vn, gi)
        else:
            V = splu(M0).solve(rhs)
        tau += h
        if abs(tau / dt - round(tau / dt)) < 1e-9 and round(tau / dt) >= 1:
            ex = ((V - gi) < 1e-6) & (gi > 0) if american else np.zeros(n, bool)
            bnd.append(Sg[1:-1][ex].max() if ex.any() else np.nan); tgrid.append(tau)
    Vfull = np.concatenate([[left(T)], V, [0.0]])
    value = float(np.interp(math.log(S0), y, Vfull)) if False else float(__import__("scipy.interpolate", fromlist=["CubicSpline"]).CubicSpline(y, Vfull)(math.log(S0)))
    return dict(value=value, S=Sg, V=Vfull, tau=np.array(tgrid), boundary=np.array(bnd))

def lsm_put(S0, K, r, sigma, T, n_dates, M, rng, deg=3, q=0.0):
    dt = T / n_dates; disc = math.exp(-r * dt)
    Z = rng.standard_normal((M, n_dates))
    S = S0 * np.exp(np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z, axis=1))
    S = np.concatenate([np.full((M, 1), float(S0)), S], axis=1)
    cash = np.maximum(K - S[:, -1], 0.0)
    for j in range(n_dates - 1, 0, -1):
        cash *= disc; pay = np.maximum(K - S[:, j], 0.0); itm = pay > 0
        if itm.sum() > deg + 1:
            coef = np.polyfit(S[itm, j] / K, cash[itm], deg); cont = np.polyval(coef, S[itm, j] / K)
            ex = pay[itm] > cont; idx = np.where(itm)[0][ex]; cash[idx] = pay[idx]
    disc_cash = disc * cash; est = max(K - S0, disc_cash.mean())
    return est, disc_cash.std(ddof=1) / math.sqrt(M)

def baw_put(S, K, r, sigma, T, q=0.0):
    b = r - q; Mm = 2 * r / sigma**2; Nn = 2 * b / sigma**2; kap = 1 - math.exp(-r * T)
    q1 = 0.5 * (-(Nn - 1) - math.sqrt((Nn - 1)**2 + 4 * Mm / kap))
    def parts(Ss):
        d1 = (math.log(Ss / K) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return d1, 1 - math.exp((b - r) * T) * norm.cdf(-d1)
    def g(Ss):
        d1, f = parts(Ss); return K - Ss - float(bs_price(Ss, K, r, sigma, T, "put", q)) + f * Ss / q1
    Sc = brentq(g, 1e-6 * K, K)
    d1c, fc = parts(Sc); A1 = -(Sc / q1) * fc
    S = np.asarray(S, float)
    val = np.where(S > Sc, bs_price(S, K, r, sigma, T, "put", q) + A1 * (S / Sc) ** q1, K - S)
    return val, Sc

def eep(S_, K_, r_, sig_, T_, N=600):
    # American minus European on the SAME lattice, so the lattice error cancels and the premium is exactly 0 when there is no early exercise
    return tree_price(S_, K_, r_, sig_, T_, N, "put", american=True) - tree_price(S_, K_, r_, sig_, T_, N, "put", american=False)

class CRRTreeGen(CRRTree):
    # [EXT] American/European call or put with a continuous dividend yield, built on top of CRRTree
    def __init__(self, S0=100, K=100, mu=0.05, vol=0.3, r=0.05, T=1, steps=100, kind="call", q=0.0, american=True, seed=SEED):
        super().__init__(S0=S0, K=K, mu=mu, vol=vol, r=r, T=T, steps=steps, seed=seed)
        self.kind, self.div, self.american = kind, q, american
        self.risk_neutral_p = (math.exp((r - q) * self.stepsize) - self.d) / (self.u - self.d)
        self.risk_neutral_q = 1 - self.risk_neutral_p; self.p, self.q = self.risk_neutral_p, self.risk_neutral_q

    def build_option_tree(self):
        M, disc = self.steps, math.exp(-self.rate * self.stepsize); sgn = 1.0 if self.kind == "call" else -1.0
        Sj = [np.array(layer) for layer in self.stock_tree]
        V = [None] * (M + 1); ex = [None] * (M + 1)
        V[M] = np.maximum(sgn * (Sj[M] - self.strike_price), 0); ex[M] = V[M] > 0
        for j in range(M - 1, -1, -1):
            a = np.maximum(sgn * (Sj[j] - self.strike_price), 0)
            b = disc * (self.risk_neutral_p * V[j + 1][1:] + self.risk_neutral_q * V[j + 1][:-1])
            ex[j] = (a > b + 1e-12) if self.american else np.zeros(len(a), bool)
            V[j] = np.maximum(a, b) if self.american else b
        self.option_tree, self.exercise = V, ex
        self.exercise_grid = np.zeros((M + 1, M + 1), bool)
        for j in range(M + 1): self.exercise_grid[j, :j + 1] = ex[j]
        pick = np.min if self.kind == "call" else np.max
        self.decision_bound = np.array([pick(Sj[j][ex[j]]) if ex[j].any() else np.nan for j in range(M + 1)])
        self.hedgings = hedgingformula(self.option_tree, Sj)
        return self

def pnl_paths(model, n, p_up, rng, V0=None, r_=None):
    Mm = model.steps; r_ = model.r if r_ is None else r_; V0 = model.option_tree[0][0] if V0 is None else V0
    ups = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(rng.random((n, Mm)) < p_up, axis=1, dtype=np.int32)], axis=1)
    flags = model.exercise_grid[np.arange(Mm + 1)[None, :], ups]
    any_ = flags.any(axis=1); tau = np.where(any_, flags.argmax(axis=1), Mm)
    Stau = model.S0 * model.u ** (2 * ups[np.arange(n), tau] - tau)
    return np.exp(-r_ * tau * model.dt) * np.maximum(model.K - Stau, 0) - V0, tau * model.dt, tau < Mm

def risk_metrics(x, V0):
    # the buyer's loss is capped at the premium, so the left tail is uninformative (more than half the paths lose all of it);
    # the tail that matters is the SELLER's: seller loss = buyer P&L. Report P(total loss), sd and the right tail of the buyer's P&L.
    q95, q99 = np.quantile(x, 0.95), np.quantile(x, 0.99)
    return dict(mean=x.mean(), sd=x.std(), ptot=(x < -V0 + 1e-9).mean(), q95=q95, q99=q99, es95=x[x >= q95].mean())

def b0_bisect(T_, M_, lo=3.0, hi=10.0, tol=1e-4):
    # largest spot at which exercise is optimal at date 0 (exercise region = {x <= b}); the root decision is monotone in x
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if tree_price(mid, 10, 0.02, 0.2, T_, M_, "put", american=True, root_flag=True)[1]: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def add_row(name, fn, key=None):
    (val, secs) = timeit(fn); val, se = (val if isinstance(val, tuple) else (val, None))
    rows.append([name, f"{val:.4f}", f"{val-BS:+.4f}", "--" if se is None else f"{se:.4f}", f"{1000*secs:.1f}"])
    if key: dc[key] = (val, se, secs)
    return val, secs

def first_hit(fn, Ns_list):
    for N in Ns_list:
        v, secs = timeit(fn, N)
        if abs(v - BS) <= tol: return N, secs
    return None, float("nan")

def add_am(name, fn, key=None):
    (val, secs) = timeit(fn); val, se = (val if isinstance(val, tuple) else (val, None))
    rows.append([name, f"{val:.5f}", f"{val-V_ref:+.5f}", "--" if se is None else f"{se:.4f}", f"{1000*secs:.0f}"])
    if key: da[key] = (val, se, secs)
    return val

def fix_sign(v):
    v = str(v)
    m_ = re.fullmatch(r"([-+]?)(\d+(?:\.\d+)?)e([-+])0*(\d+)", v)          # 8.4e-02 -> 8.4\times10^{-2}
    if m_:
        sgn, mant, es, ex = m_.groups(); return (("\\ensuremath{-}" if sgn == "-" else "") + r"\ensuremath{" + mant + r"\times10^{" + ("-" if es == "-" else "") + ex + "}}")
    return ("\\ensuremath{-}" + v[1:]) if v.startswith("-") else (("\\ensuremath{+}" + v[1:]) if v.startswith("+") else v)
