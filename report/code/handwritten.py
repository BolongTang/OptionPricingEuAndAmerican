# Hand-written classes and functions, exactly as used in the notebook
import math, random
import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion():
    def __init__(self, N1 = 10_000, time_horizon = [0,1]):
        self.position = 0
        self.path = [0, ]
        self.N1 = N1
        self.time_horizon = time_horizon

    # Run BrownianMotion for one step
    # (independent from previous moves)
    def run(self, step, dim):
        self.position += random.gauss(mu = 0, sigma = step) # only support 1D for now
        self.path.append(self.position)

    def simulate(self):
        steps = self.N1
        interval = self.time_horizon

        dt = (interval[1] - interval[0]) / steps
        for _ in range(steps):
            self.run(step = math.sqrt(dt), dim = 1)
        return self.path

class symmetric_random_walk(BrownianMotion):
    def __init__(self, N2 = 10, time_horizon = [0,1]):
        super().__init__(N1 = N2, time_horizon = time_horizon)
        self.N2 = N2

    def run(self, step, dim):
        self.position += random.choice([step, -step]) # proba 1/2
        self.path.append(self.position)

    def simulate(self):
        steps = self.N2
        interval = self.time_horizon
        dt = (interval[1] - interval[0]) / steps
        for _ in range(steps):
            self.run(step = math.sqrt(dt), dim = 2)
        return self.path

class CRRBinomialModel():
    def __init__(self, initial_stock_value = 1, rate = risk_free_CAD_rate, volatility = annualized_252_volatility, steps = 100, days = 30, strike_price = 3):
        self.strike_price = strike_price
        self.initial_stock_value = initial_stock_value
        self.days = days
        self.rate = rate
        self.volatility = volatility
        self.steps = steps
        self.T = days / 252
        self.stepsize = self.T / steps
        self.u = math.exp(volatility * math.sqrt(self.stepsize)) # u = e^{sigma sqrt(dt)}
        self.d = 1 / self.u
        self.risk_neutral_p = (math.exp(rate * self.stepsize) - self.d) / (self.u - self.d) # p = (e^{r dt} - d)/(u - d)
        self.risk_neutral_q = 1 - self.risk_neutral_p
        self.stock_tree = []
        self.option_tree = []

    def clear_tree(self):
        self.stock_tree = []
        self.option_tree = []

    def generate_stock_tree(self):
        self.clear_tree()

        # O(N) with 2 base cases: (assume at least 1 period) every round compute two numbers and copy an existing list
        self.stock_tree.append([self.initial_stock_value]) # period0
        self.stock_tree.append([self.initial_stock_value * self.d, self.initial_stock_value * self.u]) # period1

        for day in range(self.steps - 1):
            self.stock_tree.append([self.stock_tree[-2][0] * (self.d ** 2)] + self.stock_tree[-2] + [self.stock_tree[-2][-1] * (self.u ** 2)])

        # Stock tree done, then generate option tree backward induction
        return self.stock_tree

    def generate_option_tree(self):
        import numpy as np
        import copy
        self.option_tree = copy.deepcopy(self.stock_tree)

        last_day_values = []
        for elem in self.option_tree[-1]:
            elem -= self.strike_price
            elem = max(elem, 0)
            last_day_values.append(elem)
        self.option_tree[-1] = last_day_values

        # across days
        for i in range(len(self.option_tree)-2, -1, -1):
            # across values in a day
            for j in range(len(self.option_tree[i])-1, -1, -1):

                deltai = (self.option_tree[i+1][j] - self.option_tree[i+1][j+1]) / (self.stock_tree[i+1][j] - self.stock_tree[i+1][j+1])

                xi = vi = math.exp(-self.rate * self.stepsize) * (self.risk_neutral_p * self.option_tree[i+1][j+1] + self.risk_neutral_q * self.option_tree[i+1][j])

                self.option_tree[i][j] = vi

        return self.option_tree

    # Need to have had a tree
    def pricing(self):
        return self.option_tree[0][0]

    def simulate_stock(self):
        value = self.initial_stock_value
        path = [value]

        j = 0
        for i in range(self.steps):
            j += 1 if random.random() < self.risk_neutral_p else 0
            value = self.stock_tree[i + 1][j]
            path.append(value)

        days_per_step = self.stepsize * 252
        x = [i * days_per_step for i in range(len(path))]

        plt.plot(x, path)
        plt.xlabel("Days")
        plt.ylabel("Stock Value")
        plt.title(f"Stock Simulation N={self.steps}")
        return path

    def simulate_option(self):
        value = self.option_tree[0][0]
        path = [value]

        j = 0
        for i in range(self.steps):
            j += 1 if random.random() < self.risk_neutral_p else 0
            value = self.option_tree[i + 1][j]
            path.append(value)

        days_per_step = self.stepsize * 252
        x = [i * days_per_step for i in range(len(path))]

        plt.plot(x, path)
        plt.xlabel("Days")
        plt.ylabel("Option Value")
        plt.title(f"Option Simulation N={self.steps}")
        return path

    def plot_stock_tree(self, fname = "binomial_stock_tree.png"):
        for i, level in enumerate(self.stock_tree):
            plt.scatter([i] * len(level), level, color = "C0", s = 12)
        plt.title(f"CRRBinomialStockTreeN{self.steps}")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.savefig(fname, bbox_inches = "tight")
        plt.show()

    def plot_option_tree(self, fname = "binomial_stock_tree.png"):
        for i, level in enumerate(self.option_tree):
            plt.scatter([i] * len(level), level, color = "C0", s = 12)
        plt.title(f"CRRBinomialOptionTreeN{self.steps}")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.savefig(fname, bbox_inches = "tight")
        plt.show()

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def gbm_euler_maruyama(S0, mu, sigma, T, N, n_paths=1, seed=None):
    # Euler-Maruyama simulation of GBM: S_{n+1} = S_n + mu*S_n*dt + sigma*S_n*dW_n
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    Z = rng.standard_normal((n_paths, N))
    dW = np.sqrt(dt) * Z

    S = np.empty((n_paths, N + 1))
    S[:, 0] = S0

    for n in range(N):
        S[:, n + 1] = S[:, n] + mu * S[:, n] * dt + sigma * S[:, n] * dW[:, n]

    return t, S

def gbm_exact(S0, mu, sigma, T, N, n_paths=1, seed=None):
    # Exact (log-Euler) simulation of GBM for comparison: S_{n+1} = S_n * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_n)
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    Z = rng.standard_normal((n_paths, N))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.log(S0) + np.cumsum(increments, axis=1)

    S = np.empty((n_paths, N + 1))
    S[:, 0] = S0
    S[:, 1:] = np.exp(log_S)

    return t, S

def plotsmart(x=None, ys=(), labels=None, xlabel="", ylabel="", title="", fname=None,
              xscale="linear", single_points=None, kind="line"):
    # To save space, this is an all-in-one helper for every figure, which is written after all the plotting commands are written (collecting all needs), and always saves to figures/<fname>.png.
    labels = labels or [None] * len(ys)
    if kind == "hist":                      # one histogram panel per dataset
        fig, axs = plt.subplots(1, len(ys), figsize=(5 * len(ys), 4))
        for ax, y, lab in zip(axs, ys, labels):
            y = np.asarray(y)
            if np.unique(y).size <= 2:      # 0/1 indicator (early exercise)
                ax.hist(y, bins=[-0.5, 0.5, 1.5], rwidth=0.6); ax.set_xticks([0, 1])
            else:
                ax.hist(y, bins=40)
            ax.set_xlabel(lab); ax.set_ylabel("count")
        fig.suptitle(title)
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        per_curve_x = isinstance(x, list) and len(x) > 0 and isinstance(x[0], np.ndarray)
        for k, (y, lab) in enumerate(zip(ys, labels)):
            ax.plot(x[k] if per_curve_x else x, y, label=lab)
        for (px, py, lab) in (single_points or []):
            ax.plot([px], [py], "k*", ms=12, label=lab)
        ax.set_xscale(xscale); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        if any(labels) or single_points: ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if fname: fig.savefig(f"figures/{fname}.png", dpi=200)
    plt.show()

def perp_params(r, vol, K=K):
    kappa = r / vol**2
    Delta = math.sqrt((kappa - 1/2)**2 + 2*kappa)
    beta = (kappa - 1/2) + Delta          # = 2 r / vol^2
    return beta, K * beta / (beta + 1)

def Vinf(x, r=r, vol=vol, K=K):
    beta, Sstar = perp_params(r, vol, K)
    x = np.asarray(x, float)
    return np.where(x <= Sstar, K - x, K / (beta + 1) * (Sstar / x) ** beta)

class CRRTree(CRRBinomialModel):
    def __init__(self, S0=10, K=10, mu=0.05, vol=0.2, r=0.02, T=1, steps=100, seed=SEED):
        # the base class counts maturity in trading days (T = days/252), so days = round(252 T)
        super().__init__(initial_stock_value=S0, rate=r, volatility=vol, steps=steps,
                         days=round(252 * T), strike_price=K)
        self.S0, self.K, self.mu, self.vol, self.r = S0, K, mu, vol, r
        self.dt, self.p, self.q = self.stepsize, self.risk_neutral_p, self.risk_neutral_q
        self.rng = np.random.default_rng(seed)
        self.tree = []
        self.build_stock_tree()
        self.decision_bound = 0
        self.hedgings = []

    def build_stock_tree(self):
        return self.generate_stock_tree()

    def S(self, j, i):
        return self.stock_tree[j][i]

    def build_option_tree(self):
        # same backward induction as generate_option_tree, with the early-exercise max (American put)
        M, disc = self.steps, math.exp(-self.rate * self.stepsize)
        Sj = [np.array(layer) for layer in self.stock_tree]
        V = [None] * (M + 1); ex = [None] * (M + 1)
        V[M] = np.maximum(self.strike_price - Sj[M], 0); ex[M] = V[M] > 0
        for j in range(M - 1, -1, -1):
            a = np.maximum(self.strike_price - Sj[j], 0)
            changes = self.risk_neutral_p * V[j + 1][1:] + self.risk_neutral_q * V[j + 1][:-1]
            b = disc * changes
            ex[j] = a > b + 1e-12               # strictly better to exercise (ties continue)
            V[j] = np.maximum(a, b)
        self.option_tree, self.exercise = V, ex
        self.exercise_grid = np.zeros((M + 1, M + 1), bool)
        for j in range(M + 1): self.exercise_grid[j, :j + 1] = ex[j]
        # exercise boundary b(t_j): largest stock node in the exercise region (nan if none)
        self.decision_bound = np.array([Sj[j][ex[j]].max() if ex[j].any() else np.nan for j in range(M + 1)])
        self.hedgings = hedgingformula(self.option_tree, Sj)
        return self

    def simulate(self):
        # returns the prices and the node index j (= number of up moves) at each step
        ups = np.concatenate(([0], np.cumsum(self.rng.random(self.steps) < self.risk_neutral_p)))
        return self.S0 * self.u ** (2 * ups - np.arange(self.steps + 1)), ups     # = stock_tree[i][ups[i]]

    def stopping_index(self, ups):
        # optimal stopping: first step where the path is in the exercise region (M = held to maturity)
        flags = self.exercise_grid[np.arange(self.steps + 1), ups]
        return int(np.argmax(flags)) if flags.any() else self.steps

def hedgingformula(option_tree, stock_tree):
    # delta at node (j,i) = (V_up - V_down) / (S_up - S_down)   (same ratio as deltai in the European model)
    hedgings = []
    for j in range(len(option_tree) - 1):
        hedgings.append((option_tree[j + 1][1:] - option_tree[j + 1][:-1]) /
                        (stock_tree[j + 1][1:] - stock_tree[j + 1][:-1]))
    return hedgings

def CRRDiscretize_run(steps):
    model = CRRTree(steps=steps)
    model.build_option_tree()
    V0s.append(model.option_tree[0][0]); bounds[steps] = model
    return model

def try_paths(model, prompt=("early exercise occurs", "the option is not exercised until maturity."), steps=100):
    M = model.steps; p1 = p2 = None
    while p1 is None or p2 is None:
        prices, ups = model.simulate(); tau = model.stopping_index(ups)
        if p1 is None and tau < M and abs(tau * model.dt - 0.5) < 0.1: p1 = (prices, ups, tau)
        if p2 is None and tau == M: p2 = (prices, ups, tau)
    return p1, p2

def simulate_10000_paths(vol=0.2, r=0.02, n=10000):
    pnls, stopping_times, early_exercise_probas = [], [], []
    model = CRRTree(vol=vol, r=r, steps=Mbin_sim).build_option_tree()
    V0 = model.option_tree[0][0]                  # fair value under (vol, r)
    for i in range(n):
        simulated_stock_prices, ups = model.simulate()
        tau = model.stopping_index(ups)           # optimal stopping index
        Stau = simulated_stock_prices[tau]
        pnls.append(math.exp(-r * tau * model.dt) * max(K - Stau, 0) - V0)
        stopping_times.append(tau * model.dt)
        early_exercise_probas.append(int(tau < model.steps))
    return np.array(pnls), np.array(stopping_times), np.array(early_exercise_probas)

def simulate_with_priced_boundary(real_vol, n=10000):
    model = CRRTree(vol=real_vol, r=r, steps=Mbin_sim).build_option_tree()   # realised-vol world (same dt grid)
    out = {k: [] for k in ["pnl", "tau", "ex", "pnl_opt", "tau_opt"]}
    for i in range(n):
        prices, ups = model.simulate()
        hit = prices <= exercise_boundary_voldot2 * (1 + 1e-9)   # follow the priced (sigma=20%) boundary (tolerance: float noise in the tree nodes)
        tau = int(np.argmax(hit)) if hit.any() else model.steps
        tau_opt = model.stopping_index(ups)                # truly optimal rule under the realised vol (same path)
        out["pnl"].append(math.exp(-r * tau * model.dt) * max(K - prices[tau], 0) - fair_value_voldot2)
        out["pnl_opt"].append(math.exp(-r * tau_opt * model.dt) * max(K - prices[tau_opt], 0) - fair_value_voldot2)
        out["tau"].append(tau * model.dt); out["tau_opt"].append(tau_opt * model.dt)
        out["ex"].append(int(tau < model.steps))
    return {k: np.array(v) for k, v in out.items()}
