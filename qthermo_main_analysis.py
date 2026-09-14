from __future__ import annotations

import csv
import itertools
import math
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# ============================================================
# Output folders and run controls
# ============================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FIG_DIR = BASE_DIR / "figures"
SUPP_DIR = FIG_DIR / "supp"
CACHE_DIR = BASE_DIR / "plot_cache"
SPECTRAL_DIR = BASE_DIR / "spectral_diagnostics"

for folder in (FIG_DIR, SUPP_DIR, CACHE_DIR, SPECTRAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

FORCE_RECOMPUTE = False
RUN_VERY_HEAVY_SUPPLEMENT = True
MAKE_OPTIONAL_S13 = True
RUN_SPECTRAL_THEOREM_TESTS = True
RUN_EXACT_SPECTRAL_CURRENT_TESTS = True

# Exact state- and current-resolved spectral diagnostics. These complement the
# generic Short-Farrelly bound by resolving the actual macrocurrent amplitudes
# carried by each energy-gap channel.
SPECTRAL_CURRENT_MODES = ("mixing", "free", "engineered")
SPECTRAL_CURRENT_N16_DT = 0.05
SPECTRAL_CURRENT_SIZE_DT = 0.10
SPECTRAL_CURRENT_WINDOW = (8.0, 11.0)
SPECTRAL_CURRENT_TIME_CHUNK = 64

# Numerical controls for the spectral return-suppression tests.
# The N=16,Q=2 comparison uses a denser grid than the main figures because
# the plotted quantity is a time-window RMS rather than a pointwise value.
SPECTRAL_N16_DT = 0.05
SPECTRAL_SIZE_DT = 0.10
SPECTRAL_COMMON_WINDOW = (8.0, 11.0)
SPECTRAL_INTERIOR_P_TOL = 1e-12
SPECTRAL_EPS_GRID_SIZE = 160
SPECTRAL_SIZE_SYSTEMS = [(8, 2), (12, 2), (16, 2), (20, 2)]

# Microscopic parameters used throughout the manuscript.
J = 1.0
V1 = 1.0
V2 = 0.7

# NPG-style palette requested for publication plots.
NPG = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
]

RED = NPG[0]
BLUE = NPG[1]
GREEN = NPG[2]
NAVY = NPG[3]
SALMON = NPG[4]
PURPLE = NPG[5]
MINT = NPG[6]
DARK_RED = NPG[7]
BROWN = NPG[8]
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"
BLACK = "black"


# ============================================================
# Plot styling
# ============================================================

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
MAIN_FONT = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
Fontsize = 14
plt.rcParams.update(
    {
        "font.family": MAIN_FONT,
        "font.size": Fontsize,
        "axes.labelsize": Fontsize,
        "xtick.labelsize": Fontsize,
        "ytick.labelsize": Fontsize,
        "legend.fontsize": Fontsize,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.frameon": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.max_open_warning": 0,
    }
)


def finish_axes(ax):
    ax.minorticks_on()


def panel_label(ax, label, x=0.03, y=0.95):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color=BLACK,
    )


def savefig(fig, path: Path, h_pad=None, w_pad=None):
    fig.tight_layout(pad=0.35, h_pad=h_pad, w_pad=w_pad)
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=False,
    )
    print(f"Saved: {path}")
    plt.show()


# ============================================================
# Small utilities and caching
# ============================================================


def log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def cached(name: str, builder: Callable[[], dict]):
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists() and not FORCE_RECOMPUTE:
        print(f"Loading cache: {path}")
        with path.open("rb") as f:
            return pickle.load(f)

    print(f"Computing: {name}")
    data = builder()
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached: {path}")
    return data


# ============================================================
# Closed fixed-Q qubit chain
# ============================================================

_MODEL_CACHE = {}
_MACRO_CACHE = {}
_CROSS_H_CACHE = {}
_EDGE_CACHE = {}


def fixed_q_basis(N: int, Q: int):
    basis = []
    for occupied in itertools.combinations(range(N), Q):
        b = 0
        for j in occupied:
            b |= 1 << j
        basis.append(b)
    return basis, {b: i for i, b in enumerate(basis)}


def hopping_strength(N: int, bond0: int, mode: str) -> float:
    if mode in ("mixing", "free"):
        return -J
    if mode == "engineered":
        j = bond0 + 1
        return -J * math.sqrt(j * (N - j)) / (N / 2.0)
    raise ValueError(mode)


def get_model(N: int, Q: int, mode: str):
    key = (N, Q, mode)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    basis, index = fixed_q_basis(N, Q)
    rows, cols, vals = [], [], []

    for i, b in enumerate(basis):
        if mode == "mixing":
            diag = 0.0
            for j in range(N - 1):
                diag += V1 * ((b >> j) & 1) * ((b >> (j + 1)) & 1)
            for j in range(N - 2):
                diag += V2 * ((b >> j) & 1) * ((b >> (j + 2)) & 1)
            rows.append(i)
            cols.append(i)
            vals.append(diag)
        elif mode in ("free", "engineered"):
            rows.append(i)
            cols.append(i)
            vals.append(0.0)
        else:
            raise ValueError(mode)

        for j0 in range(N - 1):
            nj = (b >> j0) & 1
            nk = (b >> (j0 + 1)) & 1
            if nj == nk:
                continue

            b2 = b ^ (1 << j0) ^ (1 << (j0 + 1))
            rows.append(index[b2])
            cols.append(i)
            vals.append(hopping_strength(N, j0, mode))

    H = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(basis), len(basis)),
        dtype=np.complex128,
    )

    _MODEL_CACHE[key] = (H, basis, index)
    return H, basis, index


def get_macro_data(N: int, Q: int, K: int):
    key = (N, Q, K)
    if key in _MACRO_CACHE:
        return _MACRO_CACHE[key]

    if N % K != 0:
        raise ValueError("N must be divisible by K.")

    _, basis, _ = get_model(N, Q, "free")
    ell = N // K

    tuples = []

    def rec_build(prefix, remaining, cells_left):
        if cells_left == 1:
            if 0 <= remaining <= ell:
                tuples.append(tuple(prefix + [remaining]))
            return
        for n in range(min(ell, remaining) + 1):
            rec_build(prefix + [n], remaining - n, cells_left - 1)

    rec_build([], Q, K)
    tuples = sorted(tuples)
    tuple_to_id = {r: i for i, r in enumerate(tuples)}

    macro_id = np.empty(len(basis), dtype=np.int32)
    for i, b in enumerate(basis):
        counts = []
        for a in range(K):
            start = a * ell
            stop = (a + 1) * ell
            counts.append(sum((b >> j) & 1 for j in range(start, stop)))
        macro_id[i] = tuple_to_id[tuple(counts)]

    Omega = np.array(
        [math.prod(math.comb(ell, n) for n in r) for r in tuples],
        dtype=float,
    )

    data = {
        "K": K,
        "ell": ell,
        "tuples": tuples,
        "tuple_to_id": tuple_to_id,
        "macro_id": macro_id,
        "Omega": Omega,
    }
    _MACRO_CACHE[key] = data
    return data


def initial_indices(N: int, Q: int, K: int, init_mode: str, R0=None):
    _, basis, _ = get_model(N, Q, "free")
    macro = get_macro_data(N, Q, K)

    if init_mode == "left_half":
        left_mask = (1 << (N // 2)) - 1
        return np.array(
            [i for i, b in enumerate(basis) if (b & ~left_mask) == 0],
            dtype=int,
        )

    if init_mode == "fixed_record":
        if R0 is None:
            raise ValueError("R0 is required for init_mode='fixed_record'.")
        rid = macro["tuple_to_id"][tuple(R0)]
        return np.where(macro["macro_id"] == rid)[0]

    raise ValueError(init_mode)


def get_cross_hamiltonian(N: int, Q: int, K: int, mode: str):
    key = (N, Q, K, mode)
    if key in _CROSS_H_CACHE:
        return _CROSS_H_CACHE[key]

    _, basis, index = get_model(N, Q, mode)
    ell = N // K
    boundaries = {a * ell - 1 for a in range(1, K)}

    rows, cols, vals = [], [], []
    for i, b in enumerate(basis):
        for j0 in boundaries:
            nj = (b >> j0) & 1
            nk = (b >> (j0 + 1)) & 1
            if nj == nk:
                continue
            b2 = b ^ (1 << j0) ^ (1 << (j0 + 1))
            rows.append(index[b2])
            cols.append(i)
            vals.append(hopping_strength(N, j0, mode))

    Hcross = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(basis), len(basis)),
        dtype=np.complex128,
    )
    _CROSS_H_CACHE[key] = Hcross
    return Hcross


def get_macro_edges(N: int, Q: int, K: int, mode: str):
    key = (N, Q, K, mode)
    if key in _EDGE_CACHE:
        return _EDGE_CACHE[key]

    _, basis, index = get_model(N, Q, mode)
    macro = get_macro_data(N, Q, K)
    macro_id = macro["macro_id"]
    ell = N // K
    boundaries = {a * ell - 1 for a in range(1, K)}

    pair_map = {}

    for i, b in enumerate(basis):
        for j0 in boundaries:
            nj = (b >> j0) & 1
            nk = (b >> (j0 + 1)) & 1

            # Count each undirected hopping edge once from occupation 10.
            if not (nj == 1 and nk == 0):
                continue

            k = index[b ^ (1 << j0) ^ (1 << (j0 + 1))]
            ma = int(macro_id[i])
            mb = int(macro_id[k])
            if ma == mb:
                continue

            a = min(ma, mb)
            bb = max(ma, mb)

            if ma == a:
                ia, ib = i, k
            else:
                ia, ib = k, i

            pair_map.setdefault((a, bb), [[], [], []])
            pair_map[(a, bb)][0].append(ia)
            pair_map[(a, bb)][1].append(ib)
            pair_map[(a, bb)][2].append(hopping_strength(N, j0, mode))

    pairs = sorted(pair_map.keys())
    ia_all, ib_all, h_all, pid_all = [], [], [], []

    for pid, pair in enumerate(pairs):
        ia, ib, hh = pair_map[pair]
        ia_all.extend(ia)
        ib_all.extend(ib)
        h_all.extend(hh)
        pid_all.extend([pid] * len(ia))

    out = {
        "pairs": pairs,
        "ia": np.asarray(ia_all, dtype=int),
        "ib": np.asarray(ib_all, dtype=int),
        "h": np.asarray(h_all, dtype=float),
        "pid": np.asarray(pid_all, dtype=int),
    }
    _EDGE_CACHE[key] = out
    return out


def make_initial_columns(dim, init_idx, exact=True, ntrace=8, seed=20260810):
    D0 = len(init_idx)

    if exact:
        B = np.zeros((dim, D0), dtype=np.complex128)
        B[init_idx, np.arange(D0)] = 1.0
        weights = np.ones(D0, dtype=float) / D0
        return B, weights

    rng = np.random.default_rng(seed)
    B = np.zeros((dim, ntrace), dtype=np.complex128)
    for c in range(ntrace):
        phases = np.exp(2j * np.pi * rng.random(D0))
        B[init_idx, c] = phases / math.sqrt(D0)
    weights = np.ones(ntrace, dtype=float) / ntrace
    return B, weights


def entropy_from_p(p, Omega):
    p = np.asarray(p, dtype=float)
    safe = np.maximum(p, 1e-300)
    return -np.sum(np.where(p > 0.0, p * np.log(safe), 0.0), axis=-1) + np.sum(
        p * np.log(Omega), axis=-1
    )


def class_trajectory(
    N: int,
    Q: int,
    mode: str,
    K: int,
    init_mode: str,
    R0=None,
    dt=0.1,
    tmax=10.0,
    exact=True,
    ntrace=8,
    seed=20260810,
    compute_currents=False,
    block_steps=5,
):
    H, basis, _ = get_model(N, Q, mode)
    macro = get_macro_data(N, Q, K)
    Omega = macro["Omega"]
    macro_id = macro["macro_id"]
    init_idx = initial_indices(N, Q, K, init_mode, R0=R0)
    Hcross = get_cross_hamiltonian(N, Q, K, mode)

    B, weights = make_initial_columns(
        len(basis),
        init_idx,
        exact=exact,
        ntrace=ntrace,
        seed=seed,
    )

    nsteps = int(round(tmax / dt))
    target_times = np.arange(nsteps + 1) * dt

    p_list = []
    S_list = []
    dS_list = []
    sigma_spread_list = []
    sigma_ret_list = []

    edges = get_macro_edges(N, Q, K, mode) if compute_currents else None

    current_step = 0
    while current_step < nsteps:
        remaining = nsteps - current_step
        step_block = min(block_steps, remaining)

        traj = expm_multiply(
            -1j * H,
            B,
            start=0.0,
            stop=step_block * dt,
            num=step_block + 1,
            endpoint=True,
        )

        start_local = 0 if current_step == 0 else 1

        for local in range(start_local, step_block + 1):
            psi = traj[local]
            basis_prob = np.sum(np.abs(psi) ** 2 * weights[None, :], axis=1)
            p = np.bincount(macro_id, weights=basis_prob, minlength=len(Omega))

            dpsi_cross = -1j * (Hcross @ psi)
            basis_dprob = np.sum(
                2.0 * np.real(np.conjugate(psi) * dpsi_cross) * weights[None, :],
                axis=1,
            )
            dp = np.bincount(macro_id, weights=basis_dprob, minlength=len(Omega))
            dp -= dp.sum() / len(dp)

            safe = np.maximum(p, 1e-300)
            S = float(entropy_from_p(p, Omega))
            dS = float(-np.sum(dp * np.log(safe / Omega)))

            p_list.append(p)
            S_list.append(S)
            dS_list.append(dS)

            if compute_currents:
                corr = np.sum(
                    np.conjugate(psi[edges["ia"], :])
                    * psi[edges["ib"], :]
                    * weights[None, :],
                    axis=1,
                )
                edge_current = 2.0 * np.imag(edges["h"] * corr)
                pair_current = np.bincount(
                    edges["pid"],
                    weights=edge_current,
                    minlength=len(edges["pairs"]),
                )

                ell = np.log(safe / Omega)
                contributions = np.zeros(len(edges["pairs"]), dtype=float)
                for ip, (a, b) in enumerate(edges["pairs"]):
                    contributions[ip] = pair_current[ip] * (ell[b] - ell[a])

                sigma_spread = float(np.maximum(contributions, 0.0).sum())
                sigma_ret = float(np.maximum(-contributions, 0.0).sum())

                # Direct and pair-current forms should agree to roundoff.
                if abs((sigma_spread - sigma_ret) - dS) > 1e-8:
                    raise RuntimeError("Macrocurrent entropy identity failed.")

                sigma_spread_list.append(sigma_spread)
                sigma_ret_list.append(sigma_ret)

        B = traj[-1]
        current_step += step_block

    p = np.asarray(p_list)
    S = np.asarray(S_list)
    dS = np.asarray(dS_list)
    times = target_times[: len(S)]

    if init_mode == "left_half":
        S0_exact = log_binom(N // 2, Q)
    else:
        rid = macro["tuple_to_id"][tuple(R0)]
        S0_exact = math.log(Omega[rid])

    Smax = log_binom(N, Q)
    Sigma = (S - S0_exact) / (Smax - S0_exact)

    out = {
        "times": times,
        "p": p,
        "S": S,
        "dS": dS,
        "Sigma": Sigma,
        "Omega": Omega,
        "pi": Omega / Omega.sum(),
        "S0_exact": S0_exact,
        "Smax": Smax,
        "D0": len(init_idx),
        "dim": len(basis),
        "exact": exact,
        "ntrace": len(init_idx) if exact else ntrace,
    }

    if compute_currents:
        out["sigma_spread"] = np.asarray(sigma_spread_list)
        out["sigma_ret"] = np.asarray(sigma_ret_list)

    return out


def individual_history_trajectories(
    N: int,
    Q: int,
    mode: str,
    K: int,
    R0,
    dt=0.1,
    tmax=8.0,
    block_steps=5,
):
    H, basis, _ = get_model(N, Q, mode)
    macro = get_macro_data(N, Q, K)
    Omega = macro["Omega"]
    macro_id = macro["macro_id"]
    init_idx = initial_indices(N, Q, K, "fixed_record", R0=R0)
    Hcross = get_cross_hamiltonian(N, Q, K, mode)

    B = np.zeros((len(basis), len(init_idx)), dtype=np.complex128)
    B[init_idx, np.arange(len(init_idx))] = 1.0

    nsteps = int(round(tmax / dt))
    target_times = np.arange(nsteps + 1) * dt

    S_hist = []
    dS_hist = []

    current_step = 0
    while current_step < nsteps:
        remaining = nsteps - current_step
        step_block = min(block_steps, remaining)

        traj = expm_multiply(
            -1j * H,
            B,
            start=0.0,
            stop=step_block * dt,
            num=step_block + 1,
            endpoint=True,
        )

        start_local = 0 if current_step == 0 else 1

        for local in range(start_local, step_block + 1):
            psi = traj[local]
            dpsi_cross = -1j * (Hcross @ psi)
            prob = np.abs(psi) ** 2
            dprob = 2.0 * np.real(np.conjugate(psi) * dpsi_cross)

            p = np.zeros((len(Omega), len(init_idx)), dtype=float)
            dp = np.zeros_like(p)
            for rid in range(len(Omega)):
                mask = macro_id == rid
                p[rid] = prob[mask, :].sum(axis=0)
                dp[rid] = dprob[mask, :].sum(axis=0)

            dp -= dp.sum(axis=0, keepdims=True) / len(Omega)
            safe = np.maximum(p, 1e-300)
            S = -np.sum(np.where(p > 0.0, p * np.log(safe), 0.0), axis=0)
            S += np.sum(p * np.log(Omega[:, None]), axis=0)
            dS = -np.sum(dp * np.log(safe / Omega[:, None]), axis=0)

            S_hist.append(S)
            dS_hist.append(dS)

        B = traj[-1]
        current_step += step_block

    S_hist = np.asarray(S_hist)
    dS_hist = np.asarray(dS_hist)
    times = target_times[: len(S_hist)]

    rid = macro["tuple_to_id"][tuple(R0)]
    S0 = math.log(Omega[rid])
    Smax = log_binom(N, Q)
    Sigma_hist = (S_hist - S0) / (Smax - S0)

    return {
        "times": times,
        "S_hist": S_hist,
        "dS_hist": dS_hist,
        "Sigma_hist": Sigma_hist,
        "Omega": Omega,
        "D0": len(init_idx),
    }


# ============================================================
# Relative majorization and history-sensitive propagation
# ============================================================


def lorenz_curve(p, pi):
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    order = np.argsort(-(p / pi))
    x = np.concatenate([[0.0], np.cumsum(pi[order])])
    y = np.concatenate([[0.0], np.cumsum(p[order])])
    x[-1] = 1.0
    y[-1] = 1.0
    return x, y


def majorization_defect(p, q, pi):
    xp, yp = lorenz_curve(p, pi)
    xq, yq = lorenz_curve(q, pi)
    grid = np.unique(np.concatenate([xp, xq]))
    Lp = np.interp(grid, xp, yp)
    Lq = np.interp(grid, xq, yq)
    return float(max(0.0, np.max(Lq - Lp)))


def lorenz_margin(p, q, pi):
    xp, yp = lorenz_curve(p, pi)
    xq, yq = lorenz_curve(q, pi)
    grid = np.unique(np.concatenate([xp, xq]))
    Lp = np.interp(grid, xp, yp)
    Lq = np.interp(grid, xq, yq)
    return float(np.min(Lp - Lq))


def build_coarse_map_K(N: int, Q: int, mode: str, Kcells: int, tau: float, chunk=32):
    H, basis, _ = get_model(N, Q, mode)
    macro = get_macro_data(N, Q, Kcells)
    macro_id = macro["macro_id"]
    Omega = macro["Omega"]
    dim = len(basis)

    Kmat = np.zeros((len(Omega), len(Omega)), dtype=float)

    for start in range(0, dim, chunk):
        stop = min(start + chunk, dim)
        inds = np.arange(start, stop)
        B0 = np.zeros((dim, len(inds)), dtype=np.complex128)
        B0[inds, np.arange(len(inds))] = 1.0
        Bout = expm_multiply((-1j * tau) * H, B0)

        for c, idx0 in enumerate(inds):
            r0 = macro_id[idx0]
            macro_prob = np.bincount(
                macro_id,
                weights=np.abs(Bout[:, c]) ** 2,
                minlength=len(Omega),
            )
            Kmat[:, r0] += macro_prob / Omega[r0]

    return Kmat


def history_defect_dataset(N=16, Q=2, mode="mixing", dt=0.1, tmax=30.0):
    traj = class_trajectory(
        N=N,
        Q=Q,
        mode=mode,
        K=4,
        init_mode="left_half",
        dt=dt,
        tmax=tmax,
        exact=True,
        compute_currents=True,
    )

    Kmat = build_coarse_map_K(N, Q, mode, 4, dt)
    pi = traj["pi"]
    Omega = traj["Omega"]
    p = traj["p"]

    eps_hist = []
    Vpi = []
    dS_mix = []
    dS_corr = []
    dS_actual = []
    return_ratio = []
    margins = []

    for k in range(len(p) - 1):
        pk = p[k]
        q = p[k + 1]
        q0 = Kmat @ pk
        r = q - q0

        eps_hist.append(0.5 * np.sum(np.abs(r)))
        Vpi.append(majorization_defect(pk, q, pi))
        margins.append(lorenz_margin(pk, q, pi))

        Sp = float(entropy_from_p(pk, Omega))
        Smix = float(entropy_from_p(q0, Omega))
        Sq = float(entropy_from_p(q, Omega))

        dmix = Smix - Sp
        dcorr = Sq - Smix
        dactual = Sq - Sp

        dS_mix.append(dmix)
        dS_corr.append(dcorr)
        dS_actual.append(dactual)

        lowering = max(0.0, -dcorr)
        return_ratio.append(lowering / dmix if dmix > 1e-14 else 0.0)

    out = dict(traj)
    out.update(
        {
            "Kmat": Kmat,
            "eps_hist": np.asarray(eps_hist),
            "Vpi": np.asarray(Vpi),
            "margins": np.asarray(margins),
            "dS_mix": np.asarray(dS_mix),
            "dS_corr": np.asarray(dS_corr),
            "dS_actual_step": np.asarray(dS_actual),
            "return_ratio": np.asarray(return_ratio),
            "stochastic_error": float(np.max(np.abs(Kmat.sum(axis=0) - 1.0))),
            "pi_error": float(np.max(np.abs(Kmat @ pi - pi))),
        }
    )
    return out



# ============================================================
# Spectral return-suppression diagnostics
# ============================================================

_SPECTRAL_STATIC_CACHE = {}


def _cluster_sorted_indices(values, tol):
    """Cluster an already sorted one-dimensional array within absolute tolerance."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one dimensional.")
    if len(values) == 0:
        return []

    groups = []
    start = 0
    for k in range(1, len(values)):
        if abs(values[k] - values[k - 1]) > tol:
            groups.append(np.arange(start, k, dtype=int))
            start = k
    groups.append(np.arange(start, len(values), dtype=int))
    return groups


def _cluster_sorted_values_and_counts(values, tol):
    """Return numerical cluster centers and multiplicities for sorted values."""
    values = np.asarray(values, dtype=float)
    groups = _cluster_sorted_indices(values, tol)
    centers = np.array([float(np.mean(values[g])) for g in groups], dtype=float)
    counts = np.array([len(g) for g in groups], dtype=int)
    return centers, counts


def _gap_concentration_count(sorted_gaps, epsilon):
    """
    Exact sliding-window count for the finite numerical gap list.

    The interval convention is [x, x+epsilon), matching the definition used
    in the manuscript. Starting an optimal interval at its first included gap
    cannot decrease the number of included gaps, so it is sufficient to test
    windows starting at the sorted gap values.
    """
    gaps = np.asarray(sorted_gaps, dtype=float)
    if len(gaps) == 0:
        return 0
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    right = np.searchsorted(gaps, gaps + float(epsilon), side="left")
    counts = right - np.arange(len(gaps))
    return int(np.max(counts))


def _optimize_finite_time_gap_factor(sorted_gaps, dE, T, epsilon_min):
    """
    Minimize the Short-Farrelly finite-time gap factor over a fixed epsilon grid.

    Every epsilon on the grid yields a valid bound. Taking the minimum of this
    finite collection therefore remains a valid upper bound; it is not assumed
    to be the globally optimal continuous-epsilon choice.
    """
    if T <= 0.0:
        raise ValueError("The observation-window duration T must be positive.")
    gaps = np.asarray(sorted_gaps, dtype=float)
    if len(gaps) == 0:
        return {
            "epsilon_opt": np.nan,
            "N_epsilon_opt": 0,
            "gT_opt": 0.0,
            "spectral_factor_finite": 0.0,
        }

    gap_span = float(gaps[-1] - gaps[0])
    finite_scale = max(float(np.max(np.abs(gaps))), 1.0)
    eps_floor = max(100.0 * np.finfo(float).eps * finite_scale, 1e-14)

    if np.isfinite(epsilon_min) and epsilon_min > eps_floor:
        eps_low = max(0.5 * epsilon_min, eps_floor)
    else:
        positive = np.abs(gaps[np.abs(gaps) > eps_floor])
        eps_low = max(float(np.min(positive)) * 1e-3 if len(positive) else eps_floor, eps_floor)

    eps_high = max(gap_span * 1.001, eps_low * 10.0)
    eps_grid = np.geomspace(eps_low, eps_high, SPECTRAL_EPS_GRID_SIZE)

    if np.isfinite(epsilon_min) and epsilon_min > 0.0:
        eps_grid = np.unique(
            np.concatenate(
                [
                    eps_grid,
                    np.array(
                        [
                            max(0.5 * epsilon_min, eps_floor),
                            max(0.999 * epsilon_min, eps_floor),
                            epsilon_min,
                            1.001 * epsilon_min,
                        ],
                        dtype=float,
                    ),
                ]
            )
        )

    coeff = 8.0 * math.log2(max(int(dE), 2)) / float(T)
    best = None

    for eps in eps_grid:
        N_eps = _gap_concentration_count(gaps, eps)
        gT = float(N_eps * (1.0 + coeff / eps))
        if best is None or gT < best[0]:
            best = (gT, float(eps), int(N_eps))

    gT_opt, eps_opt, N_opt = best
    return {
        "epsilon_opt": eps_opt,
        "N_epsilon_opt": N_opt,
        "gT_opt": gT_opt,
    }


def _spectral_static_data(N, Q, mode, Kcells=4, init_mode="left_half", R0=None):
    """
    Spectral data entering the return-suppression theorem.

    The exact preparation is the macro-uniform thermodynamic class used in the
    corresponding trajectory. Degenerate energy eigenspaces are retained as
    blocks when constructing the dephased state omega.
    """
    key = (N, Q, mode, Kcells, init_mode, None if R0 is None else tuple(R0))
    if key in _SPECTRAL_STATIC_CACHE:
        return _SPECTRAL_STATIC_CACHE[key]

    H, basis, _ = get_model(N, Q, mode)
    H_dense = np.asarray(H.toarray(), dtype=np.complex128)
    evals, evecs = np.linalg.eigh(H_dense)

    spectral_span = float(np.ptp(evals)) if len(evals) > 1 else 0.0
    energy_tol = max(1e-11 * max(1.0, spectral_span), 1e-12)
    energy_groups = _cluster_sorted_indices(evals, energy_tol)
    distinct_energies = np.array(
        [float(np.mean(evals[g])) for g in energy_groups],
        dtype=float,
    )
    dE = len(distinct_energies)

    init_idx = initial_indices(N, Q, Kcells, init_mode, R0=R0)
    D0 = len(init_idx)

    # rho_0 = P_init/D0 in the computational basis. Transform it to the
    # energy basis without explicitly forming a dense computational rho_0.
    Vinit = evecs[init_idx, :]
    rho_energy = (Vinit.conj().T @ Vinit) / float(D0)

    energy_weights = np.array(
        [float(np.trace(rho_energy[np.ix_(g, g)]).real) for g in energy_groups],
        dtype=float,
    )
    energy_weights /= energy_weights.sum()
    deff = float(1.0 / np.sum(energy_weights**2))

    rho_energy_deph = np.zeros_like(rho_energy)
    for g in energy_groups:
        rho_energy_deph[np.ix_(g, g)] = rho_energy[np.ix_(g, g)]
    omega = evecs @ rho_energy_deph @ evecs.conj().T
    omega = 0.5 * (omega + omega.conj().T)

    macro = get_macro_data(N, Q, Kcells)
    macro_id = macro["macro_id"]
    Omega = macro["Omega"]
    macro_indices = [np.where(macro_id == r)[0] for r in range(len(Omega))]

    pair_a = []
    pair_b = []
    current_norms = []
    Jomega = []

    coupling_tol = max(1e-12, 1e-11 * max(1.0, float(np.linalg.norm(H_dense, 2))))

    for a in range(len(Omega)):
        ia = macro_indices[a]
        for b in range(a + 1, len(Omega)):
            ib = macro_indices[b]
            block = H_dense[np.ix_(ia, ib)]
            if block.size == 0:
                continue

            svals = np.linalg.svd(block, compute_uv=False)
            block_norm = float(svals[0]) if len(svals) else 0.0
            if block_norm <= coupling_tol:
                continue

            # For I_e = i(P_b H P_a - P_a H P_b), ||I_e||_infty equals
            # the largest singular value of the off-diagonal block P_a H P_b.
            omega_ba = omega[np.ix_(ib, ia)]
            z = np.trace(block @ omega_ba)
            jomega = float(2.0 * np.imag(z))

            pair_a.append(a)
            pair_b.append(b)
            current_norms.append(block_norm)
            Jomega.append(jomega)

    pair_a = np.asarray(pair_a, dtype=int)
    pair_b = np.asarray(pair_b, dtype=int)
    current_norms = np.asarray(current_norms, dtype=float)
    Jomega = np.asarray(Jomega, dtype=float)

    I2 = float(np.sqrt(np.sum(current_norms**2)))
    Jomega_norm = float(np.linalg.norm(Jomega))

    if dE > 1:
        gap_matrix = distinct_energies[:, None] - distinct_energies[None, :]
        gap_mask = ~np.eye(dE, dtype=bool)
        ordered_gaps = np.sort(gap_matrix[gap_mask].ravel())

        gap_scale = max(1.0, float(np.max(np.abs(ordered_gaps))))
        gap_tol = max(5.0 * energy_tol, 1e-11 * gap_scale)
        gap_centers, gap_counts = _cluster_sorted_values_and_counts(ordered_gaps, gap_tol)
        DG = int(np.max(gap_counts))

        if len(gap_centers) > 1:
            gap_separations = np.diff(gap_centers)
            epsilon_min = float(np.min(gap_separations[gap_separations > gap_tol])) \
                if np.any(gap_separations > gap_tol) else np.nan
        else:
            epsilon_min = np.nan
    else:
        ordered_gaps = np.array([], dtype=float)
        gap_centers = np.array([], dtype=float)
        gap_counts = np.array([], dtype=int)
        DG = 0
        epsilon_min = np.nan
        gap_tol = np.nan

    data = {
        "N": N,
        "Q": Q,
        "mode": mode,
        "Kcells": Kcells,
        "dim": len(basis),
        "D0": D0,
        "dE": dE,
        "deff": deff,
        "energy_tol": energy_tol,
        "evals": evals,
        "evecs": evecs,
        "energy_groups": energy_groups,
        "distinct_energies": distinct_energies,
        "energy_weights": energy_weights,
        "rho_energy": rho_energy,
        "omega": omega,
        "H_dense": H_dense,
        "macro_indices": macro_indices,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "current_norms": current_norms,
        "I2": I2,
        "Jomega": Jomega,
        "Jomega_norm": Jomega_norm,
        "ordered_gaps": ordered_gaps,
        "gap_centers": gap_centers,
        "gap_counts": gap_counts,
        "gap_tol": gap_tol,
        "DG": DG,
        "epsilon_min": epsilon_min,
        "spectral_factor_long": float(math.sqrt(DG / deff)) if deff > 0.0 else np.inf,
    }

    _SPECTRAL_STATIC_CACHE[key] = data
    return data


def _longest_contiguous_true_run(mask):
    """Return the inclusive index range of the longest True run in a Boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    best = None
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if start is not None and ((not val) or i == len(mask) - 1):
            stop = i if val and i == len(mask) - 1 else i - 1
            length = stop - start + 1
            if best is None or length > best[0]:
                best = (length, start, stop)
            start = None

    if best is None:
        return None
    return best[1], best[2]


def _primary_time_bounds(traj):
    sigma = np.asarray(traj["Sigma"], dtype=float)
    i10s = np.where(sigma >= 0.10)[0]
    if len(i10s) == 0:
        raise RuntimeError("Trajectory does not reach 10% of the entropy range.")
    i10 = int(i10s[0])

    i90s = np.where((np.arange(len(sigma)) >= i10) & (sigma >= 0.90))[0]
    if len(i90s) == 0:
        raise RuntimeError("Trajectory does not reach 90% of the entropy range.")
    i90 = int(i90s[0])

    return float(traj["times"][i10]), float(traj["times"][i90])


def _interior_window_indices(traj, static, requested_t0, requested_t1, p_tol):
    """
    Select the longest numerically resolved interior subwindow.

    The mathematical theorem requires strictly positive probabilities for all
    coupled macrospaces. Numerically we impose the stronger p > p_tol condition
    so that logarithmic affinities are not dominated by floating-point noise.
    """
    times = np.asarray(traj["times"], dtype=float)
    p = np.asarray(traj["p"], dtype=float)

    coupled = np.unique(np.concatenate([static["pair_a"], static["pair_b"]]))
    in_time = (times >= requested_t0 - 1e-12) & (times <= requested_t1 + 1e-12)

    if len(coupled):
        interior = np.all(p[:, coupled] > p_tol, axis=1)
    else:
        interior = np.ones(len(times), dtype=bool)

    run = _longest_contiguous_true_run(in_time & interior)
    if run is None:
        min_prob = float(np.min(p[in_time][:, coupled])) if np.any(in_time) and len(coupled) else np.nan
        raise RuntimeError(
            f"No interior spectral window found in [{requested_t0}, {requested_t1}] "
            f"with p_tol={p_tol:g}; minimum coupled probability was {min_prob:.3e}."
        )

    i0, i1 = run
    if i1 - i0 < 2:
        raise RuntimeError("Interior spectral window has fewer than three samples.")

    return np.arange(i0, i1 + 1, dtype=int)


def spectral_window_diagnostics(
    N,
    Q,
    mode,
    traj,
    window_label,
    requested_t0,
    requested_t1,
    Kcells=4,
    p_tol=SPECTRAL_INTERIOR_P_TOL,
):
    """Evaluate the complete spectral return-suppression bound on one window."""
    static = _spectral_static_data(
        N=N,
        Q=Q,
        mode=mode,
        Kcells=Kcells,
        init_mode="left_half",
    )

    idx = _interior_window_indices(
        traj,
        static,
        requested_t0=requested_t0,
        requested_t1=requested_t1,
        p_tol=p_tol,
    )

    times = np.asarray(traj["times"], dtype=float)[idx]
    p = np.asarray(traj["p"], dtype=float)[idx]
    sigma_ret = np.asarray(traj["sigma_ret"], dtype=float)[idx]
    sigma_spread = np.asarray(traj["sigma_spread"], dtype=float)[idx]
    dS = np.asarray(traj["dS"], dtype=float)[idx]
    Omega = np.asarray(traj["Omega"], dtype=float)

    T = float(times[-1] - times[0])
    if T <= 0.0:
        raise RuntimeError("Spectral window duration vanished.")

    a = static["pair_a"]
    b = static["pair_b"]

    if len(a):
        affinities = np.log(
            (p[:, b] / Omega[b][None, :])
            / (p[:, a] / Omega[a][None, :])
        )
        A2_t = np.sqrt(np.sum(affinities**2, axis=1))
        A2 = float(np.max(A2_t))
    else:
        affinities = np.zeros((len(times), 0), dtype=float)
        A2_t = np.zeros(len(times), dtype=float)
        A2 = 0.0

    rms_sigma_ret = float(
        math.sqrt(np.trapz(sigma_ret**2, times) / T)
    )
    mean_sigma_ret = float(np.trapz(sigma_ret, times) / T)
    max_sigma_ret = float(np.max(sigma_ret))
    s0 = float(np.min(sigma_spread))

    neg_indicator = (dS <= 0.0).astype(float)
    negative_rate_fraction = float(np.trapz(neg_indicator, times) / T)

    opt = _optimize_finite_time_gap_factor(
        static["ordered_gaps"],
        static["dE"],
        T,
        static["epsilon_min"],
    )
    gT_opt = float(opt["gT_opt"])
    finite_spectral_factor = float(math.sqrt(gT_opt / static["deff"]))

    I2 = static["I2"]
    Jomega_norm = static["Jomega_norm"]

    bound_fluctuation = float(A2 * I2 * finite_spectral_factor)
    bound_general = float(A2 * (Jomega_norm + I2 * finite_spectral_factor))

    # The static Hamiltonians and thermodynamic-class preparations are real in
    # the computational basis, so the time-reversal corollary predicts J^omega=0.
    # We retain the directly evaluated residual current as an explicit numerical
    # symmetry check rather than silently setting it to zero.
    tr_symmetry_resolved = bool(Jomega_norm <= 1e-10 * max(1.0, I2))

    normalized_actual = (
        float(rms_sigma_ret / (A2 * I2))
        if A2 > 0.0 and I2 > 0.0
        else np.nan
    )
    bound_ratio = float(rms_sigma_ret / bound_general) if bound_general > 0.0 else np.nan

    chebyshev_general = (
        float(min(1.0, (bound_general / s0) ** 2))
        if s0 > 0.0
        else np.inf
    )
    chebyshev_tr = (
        float(min(1.0, (bound_fluctuation / s0) ** 2))
        if s0 > 0.0
        else np.inf
    )

    coupled = np.unique(np.concatenate([a, b])) if len(a) else np.array([], dtype=int)
    min_coupled_probability = (
        float(np.min(p[:, coupled])) if len(coupled) else 1.0
    )

    row = {
        "N": int(N),
        "Q": int(Q),
        "mode": mode,
        "window": window_label,
        "requested_t0": float(requested_t0),
        "requested_t1": float(requested_t1),
        "t0": float(times[0]),
        "t1": float(times[-1]),
        "T": T,
        "n_samples": int(len(times)),
        "p_tol": float(p_tol),
        "min_coupled_probability": min_coupled_probability,
        "A2": A2,
        "I2": float(I2),
        "Jomega_norm": float(Jomega_norm),
        "time_reversal_stationary_current_resolved": tr_symmetry_resolved,
        "dE": int(static["dE"]),
        "deff": float(static["deff"]),
        "DG": int(static["DG"]),
        "DG_over_deff": float(static["DG"] / static["deff"]),
        "spectral_factor_long": float(static["spectral_factor_long"]),
        "epsilon_min": float(static["epsilon_min"]),
        "epsilon_opt": float(opt["epsilon_opt"]),
        "N_epsilon_opt": int(opt["N_epsilon_opt"]),
        "gT_opt": gT_opt,
        "spectral_factor_finite": finite_spectral_factor,
        "rms_sigma_ret": rms_sigma_ret,
        "mean_sigma_ret": mean_sigma_ret,
        "max_sigma_ret": max_sigma_ret,
        "normalized_rms_sigma_ret": normalized_actual,
        "bound_fluctuation_TR": bound_fluctuation,
        "bound_general": bound_general,
        "actual_over_general_bound": bound_ratio,
        "s0_min_sigma_spread": s0,
        "negative_rate_fraction": negative_rate_fraction,
        "chebyshev_fraction_bound_general": chebyshev_general,
        "chebyshev_fraction_bound_TR": chebyshev_tr,
    }
    return row


def _write_csv_rows(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved diagnostics: {path}")


def _print_spectral_row(row):
    print(
        f"{row['mode']:>11s} | {row['window']:<18s} | "
        f"Jt=[{row['t0']:.3f},{row['t1']:.3f}] | "
        f"d_eff={row['deff']:.6g} | D_G={row['DG']} | "
        f"D_G/d_eff={row['DG_over_deff']:.6e}"
    )
    print(
        f"    A2={row['A2']:.6e}, I2/J={row['I2']:.6e}, "
        f"||J^omega||_2/J={row['Jomega_norm']:.3e}, "
        f"epsilon_opt/J={row['epsilon_opt']:.6e}, "
        f"N(epsilon_opt)={row['N_epsilon_opt']}"
    )
    print(
        f"    RMS sigma_ret/(kB J)={row['rms_sigma_ret']:.6e}, "
        f"full bound/(kB J)={row['bound_general']:.6e}, "
        f"actual/bound={row['actual_over_general_bound']:.6e}, "
        f"finite spectral factor={row['spectral_factor_finite']:.6e}"
    )
    print(
        f"    min sigma_spread/(kB J)={row['s0_min_sigma_spread']:.6e}, "
        f"actual fraction(dot S_R<=0)={row['negative_rate_fraction']:.6e}, "
        f"Chebyshev bound={row['chebyshev_fraction_bound_general']:.6e}"
    )


def spectral_N16_Q2_diagnostics():
    """Detailed theorem test for the two central N=16,Q=2 Hamiltonians."""
    rows = []

    for mode in ("mixing", "engineered"):
        traj = cached(
            f"spectral_v1_N16_Q2_{mode}_dt{SPECTRAL_N16_DT:.3f}",
            lambda mode=mode: class_trajectory(
                16,
                2,
                mode,
                4,
                "left_half",
                dt=SPECTRAL_N16_DT,
                tmax=30.0,
                exact=True,
                compute_currents=True,
                block_steps=5,
            ),
        )

        p0, p1 = _primary_time_bounds(traj)
        rows.append(
            spectral_window_diagnostics(
                16,
                2,
                mode,
                traj,
                window_label="primary_10_90",
                requested_t0=p0,
                requested_t1=p1,
            )
        )

        rows.append(
            spectral_window_diagnostics(
                16,
                2,
                mode,
                traj,
                window_label="common_8_11",
                requested_t0=SPECTRAL_COMMON_WINDOW[0],
                requested_t1=SPECTRAL_COMMON_WINDOW[1],
            )
        )

    path = SPECTRAL_DIR / "spectral_theorem_N16_Q2.csv"
    _write_csv_rows(path, rows)

    print("\n=== Spectral return-suppression diagnostics: N=16, Q=2 ===")
    for row in rows:
        _print_spectral_row(row)

    return rows


def spectral_size_sequence_diagnostics():
    """
    Fixed-Q=2 size sequence for the mixing Hamiltonian.

    The exact trajectory is used at every size. The theorem is evaluated on
    each system's own numerically resolved primary 10%-90% relaxation window.
    """
    rows = []

    for N, Q in SPECTRAL_SIZE_SYSTEMS:
        traj = cached(
            f"spectral_v1_size_mixing_N{N}_Q{Q}_dt{SPECTRAL_SIZE_DT:.3f}",
            lambda N=N, Q=Q: class_trajectory(
                N,
                Q,
                "mixing",
                4,
                "left_half",
                dt=SPECTRAL_SIZE_DT,
                tmax=30.0,
                exact=True,
                compute_currents=True,
                block_steps=5,
            ),
        )

        p0, p1 = _primary_time_bounds(traj)
        rows.append(
            spectral_window_diagnostics(
                N,
                Q,
                "mixing",
                traj,
                window_label="primary_10_90",
                requested_t0=p0,
                requested_t1=p1,
            )
        )

        # The primary Q=2 mixing windows have sigma_ret=0 at the sampled
        # times. A separate common late window is therefore also evaluated to
        # test whether the finite-size spectral organization tracks the
        # nonzero return-oriented current burden after the primary relaxation.
        rows.append(
            spectral_window_diagnostics(
                N,
                Q,
                "mixing",
                traj,
                window_label="late_8_30",
                requested_t0=8.0,
                requested_t1=30.0,
            )
        )

    path = SPECTRAL_DIR / "spectral_theorem_size_sequence_Q2.csv"
    _write_csv_rows(path, rows)

    print("\n=== Spectral return-suppression size sequence: fixed Q=2 ===")
    for row in rows:
        _print_spectral_row(row)

    return rows


# ============================================================
# Supplementary Figures S14-S16: quantitative theorem tests
# ============================================================


def make_supp_S14_spectral_theorem_test():
    if not RUN_SPECTRAL_THEOREM_TESTS:
        print("Skipping S14 because RUN_SPECTRAL_THEOREM_TESTS=False")
        return

    rows = spectral_N16_Q2_diagnostics()
    rows = [r for r in rows if r["window"] == "common_8_11"]
    rows = sorted(rows, key=lambda r: 0 if r["mode"] == "mixing" else 1)

    x = np.arange(len(rows), dtype=float)
    actual = np.array([r["rms_sigma_ret"] for r in rows], dtype=float)
    bound = np.array([r["bound_general"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.semilogy(
        x,
        actual,
        color=RED,
        marker="o",
        linestyle="-",
        label=r"actual $\sqrt{\langle\sigma_{\rm ret}^2\rangle}$",
    )
    ax.semilogy(
        x,
        bound,
        color=NAVY,
        marker="s",
        linestyle="--",
        label="spectral upper bound",
    )
    ax.set_xticks(x, ["mixing", "engineered\nreturn"])
    ax.set_ylabel(r"$\sqrt{\langle\sigma_{\rm ret}^2\rangle}/(k_{\rm B}J)$")
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=12,
    )
    savefig(fig, SUPP_DIR / "figS14_spectral_theorem_test.pdf")


def make_supp_S15_spectral_size_sequence():
    if not RUN_SPECTRAL_THEOREM_TESTS:
        print("Skipping S15 because RUN_SPECTRAL_THEOREM_TESTS=False")
        return

    rows = [
        r for r in spectral_size_sequence_diagnostics()
        if r["window"] == "late_8_30"
    ]
    Nvals = np.array([r["N"] for r in rows], dtype=int)
    actual_norm = np.array([r["normalized_rms_sigma_ret"] for r in rows], dtype=float)
    theorem_factor = np.array([r["spectral_factor_finite"] for r in rows], dtype=float)
    long_factor = np.array([r["spectral_factor_long"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.semilogy(
        Nvals,
        actual_norm,
        color=RED,
        marker="o",
        linestyle="-",
        label=r"$\sqrt{\langle\sigma_{\rm ret}^2\rangle}/(\mathcal{A}_2\mathcal{I}_2)$",
    )
    ax.semilogy(
        Nvals,
        theorem_factor,
        color=BLUE,
        marker="s",
        linestyle="--",
        label=r"$\sqrt{g_T(\epsilon_{\rm opt})/d_{\rm eff}}$",
    )
    ax.semilogy(
        Nvals,
        long_factor,
        color=GREEN,
        marker="^",
        linestyle="-.",
        label=r"$\sqrt{D_G/d_{\rm eff}}$",
    )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("dimensionless return scale")
    ax.set_xticks(Nvals)
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=11,
    )
    savefig(fig, SUPP_DIR / "figS15_spectral_size_sequence.pdf")


def make_supp_S16_gap_concentration_vs_return():
    if not RUN_SPECTRAL_THEOREM_TESTS:
        print("Skipping S16 because RUN_SPECTRAL_THEOREM_TESTS=False")
        return

    rows = [
        r for r in spectral_size_sequence_diagnostics()
        if r["window"] == "late_8_30"
    ]
    x = np.array([r["DG_over_deff"] for r in rows], dtype=float)
    y = np.array([r["rms_sigma_ret"] for r in rows], dtype=float)
    Nvals = np.array([r["N"] for r in rows], dtype=int)

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.loglog(
        x,
        y,
        color=GREEN,
        marker="o",
        linestyle="-",
        label=r"mixing, fixed $Q=2$",
    )
    for xx, yy, N in zip(x, y, Nvals):
        ax.annotate(
            rf"$N={N}$",
            xy=(xx, yy),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set_xlabel(r"$D_G/d_{\rm eff}$")
    ax.set_ylabel(r"$\sqrt{\langle\sigma_{\rm ret}^2\rangle}/(k_{\rm B}J)$")
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=12,
    )
    savefig(fig, SUPP_DIR / "figS16_gap_concentration_vs_return.pdf")



# ============================================================
# Exact state-resolved spectral macrocurrent decomposition
# ============================================================

_SPECTRAL_CURRENT_CACHE = {}


def _exact_spectral_current_data(N, Q, mode, Kcells=4, init_mode="left_half", R0=None):
    r"""
    Construct the exact spectral decomposition of all macrocurrents.

    For distinct energy projectors Pi_alpha and a macro-edge e=(R,R'), define

        c_{e,alpha beta} = Tr[I_e Pi_alpha rho_0 Pi_beta],
        C_{e,g} = sum_{E_alpha-E_beta=g} c_{e,alpha beta}.

    Then

        J_e(t) = J_e^omega + sum_{g != 0} C_{e,g} exp(-i g t).

    The long-time fluctuating current power is

        P_spec = sum_e sum_g |C_{e,g}|^2,

    while

        P_inc = sum_e sum_{alpha != beta} |c_{e,alpha beta}|^2

    is the corresponding incoherent transition power. Their ratio

        Gamma_spec = P_spec / P_inc

    measures coherent addition among transitions with exactly equal energy
    gaps. Cauchy-Schwarz gives 0 <= Gamma_spec <= D_G, and Gamma_spec=1
    whenever all nonzero gaps are nondegenerate.
    """
    key = (N, Q, mode, Kcells, init_mode, None if R0 is None else tuple(R0))
    if key in _SPECTRAL_CURRENT_CACHE:
        return _SPECTRAL_CURRENT_CACHE[key]

    static = _spectral_static_data(
        N=N,
        Q=Q,
        mode=mode,
        Kcells=Kcells,
        init_mode=init_mode,
        R0=R0,
    )

    H_dense = static["H_dense"]
    evecs = static["evecs"]
    energy_groups = static["energy_groups"]
    energies = static["distinct_energies"]
    rho_energy = static["rho_energy"]
    macro_indices = static["macro_indices"]
    pair_a = static["pair_a"]
    pair_b = static["pair_b"]

    dE = len(energies)
    if dE <= 1:
        raise RuntimeError("At least two distinct energies are required.")

    # Ordered transitions between distinct energy eigenspaces.
    trans_alpha = []
    trans_beta = []
    trans_gaps = []
    for alpha in range(dE):
        for beta in range(dE):
            if alpha == beta:
                continue
            trans_alpha.append(alpha)
            trans_beta.append(beta)
            trans_gaps.append(float(energies[alpha] - energies[beta]))

    trans_alpha = np.asarray(trans_alpha, dtype=int)
    trans_beta = np.asarray(trans_beta, dtype=int)
    trans_gaps = np.asarray(trans_gaps, dtype=float)

    order = np.argsort(trans_gaps)
    sorted_gaps = trans_gaps[order]
    gap_scale = max(1.0, float(np.max(np.abs(sorted_gaps))))
    gap_tol = max(5.0 * static["energy_tol"], 1e-11 * gap_scale)
    sorted_gap_groups = _cluster_sorted_indices(sorted_gaps, gap_tol)

    gap_labels = np.empty(len(trans_gaps), dtype=int)
    gap_centers = np.empty(len(sorted_gap_groups), dtype=float)
    gap_degeneracies = np.empty(len(sorted_gap_groups), dtype=int)
    for gid, sorted_idx in enumerate(sorted_gap_groups):
        transition_idx = order[sorted_idx]
        gap_labels[transition_idx] = gid
        gap_centers[gid] = float(np.mean(trans_gaps[transition_idx]))
        gap_degeneracies[gid] = len(transition_idx)

    DG = int(np.max(gap_degeneracies))
    if DG != static["DG"]:
        raise RuntimeError(
            f"Gap-degeneracy mismatch: current decomposition gives {DG}, "
            f"static spectral data give {static['DG']}."
        )

    n_edges = len(pair_a)
    n_gaps = len(gap_centers)
    n_trans = len(trans_gaps)

    C_gap = np.zeros((n_edges, n_gaps), dtype=np.complex128)
    Jomega = np.zeros(n_edges, dtype=float)
    P_incoherent_edge = np.zeros(n_edges, dtype=float)
    P_spectral_edge = np.zeros(n_edges, dtype=float)
    Gamma_edge = np.full(n_edges, np.nan, dtype=float)

    # Compute each current operator directly in the energy basis. This avoids
    # constructing unnecessary projectors while preserving the exact
    # macro-edge orientation used by class_trajectory().
    for e, (a, b) in enumerate(zip(pair_a, pair_b)):
        ia = macro_indices[a]
        ib = macro_indices[b]
        Va = evecs[ia, :]
        Vb = evecs[ib, :]
        H_ab = H_dense[np.ix_(ia, ib)]
        H_ba = H_dense[np.ix_(ib, ia)]

        I_energy = (
            1j * (Vb.conj().T @ H_ba @ Va)
            - 1j * (Va.conj().T @ H_ab @ Vb)
        )
        I_energy = 0.5 * (I_energy + I_energy.conj().T)

        # Stationary current in the energy-dephased state.
        j0 = 0.0 + 0.0j
        for alpha, ga in enumerate(energy_groups):
            j0 += np.trace(
                I_energy[np.ix_(ga, ga)]
                @ rho_energy[np.ix_(ga, ga)]
            )
        Jomega[e] = float(np.real_if_close(j0, tol=1000).real)

        c_transition = np.empty(n_trans, dtype=np.complex128)
        for k, (alpha, beta) in enumerate(zip(trans_alpha, trans_beta)):
            ga = energy_groups[alpha]
            gb = energy_groups[beta]
            c_transition[k] = np.trace(
                I_energy[np.ix_(gb, ga)]
                @ rho_energy[np.ix_(ga, gb)]
            )

        P_incoherent_edge[e] = float(np.sum(np.abs(c_transition) ** 2))
        np.add.at(C_gap[e], gap_labels, c_transition)
        P_spectral_edge[e] = float(np.sum(np.abs(C_gap[e]) ** 2))

        if P_incoherent_edge[e] > 1e-30:
            Gamma_edge[e] = P_spectral_edge[e] / P_incoherent_edge[e]

    P_incoherent = float(np.sum(P_incoherent_edge))
    P_spectral = float(np.sum(P_spectral_edge))
    if P_incoherent <= 0.0:
        Gamma_spec = np.nan
    else:
        Gamma_spec = float(P_spectral / P_incoherent)

    # Current reality requires C_{e,-g}=C_{e,g}^*. Check this numerically.
    conjugacy_error = 0.0
    for gid, g in enumerate(gap_centers):
        partner = int(np.argmin(np.abs(gap_centers + g)))
        if abs(gap_centers[partner] + g) <= 10.0 * gap_tol:
            conjugacy_error = max(
                conjugacy_error,
                float(np.max(np.abs(C_gap[:, partner] - C_gap[:, gid].conj()))),
            )

    # Compare the stationary current with the independently evaluated value
    # already contained in _spectral_static_data().
    stationary_error = float(np.max(np.abs(Jomega - static["Jomega"]))) if n_edges else 0.0

    # The exact gap-coherence theorem must hold up to numerical tolerance.
    if np.isfinite(Gamma_spec) and Gamma_spec > DG * (1.0 + 1e-8) + 1e-10:
        raise RuntimeError(
            f"Gamma_spec={Gamma_spec:.12g} exceeded D_G={DG}; "
            "check gap clustering or current amplitudes."
        )
    if DG == 1 and np.isfinite(Gamma_spec) and abs(Gamma_spec - 1.0) > 1e-8:
        raise RuntimeError(
            f"Nondegenerate-gap spectrum should give Gamma_spec=1, got {Gamma_spec:.12g}."
        )

    gap_power = np.sum(np.abs(C_gap) ** 2, axis=0)

    # Positive and negative gaps carry equal current power for Hermitian
    # currents. Use only positive gaps to quantify how concentrated the
    # physically distinct oscillation frequencies are.
    positive_gap_mask = gap_centers > 10.0 * gap_tol
    positive_gap_power = gap_power[positive_gap_mask]
    positive_gap_power = positive_gap_power[positive_gap_power > 1e-30]
    if len(positive_gap_power):
        gap_weights = positive_gap_power / np.sum(positive_gap_power)
        gap_participation_number = float(1.0 / np.sum(gap_weights**2))
        sorted_weights = np.sort(gap_weights)[::-1]
        cumulative_weights = np.cumsum(sorted_weights)
        n_gap_50 = int(np.searchsorted(cumulative_weights, 0.50) + 1)
        n_gap_90 = int(np.searchsorted(cumulative_weights, 0.90) + 1)
        max_gap_power_share = float(sorted_weights[0])
    else:
        gap_participation_number = np.nan
        n_gap_50 = 0
        n_gap_90 = 0
        max_gap_power_share = np.nan

    data = {
        "N": int(N),
        "Q": int(Q),
        "mode": mode,
        "Kcells": int(Kcells),
        "pair_a": pair_a.copy(),
        "pair_b": pair_b.copy(),
        "gap_centers": gap_centers,
        "gap_degeneracies": gap_degeneracies,
        "gap_tol": float(gap_tol),
        "C_gap": C_gap,
        "Jomega": Jomega,
        "P_incoherent_edge": P_incoherent_edge,
        "P_spectral_edge": P_spectral_edge,
        "Gamma_edge": Gamma_edge,
        "P_incoherent": P_incoherent,
        "P_spectral": P_spectral,
        "Gamma_spec": Gamma_spec,
        "sqrt_Gamma_spec": float(math.sqrt(Gamma_spec)) if Gamma_spec >= 0.0 else np.nan,
        "DG": DG,
        "Gamma_over_DG": float(Gamma_spec / DG) if DG > 0 else np.nan,
        "long_time_current_rms_fluct": float(math.sqrt(P_spectral)),
        "incoherent_current_rms_scale": float(math.sqrt(P_incoherent)),
        "gap_power": gap_power,
        "gap_participation_number": gap_participation_number,
        "n_gap_50": n_gap_50,
        "n_gap_90": n_gap_90,
        "max_gap_power_share": max_gap_power_share,
        "conjugacy_error": conjugacy_error,
        "stationary_current_error": stationary_error,
        "Jomega_norm": float(np.linalg.norm(Jomega)),
        "dE": int(static["dE"]),
        "deff": float(static["deff"]),
    }

    _SPECTRAL_CURRENT_CACHE[key] = data
    return data


def _reconstruct_spectral_currents(spec_data, times, chunk=SPECTRAL_CURRENT_TIME_CHUNK):
    """Reconstruct all macrocurrents from the exact gap-resolved amplitudes."""
    times = np.asarray(times, dtype=float)
    gaps = np.asarray(spec_data["gap_centers"], dtype=float)
    C_gap = np.asarray(spec_data["C_gap"], dtype=np.complex128)
    Jomega = np.asarray(spec_data["Jomega"], dtype=float)

    out = np.empty((len(times), C_gap.shape[0]), dtype=float)
    max_imag = 0.0

    for start in range(0, len(times), int(chunk)):
        stop = min(start + int(chunk), len(times))
        phase = np.exp(-1j * times[start:stop, None] * gaps[None, :])
        block = phase @ C_gap.T
        block += Jomega[None, :]
        max_imag = max(max_imag, float(np.max(np.abs(block.imag))))
        out[start:stop] = block.real

    return out, max_imag


def exact_spectral_current_window_diagnostics(
    N,
    Q,
    mode,
    traj,
    window_label,
    requested_t0,
    requested_t1,
    Kcells=4,
    p_tol=SPECTRAL_INTERIOR_P_TOL,
):
    r"""
    Evaluate the exact current-specific spectral representation on one window.

    The reconstructed current is combined with the actual macrostate affinity
    A_e(t) to reproduce sigma_spread, sigma_ret, and dot S_R. This provides a
    stringent end-to-end test of the spectral decomposition.
    """
    static = _spectral_static_data(N, Q, mode, Kcells=Kcells, init_mode="left_half")
    spec = _exact_spectral_current_data(N, Q, mode, Kcells=Kcells, init_mode="left_half")

    idx = _interior_window_indices(
        traj,
        static,
        requested_t0=requested_t0,
        requested_t1=requested_t1,
        p_tol=p_tol,
    )

    times = np.asarray(traj["times"], dtype=float)[idx]
    p = np.asarray(traj["p"], dtype=float)[idx]
    Omega = np.asarray(traj["Omega"], dtype=float)
    T = float(times[-1] - times[0])
    if T <= 0.0:
        raise RuntimeError("Current-spectral window duration vanished.")

    J_spec, current_imag_error = _reconstruct_spectral_currents(spec, times)

    a = spec["pair_a"]
    b = spec["pair_b"]
    affinities = np.log(
        (p[:, b] / Omega[b][None, :])
        / (p[:, a] / Omega[a][None, :])
    )
    contributions = J_spec * affinities
    sigma_spread_spec = np.maximum(contributions, 0.0).sum(axis=1)
    sigma_ret_spec = np.maximum(-contributions, 0.0).sum(axis=1)
    dS_spec = contributions.sum(axis=1)

    sigma_spread_direct = np.asarray(traj["sigma_spread"], dtype=float)[idx]
    sigma_ret_direct = np.asarray(traj["sigma_ret"], dtype=float)[idx]
    dS_direct = np.asarray(traj["dS"], dtype=float)[idx]

    err_spread = float(np.max(np.abs(sigma_spread_spec - sigma_spread_direct)))
    err_ret = float(np.max(np.abs(sigma_ret_spec - sigma_ret_direct)))
    err_dS = float(np.max(np.abs(dS_spec - dS_direct)))

    A2_t = np.linalg.norm(affinities, axis=1)
    J2_t = np.linalg.norm(J_spec, axis=1)
    A2_max = float(np.max(A2_t))
    current_rms = float(math.sqrt(np.trapezoid(J2_t**2, times) / T))
    sigma_ret_rms = float(math.sqrt(np.trapezoid(sigma_ret_spec**2, times) / T))
    sigma_spread_rms = float(math.sqrt(np.trapezoid(sigma_spread_spec**2, times) / T))

    # Exact state-specific Cauchy envelope on this window:
    # sigma_ret(t) <= k_B ||A(t)||_2 ||J(t)||_2.
    current_envelope_bound = float(A2_max * current_rms)
    alignment_fraction = (
        float(sigma_ret_rms / current_envelope_bound)
        if current_envelope_bound > 0.0
        else np.nan
    )

    finite_to_long_power = (
        float(current_rms**2 / spec["P_spectral"])
        if spec["P_spectral"] > 0.0
        else np.nan
    )

    row = {
        "N": int(N),
        "Q": int(Q),
        "mode": mode,
        "window": window_label,
        "t0": float(times[0]),
        "t1": float(times[-1]),
        "T": T,
        "n_samples": int(len(times)),
        "dE": int(spec["dE"]),
        "deff": float(spec["deff"]),
        "DG": int(spec["DG"]),
        "P_incoherent": float(spec["P_incoherent"]),
        "P_spectral": float(spec["P_spectral"]),
        "Gamma_spec": float(spec["Gamma_spec"]),
        "sqrt_Gamma_spec": float(spec["sqrt_Gamma_spec"]),
        "Gamma_over_DG": float(spec["Gamma_over_DG"]),
        "gap_participation_number": float(spec["gap_participation_number"]),
        "n_gap_50": int(spec["n_gap_50"]),
        "n_gap_90": int(spec["n_gap_90"]),
        "max_gap_power_share": float(spec["max_gap_power_share"]),
        "incoherent_current_rms_scale": float(spec["incoherent_current_rms_scale"]),
        "long_time_current_rms_fluct": float(spec["long_time_current_rms_fluct"]),
        "finite_window_current_rms": current_rms,
        "finite_to_long_current_power_ratio": finite_to_long_power,
        "A2_max": A2_max,
        "rms_sigma_ret": sigma_ret_rms,
        "rms_sigma_spread": sigma_spread_rms,
        "current_envelope_bound": current_envelope_bound,
        "return_alignment_fraction": alignment_fraction,
        "Jomega_norm": float(spec["Jomega_norm"]),
        "max_current_imag_error": float(current_imag_error),
        "max_sigma_ret_reconstruction_error": err_ret,
        "max_sigma_spread_reconstruction_error": err_spread,
        "max_dS_reconstruction_error": err_dS,
        "gap_conjugacy_error": float(spec["conjugacy_error"]),
        "stationary_current_error": float(spec["stationary_current_error"]),
    }
    return row, {
        "times": times,
        "J_spec": J_spec,
        "affinities": affinities,
        "sigma_ret_spec": sigma_ret_spec,
        "sigma_spread_spec": sigma_spread_spec,
        "dS_spec": dS_spec,
        "sigma_ret_direct": sigma_ret_direct,
        "sigma_spread_direct": sigma_spread_direct,
        "dS_direct": dS_direct,
    }


def _print_exact_spectral_current_row(row):
    print(
        f"{row['mode']:>11s} | {row['window']:<14s} | "
        f"Jt=[{row['t0']:.3f},{row['t1']:.3f}] | "
        f"dE={row['dE']} | D_G={row['DG']} | Gamma_spec={row['Gamma_spec']:.6f}"
    )
    print(
        f"    P_inc={row['P_incoherent']:.6e}, P_spec={row['P_spectral']:.6e}, "
        f"sqrt(Gamma_spec)={row['sqrt_Gamma_spec']:.6f}, "
        f"Gamma_spec/D_G={row['Gamma_over_DG']:.6f}"
    )
    print(
        f"    positive-gap power participation={row['gap_participation_number']:.6f}, "
        f"channels for 50%={row['n_gap_50']}, channels for 90%={row['n_gap_90']}, "
        f"largest-channel share={row['max_gap_power_share']:.6f}"
    )
    print(
        f"    long-time current RMS/J={row['long_time_current_rms_fluct']:.6e}, "
        f"finite-window current RMS/J={row['finite_window_current_rms']:.6e}, "
        f"RMS sigma_ret/(kB J)={row['rms_sigma_ret']:.6e}"
    )
    print(
        f"    A2_max={row['A2_max']:.6e}, state-specific current envelope/(kB J)="
        f"{row['current_envelope_bound']:.6e}, alignment={row['return_alignment_fraction']:.6f}"
    )
    print(
        f"    reconstruction max errors: sigma_ret={row['max_sigma_ret_reconstruction_error']:.3e}, "
        f"sigma_spread={row['max_sigma_spread_reconstruction_error']:.3e}, "
        f"dotS={row['max_dS_reconstruction_error']:.3e}"
    )


def spectral_current_N16_Q2_diagnostics():
    """Exact current-specific spectral diagnostics for mixing/free/return."""
    rows = []
    traces = {}
    edge_rows = []
    gap_rows = []

    for mode in SPECTRAL_CURRENT_MODES:
        traj = cached(
            f"spectral_current_v1_N16_Q2_{mode}_dt{SPECTRAL_CURRENT_N16_DT:.3f}",
            lambda mode=mode: class_trajectory(
                16,
                2,
                mode,
                4,
                "left_half",
                dt=SPECTRAL_CURRENT_N16_DT,
                tmax=max(30.0, SPECTRAL_CURRENT_WINDOW[1]),
                exact=True,
                compute_currents=True,
                block_steps=5,
            ),
        )

        row, trace = exact_spectral_current_window_diagnostics(
            16,
            2,
            mode,
            traj,
            window_label="common_8_11",
            requested_t0=SPECTRAL_CURRENT_WINDOW[0],
            requested_t1=SPECTRAL_CURRENT_WINDOW[1],
        )
        rows.append(row)
        traces[mode] = trace

        spec = _exact_spectral_current_data(16, 2, mode, Kcells=4, init_mode="left_half")
        for e, (a, b) in enumerate(zip(spec["pair_a"], spec["pair_b"])):
            edge_rows.append(
                {
                    "N": 16,
                    "Q": 2,
                    "mode": mode,
                    "edge_id": int(e),
                    "macro_a": int(a),
                    "macro_b": int(b),
                    "P_incoherent_edge": float(spec["P_incoherent_edge"][e]),
                    "P_spectral_edge": float(spec["P_spectral_edge"][e]),
                    "Gamma_edge": float(spec["Gamma_edge"][e]),
                }
            )

        for gid, (g, dg, pg) in enumerate(
            zip(spec["gap_centers"], spec["gap_degeneracies"], spec["gap_power"])
        ):
            gap_rows.append(
                {
                    "N": 16,
                    "Q": 2,
                    "mode": mode,
                    "gap_id": int(gid),
                    "gap_over_J": float(g),
                    "gap_degeneracy": int(dg),
                    "current_power": float(pg),
                }
            )

    _write_csv_rows(SPECTRAL_DIR / "spectral_current_N16_Q2.csv", rows)
    _write_csv_rows(SPECTRAL_DIR / "spectral_current_edges_N16_Q2.csv", edge_rows)
    _write_csv_rows(SPECTRAL_DIR / "spectral_current_gap_power_N16_Q2.csv", gap_rows)

    print("\n=== Exact state-resolved spectral-current diagnostics: N=16, Q=2 ===")
    for row in rows:
        _print_exact_spectral_current_row(row)

    return rows, traces


def spectral_current_size_sequence_diagnostics():
    """
    Exact gap-coherence scaling for Q=2 and all three Hamiltonian classes.

    This size sequence is deliberately static: it compares the state-resolved
    spectral organization itself and does not mix different time windows or
    probability-interior restrictions across sizes.
    """
    rows = []

    for N, Q in SPECTRAL_SIZE_SYSTEMS:
        for mode in SPECTRAL_CURRENT_MODES:
            spec = _exact_spectral_current_data(N, Q, mode, Kcells=4, init_mode="left_half")
            rows.append(
                {
                    "N": int(N),
                    "Q": int(Q),
                    "mode": mode,
                    "dE": int(spec["dE"]),
                    "deff": float(spec["deff"]),
                    "DG": int(spec["DG"]),
                    "P_incoherent": float(spec["P_incoherent"]),
                    "P_spectral": float(spec["P_spectral"]),
                    "Gamma_spec": float(spec["Gamma_spec"]),
                    "sqrt_Gamma_spec": float(spec["sqrt_Gamma_spec"]),
                    "Gamma_over_DG": float(spec["Gamma_over_DG"]),
                    "gap_participation_number": float(spec["gap_participation_number"]),
                    "n_gap_50": int(spec["n_gap_50"]),
                    "n_gap_90": int(spec["n_gap_90"]),
                    "max_gap_power_share": float(spec["max_gap_power_share"]),
                    "incoherent_current_rms_scale": float(spec["incoherent_current_rms_scale"]),
                    "long_time_current_rms_fluct": float(spec["long_time_current_rms_fluct"]),
                    "Jomega_norm": float(spec["Jomega_norm"]),
                    "gap_conjugacy_error": float(spec["conjugacy_error"]),
                    "stationary_current_error": float(spec["stationary_current_error"]),
                }
            )

    _write_csv_rows(SPECTRAL_DIR / "spectral_current_size_sequence_Q2.csv", rows)

    print("\n=== Exact spectral-current coherence size sequence: fixed Q=2 ===")
    for row in rows:
        print(
            f"N={row['N']:2d} | {row['mode']:>11s} | dE={row['dE']:3d} | "
            f"D_G={row['DG']:3d} | Gamma_spec={row['Gamma_spec']:.6f} | "
            f"Gamma/D_G={row['Gamma_over_DG']:.6f} | "
            f"N_gap^(2)={row['gap_participation_number']:.6f}"
        )

    return rows


# ============================================================
# Supplementary Figures S17-S20: exact current-specific spectra
# ============================================================


def make_supp_S17_spectral_coherence_N16():
    if not RUN_EXACT_SPECTRAL_CURRENT_TESTS:
        print("Skipping S17 because RUN_EXACT_SPECTRAL_CURRENT_TESTS=False")
        return

    rows, _ = spectral_current_N16_Q2_diagnostics()
    order = {m: i for i, m in enumerate(SPECTRAL_CURRENT_MODES)}
    rows = sorted(rows, key=lambda r: order[r["mode"]])

    x = np.arange(len(rows), dtype=float)
    gamma = np.array([r["Gamma_spec"] for r in rows], dtype=float)
    DG = np.array([r["DG"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.plot(
        x,
        gamma,
        color=RED,
        marker="o",
        linestyle="-",
        label=r"$\Gamma_{\rm spec}$",
    )
    ax.plot(
        x,
        DG,
        color=NAVY,
        marker="s",
        linestyle="--",
        label=r"$D_G$",
    )
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xticks(x, ["mixing", "free", "engineered\nreturn"])
    ax.set_ylabel("spectral coherence factor")
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=12,
    )
    savefig(fig, SUPP_DIR / "figS17_spectral_coherence_N16.pdf")


def make_supp_S18_spectral_coherence_size_sequence():
    if not RUN_EXACT_SPECTRAL_CURRENT_TESTS:
        print("Skipping S18 because RUN_EXACT_SPECTRAL_CURRENT_TESTS=False")
        return

    rows = spectral_current_size_sequence_diagnostics()
    styles = {
        "mixing": (RED, "o", "-"),
        "free": (BLUE, "s", "--"),
        "engineered": (NAVY, "^", "-."),
    }
    labels = {
        "mixing": "mixing",
        "free": "free nearest-neighbor",
        "engineered": "engineered return",
    }

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for mode in SPECTRAL_CURRENT_MODES:
        sub = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["N"])
        Ns = np.array([r["N"] for r in sub], dtype=int)
        gamma = np.array([r["Gamma_spec"] for r in sub], dtype=float)
        color, marker, linestyle = styles[mode]
        ax.plot(
            Ns,
            gamma,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=labels[mode],
        )
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\Gamma_{\rm spec}$")
    ax.set_xticks([N for N, _ in SPECTRAL_SIZE_SYSTEMS])
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=11,
    )
    savefig(fig, SUPP_DIR / "figS18_spectral_coherence_size_sequence.pdf")


def make_supp_S19_gap_resolved_current_power():
    if not RUN_EXACT_SPECTRAL_CURRENT_TESTS:
        print("Skipping S19 because RUN_EXACT_SPECTRAL_CURRENT_TESTS=False")
        return

    styles = {
        "mixing": (RED, "-"),
        "free": (BLUE, "--"),
        "engineered": (NAVY, "-."),
    }
    labels = {
        "mixing": "mixing",
        "free": "free nearest-neighbor",
        "engineered": "engineered return",
    }

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for mode in SPECTRAL_CURRENT_MODES:
        spec = _exact_spectral_current_data(16, 2, mode, Kcells=4, init_mode="left_half")
        positive = spec["gap_centers"] > 10.0 * spec["gap_tol"]
        power = spec["gap_power"][positive]
        power = power[power > 1e-30]
        power = np.sort(power)[::-1]
        cumulative = np.cumsum(power) / np.sum(power)
        rank = np.arange(1, len(power) + 1, dtype=int)
        color, linestyle = styles[mode]
        ax.semilogx(
            rank,
            cumulative,
            color=color,
            linestyle=linestyle,
            label=labels[mode],
        )
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1.0)
    ax.axhline(0.9, color=GRAY, linestyle=":", linewidth=1.0)
    ax.set_xlabel("strongest gap channels retained")
    ax.set_ylabel("cumulative current-power fraction")
    ax.set_ylim(0.0, 1.02)
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=11,
    )
    savefig(fig, SUPP_DIR / "figS19_gap_resolved_current_power.pdf")

def make_supp_S20_spectral_current_reconstruction():
    if not RUN_EXACT_SPECTRAL_CURRENT_TESTS:
        print("Skipping S20 because RUN_EXACT_SPECTRAL_CURRENT_TESTS=False")
        return

    _, traces = spectral_current_N16_Q2_diagnostics()
    styles = {
        "mixing": (RED, "o", "-"),
        "free": (BLUE, "s", "--"),
        "engineered": (NAVY, "^", "-."),
    }
    labels = {
        "mixing": "mixing",
        "free": "free nearest-neighbor",
        "engineered": "engineered return",
    }

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for mode in SPECTRAL_CURRENT_MODES:
        tr = traces[mode]
        color, marker, linestyle = styles[mode]
        ax.plot(
            tr["times"],
            tr["sigma_ret_direct"],
            color=color,
            linestyle=linestyle,
            label=labels[mode],
        )
        stride = max(1, len(tr["times"]) // 12)
        ax.plot(
            tr["times"][::stride],
            tr["sigma_ret_spec"][::stride],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=3.5,
        )
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\sigma_{\rm ret}/(k_{\rm B}J)$")
    ax.set_xlim(SPECTRAL_CURRENT_WINDOW)
    finish_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=11,
    )
    savefig(fig, SUPP_DIR / "figS20_spectral_current_reconstruction.pdf")


# ============================================================
# Common cached datasets
# ============================================================


def core_mixing():
    return cached(
        "core_mixing_N16_Q2_dt010",
        lambda: history_defect_dataset(16, 2, "mixing", dt=0.1, tmax=30.0),
    )


def core_engineered():
    return cached(
        "core_engineered_N16_Q2_dt010",
        lambda: history_defect_dataset(16, 2, "engineered", dt=0.1, tmax=30.0),
    )


def dilute_sequence():
    def build():
        data = {}
        data[(8, 2)] = class_trajectory(
            8,
            2,
            "mixing",
            4,
            "left_half",
            dt=0.3,
            tmax=30.0,
            exact=True,
            compute_currents=False,
        )
        data[(20, 3)] = class_trajectory(
            20,
            3,
            "mixing",
            4,
            "left_half",
            dt=0.3,
            tmax=30.0,
            exact=True,
            compute_currents=False,
        )
        data[(40, 4)] = class_trajectory(
            40,
            4,
            "mixing",
            4,
            "left_half",
            dt=0.3,
            tmax=30.0,
            exact=False,
            ntrace=4,
            seed=20260854,
            compute_currents=False,
            block_steps=4,
        )
        return data

    return cached("main_dilute_sequence", build)


# ============================================================
# Figure 1: mechanism and actual Lorenz-curve examples
# ============================================================


def make_fig1_mechanism():
    mix = core_mixing()
    ret = core_engineered()

    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.2))

    # (a) Fine state, coarse representative, unresolved microscopic structure.
    ax = axs[0, 0]
    ax.axis("off")

    box_rho = FancyBboxPatch(
        (0.05, 0.62),
        0.4,
        0.20,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=BLACK,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box_rho)
    ax.text(0.21, 0.72, r"exact state $\rho_t$", ha="center", va="center", transform=ax.transAxes)

    box_g = FancyBboxPatch(
        (0.56, 0.72),
        0.36,
        0.18,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=BLUE,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box_g)
    ax.text(0.74, 0.81, r"$\mathcal{G}[\rho_t]$", ha="center", va="center", transform=ax.transAxes)

    box_chi = FancyBboxPatch(
        (0.56, 0.43),
        0.4,
        0.18,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=RED,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box_chi)
    ax.text(0.74, 0.52, r"unresolved $\chi_t$", ha="center", va="center", transform=ax.transAxes)

    ax.add_patch(
        FancyArrowPatch(
            (0.38, 0.72),
            (0.54, 0.81),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.38, 0.70),
            (0.54, 0.52),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            transform=ax.transAxes,
        )
    )

    y0 = 0.08
    for j, width in enumerate([0.18, 0.27, 0.13, 0.22]):
        x0 = 0.05 + sum([0.18, 0.27, 0.13, 0.22][:j]) + 0.02 * j
        rect = Rectangle(
            (x0, y0),
            width,
            0.17,
            linewidth=1.0,
            edgecolor=BLACK,
            facecolor=[LIGHT_GRAY, MINT, "#EFEFEF", "#E8E8E8"][j],
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x0 + width / 2, y0 + 0.085, rf"$P_{{R_{j+1}}}$", ha="center", va="center", transform=ax.transAxes)

    ax.text(0.50, 0.31, r"thermodynamic macrospaces", ha="center", va="center", transform=ax.transAxes)
    panel_label(ax, "(a)")

    # (b) Exact finite-step coarse propagation.
    ax = axs[0, 1]
    ax.axis("off")
    ax.text(
        0.50,
        0.82,
        r"$p(t+\tau)=K^{(U)}(\tau)p(t)+r(t,\tau)$",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.61,
        r"$K^{(U)}\pi=\pi$",
        ha="center",
        va="center",
        color=BLUE,
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.41,
        r"$\left.\partial_\tau K^{(U)}(\tau)\right|_{\tau=0}=0$",
        ha="center",
        va="center",
        color=GREEN,
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.20,
        r"$\displaystyle\lim_{\tau\to0^+}\frac{r(t,\tau)}{\tau}=\dot p(t)$",
        ha="center",
        va="center",
        color=RED,
        transform=ax.transAxes,
    )

    panel_label(ax, "(b)")

    # (c) Actual Lorenz contraction during primary mixing relaxation.
    ax = axs[1, 0]
    idx_primary = int(np.argmin(np.abs(mix["Sigma"][:-1] - 0.55)))
    p0 = mix["p"][idx_primary]
    p1 = mix["p"][idx_primary + 1]
    x0, y0c = lorenz_curve(p0, mix["pi"])
    x1, y1c = lorenz_curve(p1, mix["pi"])
    ax.plot(x0, y0c, color=NAVY, label=r"$p(t)$")
    ax.plot(x1, y1c, color=BLUE, linestyle="--", label=r"$p(t+\tau)$")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L_p^\pi(x)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    finish_axes(ax)
    ax.legend(loc="lower right")
    panel_label(ax, "(c)")

    # (d) Actual Lorenz crossing during engineered recurrence.
    ax = axs[1, 1]
    idx_return = int(np.argmax(ret["Vpi"]))
    q0 = ret["p"][idx_return]
    q1 = ret["p"][idx_return + 1]
    xr0, yr0 = lorenz_curve(q0, ret["pi"])
    xr1, yr1 = lorenz_curve(q1, ret["pi"])
    ax.plot(xr0, yr0, color=DARK_RED, label=r"$p(t)$")
    ax.plot(xr1, yr1, color=SALMON, linestyle="--", label=r"$p(t+\tau)$")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L_p^\pi(x)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    finish_axes(ax)
    ax.legend(loc="lower right")
    panel_label(ax, "(d)")

    savefig(fig, FIG_DIR / "fig1_mechanism.pdf", h_pad=0.8, w_pad=0.8)


# ============================================================
# Figure 2: mixing versus engineered return
# ============================================================


def make_fig2_mixing_vs_return():
    mix = core_mixing()
    ret = core_engineered()

    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.2))

    ax = axs[0, 0]
    ax.plot(mix["times"], mix["Sigma"], color=RED, label="mixing")
    ax.plot(ret["times"], ret["Sigma"], color=NAVY, label="engineered return")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$[S_R(t)-S_R(0)]/[S_{\max}-S_R(0)]$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    ax.plot(mix["times"], mix["dS"], color=RED, label="mixing")
    ax.plot(ret["times"], ret["dS"], color=NAVY, label="engineered return")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\dot S_R/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(b)")

    ax = axs[1, 0]
    ax.semilogy(mix["times"][:-1], np.maximum(mix["Vpi"], 1e-16), color=RED, label="mixing")
    ax.semilogy(ret["times"][:-1], np.maximum(ret["Vpi"], 1e-16), color=NAVY, label="engineered return")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\mathcal{V}_\pi(t,\Delta t)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(c)")

    ax = axs[1, 1]
    ax.plot(mix["times"], mix["sigma_ret"], color=RED, label="mixing")
    ax.plot(ret["times"], ret["sigma_ret"], color=NAVY, label="engineered return")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\sigma_{\rm ret}/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(d)")

    savefig(fig, FIG_DIR / "fig2_mixing_vs_return.pdf", h_pad=0.8, w_pad=0.8)


# ============================================================
# Figure 3: hidden history and thermodynamic return
# ============================================================


def make_fig3_history_defect():
    mix = core_mixing()
    ret = core_engineered()
    step_t = mix["times"][:-1]

    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.2))

    ax = axs[0, 0]
    ax.semilogy(step_t, np.maximum(mix["eps_hist"], 1e-16), color=BLUE, label=r"$\epsilon_{\rm hist}$")
    ax.semilogy(step_t, np.maximum(mix["Vpi"], 1e-16), color=RED, label=r"$\mathcal{V}_\pi$")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"coarse-history defect")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    tau = float(step_t[1] - step_t[0])
    ax.plot(
        step_t,
        mix["dS_mix"] / tau,
        color=GREEN,
        label=r"$\Delta_\tau S_{\rm mix}/\tau$",
    )
    ax.plot(
        step_t,
        mix["dS_corr"] / tau,
        color=BLUE,
        label=r"$\Delta_\tau S_{\rm corr}/\tau$",
    )
    ax.plot(
        step_t,
        mix["dS_actual_step"] / tau,
        color=RED,
        label=r"$\Delta_\tau S_R/\tau$",
    )
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$[\Delta_\tau S/\tau]/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(b)")

    ax = axs[1, 0]
    ax.scatter(mix["eps_hist"], mix["Vpi"], s=24, color=RED, marker="o", label="mixing")
    ax.scatter(ret["eps_hist"], ret["Vpi"], s=24, color=NAVY, marker="s", label="engineered return")
    max_eps = max(np.max(mix["eps_hist"]), np.max(ret["eps_hist"]))
    ax.plot([0, max_eps], [0, max_eps], color=GRAY, linestyle=":", linewidth=1.2, label=r"$\mathcal{V}_\pi=\epsilon_{\rm hist}$")
    ax.set_xlabel(r"$\epsilon_{\rm hist}=\|r\|_1/2$")
    ax.set_ylabel(r"$\mathcal{V}_\pi$")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(c)")

    ax = axs[1, 1]
    idx_primary = int(np.argmin(np.abs(mix["Sigma"][:-1] - 0.55)))
    idx_return = int(np.argmax(ret["Vpi"]))

    xp0, yp0 = lorenz_curve(mix["p"][idx_primary], mix["pi"])
    xp1, yp1 = lorenz_curve(mix["p"][idx_primary + 1], mix["pi"])
    xr0, yr0 = lorenz_curve(ret["p"][idx_return], ret["pi"])
    xr1, yr1 = lorenz_curve(ret["p"][idx_return + 1], ret["pi"])

    ax.plot(xp0, yp0, color=BLUE, label=r"primary $p(t)$")
    ax.plot(xp1, yp1, color=BLUE, linestyle="--", label=r"primary $p(t+\tau)$")
    ax.plot(xr0, yr0, color=RED, label=r"return $p(t)$")
    ax.plot(xr1, yr1, color=RED, linestyle="--", label=r"return $p(t+\tau)$")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L_p^\pi(x)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(d)")

    savefig(fig, FIG_DIR / "fig3_history_defect.pdf", h_pad=0.8, w_pad=0.8)


# ============================================================
# Figure 4: free expansion and classical limit
# ============================================================


def exact_dilute_endpoint_sequences():
    Ns_fixed = np.array([8, 12, 16, 20, 24, 32, 40, 64, 100, 200], dtype=int)
    Q_fixed = 2
    rho_fixed = Q_fixed / (Ns_fixed / 2.0)
    ratio_fixed = np.array(
        [
            (log_binom(N, Q_fixed) - log_binom(N // 2, Q_fixed))
            / (Q_fixed * np.log(2.0))
            for N in Ns_fixed
        ]
    )

    Mvals = np.array([5, 10, 20, 40, 80, 160], dtype=int)
    Li = Mvals**2
    Q = Mvals
    rho_seq = Q / Li
    ratio_seq = np.array(
        [
            (log_binom(2 * int(L), int(q)) - log_binom(int(L), int(q)))
            / (q * np.log(2.0))
            for L, q in zip(Li, Q)
        ]
    )

    return rho_fixed, ratio_fixed, rho_seq, ratio_seq


def make_fig4_free_expansion():
    seq = dilute_sequence()
    systems = [(8, 2), (20, 3), (40, 4)]

    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.2))

    colors = [RED, BLUE, GREEN]

    ax = axs[0, 0]
    for color, (N, Q) in zip(colors, systems):
        r = seq[(N, Q)]
        ax.plot(r["times"], r["Sigma"], color=color, label=rf"$Q={Q},\,N={N}$")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$[S_R(t)-S_R(0)]/\Delta S_{\max}$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    for color, (N, Q) in zip(colors, systems):
        r = seq[(N, Q)]
        ax.plot(r["times"], r["dS"], color=color, label=rf"$Q={Q},\,N={N}$")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\dot S_R/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=10)
    panel_label(ax, "(b)")

    ax = axs[1, 0]
    rho_fixed, ratio_fixed, rho_seq, ratio_seq = exact_dilute_endpoint_sequences()
    ax.plot(rho_fixed, ratio_fixed, color=BLUE, marker="o", label=r"fixed $Q=2$")
    ax.plot(rho_seq, ratio_seq, color=GREEN, marker="s", label=r"$Q=\sqrt{L_i}$")

    rho_dyn = np.array([2 * Q / N for N, Q in systems], dtype=float)
    ratio_dyn = np.array(
        [
            (log_binom(N, Q) - log_binom(N // 2, Q)) / (Q * np.log(2.0))
            for N, Q in systems
        ]
    )
    ax.scatter(rho_dyn, ratio_dyn, color=RED, marker="^", s=36, label="dynamical sequence")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$\rho_i=Q/L_i$")
    ax.set_ylabel(r"$\Delta S_R/[Qk_{\rm B}\ln2]$")
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(c)")

    ax = axs[1, 1]
    min_rates = []
    for N, Q in systems:
        r = seq[(N, Q)]
        i10 = np.where(r["Sigma"] >= 0.10)[0][0]
        i90 = np.where((np.arange(len(r["Sigma"])) >= i10) & (r["Sigma"] >= 0.90))[0][0]
        min_rates.append(np.min(r["dS"][i10 : i90 + 1]))
    ax.plot([2, 3, 4], min_rates, color=NAVY, marker="o")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$\min_{\rm first}\dot S_R/(k_{\rm B}J)$")
    ax.set_xticks([2, 3, 4])
    finish_axes(ax)
    panel_label(ax, "(d)")

    savefig(fig, FIG_DIR / "fig4_free_expansion.pdf", h_pad=0.8, w_pad=1)


# ============================================================
# Supplementary Figure S1: fine histories versus thermodynamic class
# ============================================================


def make_supp_S1_history_class():
    R0 = (2, 2, 0, 0)

    hist = cached(
        "S1_individual_histories_N16_Q4_R2200",
        lambda: individual_history_trajectories(
            16, 4, "mixing", 4, R0, dt=0.1, tmax=30.0
        ),
    )
    cls = cached(
        "S1_class_N16_Q4_R2200",
        lambda: class_trajectory(
            16,
            4,
            "mixing",
            4,
            "fixed_record",
            R0=R0,
            dt=0.1,
            tmax=30.0,
            exact=True,
            compute_currents=False,
        ),
    )

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axs[0]
    for h in range(hist["Sigma_hist"].shape[1]):
        ax.plot(hist["times"], hist["Sigma_hist"][:, h], color=LIGHT_GRAY, linewidth=0.55, alpha=0.75)
    ax.plot(cls["times"], cls["Sigma"], color=RED, linewidth=1.8, label="thermodynamic class")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$[S_R(t)-S_R(0)]/[S_{\max}-S_R(0)]$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(a)")

    i10 = np.where(cls["Sigma"] >= 0.10)[0][0]
    i90 = np.where((np.arange(len(cls["Sigma"])) >= i10) & (cls["Sigma"] >= 0.90))[0][0]
    mins = np.min(hist["dS_hist"][i10 : i90 + 1], axis=0)
    class_min = np.min(cls["dS"][i10 : i90 + 1])

    ax = axs[1]
    ax.hist(mins, bins=12, color=BLUE, alpha=0.8, edgecolor=BLACK, linewidth=0.6)
    ax.axvline(class_min, color=RED, linewidth=1.5, label="class-level minimum")
    ax.axvline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$\min_{\rm first}\dot S_{R|\gamma}/(k_{\rm B}J)$")
    ax.set_ylabel("count")
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(b)")

    savefig(fig, SUPP_DIR / "figS1_history_class.pdf", w_pad=0.8)


# ============================================================
# Supplementary Figure S2: record-resolution robustness
# ============================================================


def return_burden(traj):
    i10 = np.where(traj["Sigma"] >= 0.10)[0][0]
    i90 = np.where((np.arange(len(traj["Sigma"])) >= i10) & (traj["Sigma"] >= 0.90))[0][0]
    t = traj["times"][i10 : i90 + 1]
    ret = traj["sigma_ret"][i10 : i90 + 1]
    spread = traj["sigma_spread"][i10 : i90 + 1]
    return float(np.trapz(ret, t) / np.trapz(spread, t))


def make_supp_S2_record_resolution():
    two = cached(
        "S2_two_cell_N16_Q2",
        lambda: class_trajectory(
            16,
            2,
            "mixing",
            2,
            "left_half",
            dt=0.1,
            tmax=30.0,
            exact=True,
            compute_currents=True,
        ),
    )
    four = core_mixing()

    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.2))

    ax = axs[0, 0]
    ax.plot(two["times"], two["Sigma"], color=BLUE, label="two-cell record")
    ax.plot(four["times"], four["Sigma"], color=RED, label="four-cell record")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"normalized $S_R$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    ax.plot(two["times"], two["dS"], color=BLUE, label="two-cell record")
    ax.plot(four["times"], four["dS"], color=RED, label="four-cell record")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\dot S_R/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(b)")

    ax = axs[1, 0]
    ax.plot(two["times"], two["sigma_ret"], color=BLUE, label="two-cell record")
    ax.plot(four["times"], four["sigma_ret"], color=RED, label="four-cell record")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\sigma_{\rm ret}/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend()
    panel_label(ax, "(c)")

    systems = [(8, 2), (12, 3), (16, 4), (20, 5)]
    burden2, burden4 = [], []
    Ns = []
    for N, Q in systems:
        exact = N <= 16
        ntrace = 12 if N == 20 else 8
        tr2 = cached(
            f"S2_burden_K2_N{N}_Q{Q}",
            lambda N=N, Q=Q, exact=exact, ntrace=ntrace: class_trajectory(
                N,
                Q,
                "mixing",
                2,
                "left_half",
                dt=0.15,
                tmax=30.0,
                exact=exact,
                ntrace=ntrace,
                seed=20260810 + N + Q,
                compute_currents=True,
            ),
        )
        tr4 = cached(
            f"S2_burden_K4_N{N}_Q{Q}",
            lambda N=N, Q=Q, exact=exact, ntrace=ntrace: class_trajectory(
                N,
                Q,
                "mixing",
                4,
                "left_half",
                dt=0.15,
                tmax=30.0,
                exact=exact,
                ntrace=ntrace,
                seed=20260810 + N + Q,
                compute_currents=True,
            ),
        )
        Ns.append(N)
        burden2.append(return_burden(tr2))
        burden4.append(return_burden(tr4))

    ax = axs[1, 1]
    ax.semilogy(Ns, burden2, color=BLUE, marker="o", label="two-cell record")
    ax.semilogy(Ns, burden4, color=RED, marker="s", label="four-cell record")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\int\sigma_{\rm ret}dt/\int\sigma_{\rm spread}dt$")
    ax.set_xticks(Ns)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(d)")

    savefig(fig, SUPP_DIR / "figS2_record_resolution.pdf", h_pad=0.8, w_pad=0.8)


# ============================================================
# Supplementary Figure S3: finite-size return suppression
# ============================================================


def make_supp_S3_finite_size_return():
    systems = [(8, 2), (12, 3), (16, 4), (20, 5)]
    Ns, depths, rms_ret = [], [], []

    for N, Q in systems:
        exact = N <= 16
        ntrace = 12 if N == 20 else 8
        tr = cached(
            f"S3_return_N{N}_Q{Q}",
            lambda N=N, Q=Q, exact=exact, ntrace=ntrace: class_trajectory(
                N,
                Q,
                "mixing",
                4,
                "left_half",
                dt=0.15,
                tmax=30.0,
                exact=exact,
                ntrace=ntrace,
                seed=20260860 + N + Q,
                compute_currents=True,
            ),
        )

        i90s = np.where(tr["Sigma"] >= 0.90)[0]
        if len(i90s) == 0:
            raise RuntimeError(f"N={N},Q={Q} did not reach 90% entropy in the chosen window.")
        i90 = i90s[0]
        post = tr["Sigma"][i90:]
        running_peak = np.maximum.accumulate(post)
        depth = float(np.max(running_peak - post))

        late = tr["times"] >= max(tr["times"][i90], 8.0)
        rms = float(np.sqrt(np.mean(tr["sigma_ret"][late] ** 2)))

        Ns.append(N)
        depths.append(depth)
        rms_ret.append(rms)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axs[0]
    ax.semilogy(Ns, np.maximum(depths, 1e-12), color=RED, marker="o")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"maximum normalized entropy return")
    ax.set_xticks(Ns)
    finish_axes(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.semilogy(Ns, rms_ret, color=BLUE, marker="s")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\sqrt{\langle\sigma_{\rm ret}^2\rangle}/(k_{\rm B}J)$")
    ax.set_xticks(Ns)
    finish_axes(ax)
    panel_label(ax, "(b)")

    savefig(fig, SUPP_DIR / "figS3_finite_size_return.pdf", w_pad=0.8)


# ============================================================
# Supplementary Figure S4: exact macrocurrent decomposition
# ============================================================


def make_supp_S4_macrocurrent():
    mix = core_mixing()
    ret = core_engineered()

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axs[0]
    ax.plot(mix["times"], mix["sigma_spread"], color=GREEN, label=r"$\sigma_{\rm spread}$")
    ax.plot(mix["times"], mix["sigma_ret"], color=RED, label=r"$\sigma_{\rm ret}$")
    ax.plot(mix["times"], mix["dS"], color=BLACK, linewidth=1.2, label=r"$\dot S_R$")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"rate$/ (k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.plot(ret["times"], ret["sigma_spread"], color=GREEN, label=r"$\sigma_{\rm spread}$")
    ax.plot(ret["times"], ret["sigma_ret"], color=RED, label=r"$\sigma_{\rm ret}$")
    ax.plot(ret["times"], ret["dS"], color=BLACK, linewidth=1.2, label=r"$\dot S_R$")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"rate$/ (k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(b)")

    savefig(fig, SUPP_DIR / "figS4_macrocurrent_decomposition.pdf", w_pad=0.8)


# ============================================================
# Supplementary Figure S5: numerical verification of K^(U)
# ============================================================


def make_supp_S5_K_verification():
    taus = np.array([0.02, 0.05, 0.10, 0.20, 0.50, 1.00])
    stochastic_errors = []
    pi_errors = []
    macro = get_macro_data(16, 2, 4)
    pi = macro["Omega"] / macro["Omega"].sum()

    for tau in taus:
        Kmat = build_coarse_map_K(16, 2, "mixing", 4, float(tau))
        stochastic_errors.append(np.max(np.abs(Kmat.sum(axis=0) - 1.0)))
        pi_errors.append(np.max(np.abs(Kmat @ pi - pi)))

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.semilogy(taus, stochastic_errors, color=BLUE, marker="o", label=r"$\max_{R'}|\sum_R K_{RR'}-1|$")
    ax.semilogy(taus, pi_errors, color=RED, marker="s", label=r"$\|K\pi-\pi\|_\infty$")
    ax.set_xlabel(r"$J\tau$")
    ax.set_ylabel("numerical error")
    finish_axes(ax)
    ax.legend(fontsize=12)
    savefig(fig, SUPP_DIR / "figS5_K_verification.pdf")


# ============================================================
# Supplementary Figure S6: V_pi <= epsilon_hist
# ============================================================


def make_supp_S6_bound_scatter():
    mix = core_mixing()
    ret = core_engineered()

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.scatter(mix["eps_hist"], mix["Vpi"], color=RED, s=24, marker="o", label="mixing")
    ax.scatter(ret["eps_hist"], ret["Vpi"], color=NAVY, s=24, marker="s", label="engineered return")
    m = max(np.max(mix["eps_hist"]), np.max(ret["eps_hist"]))
    ax.plot([0, m], [0, m], color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$\epsilon_{\rm hist}=\|r\|_1/2$")
    ax.set_ylabel(r"$\mathcal{V}_\pi$")
    ax.set_xlim(0, m * 1.03)
    ax.set_ylim(0, m * 1.03)
    finish_axes(ax)
    ax.legend()
    savefig(fig, SUPP_DIR / "figS6_majorization_bound.pdf")


# ============================================================
# Supplementary Figure S7: time-step convergence of majorization
# ============================================================


def make_supp_S7_majorization_dt_convergence():
    dts = [0.10, 0.05, 0.02]
    colors = [RED, BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(4.3, 3.0))

    for dt, color in zip(dts, colors):
        ds = cached(
            f"S7_majorization_dt{dt:.2f}",
            lambda dt=dt: history_defect_dataset(16, 2, "mixing", dt=dt, tmax=4.8),
        )
        i10 = np.where(ds["Sigma"] >= 0.10)[0][0]
        i90 = np.where((np.arange(len(ds["Sigma"])) >= i10) & (ds["Sigma"] >= 0.90))[0][0]
        t = ds["times"][:-1][i10:i90]
        margin = ds["margins"][i10:i90]
        ax.plot(t, margin, color=color, label=rf"$\Delta(Jt)={dt:.2f}$")

    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\min_x[L_{p(t)}^\pi(x)-L_{p(t+\Delta t)}^\pi(x)]$")
    finish_axes(ax)
    ax.legend(fontsize=12)
    savefig(fig, SUPP_DIR / "figS7_majorization_dt_convergence.pdf")


# ============================================================
# Supplementary Figure S8: trace-estimator convergence
# ============================================================


def make_supp_S8_trace_convergence():
    if not RUN_VERY_HEAVY_SUPPLEMENT:
        print("Skipping S8 because RUN_VERY_HEAVY_SUPPLEMENT=False")
        return

    counts = [1, 2, 4, 8]
    seeds = [20260810, 20260811, 20260812]

    min_rate_mean, min_rate_std = [], []
    max_sigma_mean, max_sigma_std = [], []

    for ntrace in counts:
        rates = []
        max_sigmas = []
        for seed in seeds:
            ds = cached(
                f"S8_trace_N40_Q4_n{ntrace}_seed{seed}",
                lambda ntrace=ntrace, seed=seed: class_trajectory(
                    40,
                    4,
                    "mixing",
                    4,
                    "left_half",
                    dt=0.3,
                    tmax=30.0,
                    exact=False,
                    ntrace=ntrace,
                    seed=seed,
                    compute_currents=False,
                    block_steps=4,
                ),
            )
            i10 = np.where(ds["Sigma"] >= 0.10)[0][0]
            i90 = np.where((np.arange(len(ds["Sigma"])) >= i10) & (ds["Sigma"] >= 0.90))[0][0]
            rates.append(np.min(ds["dS"][i10 : i90 + 1]))
            max_sigmas.append(np.max(ds["Sigma"]))

        min_rate_mean.append(np.mean(rates))
        min_rate_std.append(np.std(rates, ddof=1))
        max_sigma_mean.append(np.mean(max_sigmas))
        max_sigma_std.append(np.std(max_sigmas, ddof=1))

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axs[0]
    ax.errorbar(counts, min_rate_mean, yerr=min_rate_std, color=RED, marker="o", capsize=3)
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel("number of trace vectors")
    ax.set_ylabel(r"$\min_{\rm first}\dot S_R/(k_{\rm B}J)$")
    ax.set_xticks(counts)
    finish_axes(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.errorbar(counts, max_sigma_mean, yerr=max_sigma_std, color=BLUE, marker="s", capsize=3)
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel("number of trace vectors")
    ax.set_ylabel(r"$\max_t\Sigma_R(t)$")
    ax.set_xticks(counts)
    finish_axes(ax)
    panel_label(ax, "(b)")

    savefig(fig, SUPP_DIR / "figS8_trace_convergence.pdf", w_pad=0.8)


# ============================================================
# Supplementary Figure S9: temporal-grid convergence
# ============================================================


def make_supp_S9_time_grid_convergence():
    if not RUN_VERY_HEAVY_SUPPLEMENT:
        print("Skipping S9 because RUN_VERY_HEAVY_SUPPLEMENT=False")
        return

    dts = [0.30, 0.15, 0.075]
    colors = [RED, BLUE, GREEN]
    datasets = []

    for dt in dts:
        ds = cached(
            f"S9_dt_N40_Q4_{dt:.3f}",
            lambda dt=dt: class_trajectory(
                40,
                4,
                "mixing",
                4,
                "left_half",
                dt=dt,
                tmax=30,
                exact=False,
                ntrace=2,
                seed=20261054,
                compute_currents=False,
                block_steps=4,
            ),
        )
        datasets.append(ds)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axs[0]
    for dt, color, ds in zip(dts, colors, datasets):
        ax.plot(ds["times"], ds["dS"], color=color, label=rf"$\Delta(Jt)={dt:.3f}$")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\dot S_R/(k_{\rm B}J)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(a)")

    mins = []
    for ds in datasets:
        i10 = np.where(ds["Sigma"] >= 0.10)[0][0]
        i90 = np.where((np.arange(len(ds["Sigma"])) >= i10) & (ds["Sigma"] >= 0.90))[0][0]
        mins.append(np.min(ds["dS"][i10 : i90 + 1]))

    ax = axs[1]
    ax.plot(dts, mins, color=NAVY, marker="o")
    ax.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$\Delta(Jt)$")
    ax.set_ylabel(r"$\min_{\rm first}\dot S_R/(k_{\rm B}J)$")
    finish_axes(ax)
    panel_label(ax, "(b)")

    savefig(fig, SUPP_DIR / "figS9_time_grid_convergence.pdf", w_pad=0.8)


# ============================================================
# Supplementary Figure S10: exact combinatorial dilute limit
# ============================================================


def make_supp_S10_combinatorial_limit():
    rho_fixed, ratio_fixed, rho_seq, ratio_seq = exact_dilute_endpoint_sequences()

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.plot(rho_fixed, ratio_fixed, color=BLUE, marker="o", label=r"fixed $Q=2$")
    ax.plot(rho_seq, ratio_seq, color=GREEN, marker="s", label=r"$Q=\sqrt{L_i}$")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Q/L_i$")
    ax.set_ylabel(r"$\Delta S_R/[Qk_{\rm B}\ln2]$")
    finish_axes(ax)
    ax.legend()
    savefig(fig, SUPP_DIR / "figS10_combinatorial_dilute_limit.pdf")


# ============================================================
# Supplementary Figure S11: general volume-ratio test
# ============================================================


def make_supp_S11_general_volume_ratio():
    lambdas = np.array([1.25, 1.5, 2.0, 3.0, 4.0])
    densities = [0.05, 0.02, 0.01]
    Li = 20000

    fig, ax = plt.subplots(figsize=(4.3, 3.0))

    for color, rho in zip([RED, BLUE, GREEN], densities):
        Q = max(1, int(round(rho * Li)))
        x = np.log(lambdas)
        y = []
        for lam in lambdas:
            Lf = int(round(lam * Li))
            delta = log_binom(Lf, Q) - log_binom(Li, Q)
            y.append(delta / Q)
        ax.plot(x, y, color=color, marker="o", label=rf"$Q/L_i={Q/Li:.2f}$")

    xline = np.linspace(np.log(lambdas.min()), np.log(lambdas.max()), 200)
    ax.plot(xline, xline, color=GRAY, linestyle=":", linewidth=1.2, label=r"$y=x$")
    ax.set_xlabel(r"$\ln(L_f/L_i)$")
    ax.set_ylabel(r"$\Delta S_R/(Qk_{\rm B})$")
    finish_axes(ax)
    ax.legend(fontsize=12)
    savefig(fig, SUPP_DIR / "figS11_general_volume_ratio.pdf")


# ============================================================
# Supplementary Figure S12: free/integrable intermediate control
# ============================================================


def make_supp_S12_integrable_control():
    modes = ["mixing", "free", "engineered"]
    colors = [RED, BLUE, NAVY]
    labels = ["mixing", "free nearest-neighbor", "engineered return"]
    data = {}

    for mode in modes:
        data[mode] = cached(
            f"S12_{mode}_N16_Q2",
            lambda mode=mode: history_defect_dataset(16, 2, mode, dt=0.1, tmax=30.0),
        )

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axs[0]
    for mode, color, label in zip(modes, colors, labels):
        ds = data[mode]
        ax.plot(ds["times"], ds["Sigma"], color=color, label=label)
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"normalized $S_R$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(a)")

    ax = axs[1]
    for mode, color, label in zip(modes, colors, labels):
        ds = data[mode]
        ax.semilogy(ds["times"][:-1], np.maximum(ds["Vpi"], 1e-16), color=color, label=label)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$\mathcal{V}_\pi(t,\Delta t)$")
    ax.set_xlim(0, 30)
    finish_axes(ax)
    ax.legend(fontsize=12)
    panel_label(ax, "(b)")

    savefig(fig, SUPP_DIR / "figS12_integrable_control.pdf", w_pad=0.8)


# ============================================================
# Supplementary Figure S13: microscopic recurrence fidelity
# ============================================================


def pure_fidelity_trajectory(N, Q, mode, occupied_sites, dt=0.1, tmax=30.0):
    H, basis, index = get_model(N, Q, mode)
    b0 = 0
    for site0 in occupied_sites:
        b0 |= 1 << site0
    psi = np.zeros(len(basis), dtype=np.complex128)
    psi[index[b0]] = 1.0
    psi0 = psi.copy()

    nsteps = int(round(tmax / dt))
    times = np.arange(nsteps + 1) * dt
    fidelity = np.empty(nsteps + 1)

    for k in range(nsteps + 1):
        fidelity[k] = abs(np.vdot(psi0, psi)) ** 2
        if k < nsteps:
            psi = expm_multiply((-1j * dt) * H, psi)

    return times, fidelity


def make_supp_S13_fidelity():
    if not MAKE_OPTIONAL_S13:
        print("Skipping optional S13 because MAKE_OPTIONAL_S13=False")
        return

    modes = ["mixing", "free", "engineered"]
    colors = [RED, BLUE, NAVY]
    labels = ["mixing", "free nearest-neighbor", "engineered return"]

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for mode, color, label in zip(modes, colors, labels):
        t, F = pure_fidelity_trajectory(16, 2, mode, occupied_sites=[0, 2], dt=0.1, tmax=30.0)
        ax.semilogy(t, np.maximum(F, 1e-12), color=color, label=label)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$|\langle\psi(0)|\psi(t)\rangle|^2$")
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-12, 1.1)
    finish_axes(ax)
    ax.legend(fontsize=12)
    savefig(fig, SUPP_DIR / "figS13_microscopic_fidelity.pdf")


# ============================================================
# Run everything
# ============================================================


def make_all_main_figures():
    make_fig1_mechanism()
    make_fig2_mixing_vs_return()
    make_fig3_history_defect()
    make_fig4_free_expansion()


def make_all_supplementary_figures():
    make_supp_S1_history_class()
    make_supp_S2_record_resolution()
    make_supp_S3_finite_size_return()
    make_supp_S4_macrocurrent()
    make_supp_S5_K_verification()
    make_supp_S6_bound_scatter()
    make_supp_S7_majorization_dt_convergence()
    make_supp_S8_trace_convergence()
    make_supp_S9_time_grid_convergence()
    make_supp_S10_combinatorial_limit()
    make_supp_S11_general_volume_ratio()
    make_supp_S12_integrable_control()
    make_supp_S13_fidelity()
    make_supp_S14_spectral_theorem_test()
    make_supp_S15_spectral_size_sequence()
    make_supp_S16_gap_concentration_vs_return()
    make_supp_S17_spectral_coherence_N16()
    make_supp_S18_spectral_coherence_size_sequence()
    make_supp_S19_gap_resolved_current_power()
    make_supp_S20_spectral_current_reconstruction()


if __name__ == "__main__":
    make_all_main_figures()
    make_all_supplementary_figures()