"""No-regularization ADMM benchmark for stochastic DER aggregation coordination.

This module mirrors the continuous centralized model in ``0527.ipynb`` and
keeps the centralized objective regularization term for the proposed/current
aggregation benchmark. It removes that original objective regularization only
from the DER-local ADMM subproblems. The ADMM augmented-Lagrangian penalty is
kept because it is an algorithmic coordination term rather than an economic
contribution.

The module adds a DER-wise consensus ADMM decomposition for the
internal-market balance constraint

    sum_i dp[i,t,s] == sum_i dm[i,t,s].

This folder is an ablation baseline: it tests how the ADMM comparison behaves
when the original economic/contribution regularization is excluded from ADMM
while the proposed centralized benchmark keeps it.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


INITIAL_SUBMISSION_FILES = [
    "1033.csv",
    "1818.csv",
    "2502.csv",
    "2503.csv",
    "2634.csv",
    "2698.csv",
    "2816.csv",
    "545.csv",
    "665.csv",
    "690.csv",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def generation_files(root: Path | None = None) -> list[str]:
    """Return generation files with the original ten-DER set first."""

    root = repo_root() if root is None else Path(root)
    data_dir = root / "data" / "generation"
    all_files = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )
    missing_initial = [name for name in INITIAL_SUBMISSION_FILES if name not in all_files]
    if missing_initial:
        raise ValueError(f"Missing initial generation files: {missing_initial}")
    rest = [name for name in all_files if name not in INITIAL_SUBMISSION_FILES]
    return INITIAL_SUBMISSION_FILES + rest


def load_generation_data(
    include_files: list[str],
    date_filter: str = "2022-07-18",
    root: Path | None = None,
) -> tuple[np.ndarray, int, int]:
    """Load hourly generation data for selected DER CSV files."""

    root = repo_root() if root is None else Path(root)
    data_dir = root / "data" / "generation"
    generation_data = np.zeros((len(include_files), 24))

    for idx, file_name in enumerate(include_files):
        df = pd.read_csv(data_dir / file_name)
        df.columns = df.columns.str.strip()

        date_col = "Date"
        hour_col = "Hour (Eastern Time, Daylight-Adjusted)"
        gen_col = "Electricity Generated"
        required = {date_col, hour_col, gen_col}
        if not required.issubset(df.columns):
            missing = ", ".join(sorted(required - set(df.columns)))
            raise ValueError(f"{file_name} is missing required columns: {missing}")

        if date_filter:
            df = df[df[date_col] == date_filter]
            if df.empty:
                raise ValueError(f"{file_name} has no rows for date {date_filter}")

        df = df[pd.Series(df[hour_col]).astype(str).str.match(r"^\d+$")]
        df["Time"] = pd.Series(df[hour_col]).astype(int)
        df = df[pd.Series(df["Time"]).between(0, 23)]

        for t in range(24):
            if t in pd.Series(df["Time"]).values:
                generation_data[idx, t] = pd.Series(
                    df[df["Time"] == t][gen_col]
                ).values[0]

    return generation_data, len(include_files), 24


def generate_randomized_generation(
    generation_data: np.ndarray,
    scenarios: int,
    randomness_level: str,
    random_seed: int,
) -> np.ndarray:
    ranges = {
        "low": (0.95, 1.05),
        "medium": (0.85, 1.15),
        "high": (0.2, 1.8),
    }
    if randomness_level not in ranges:
        raise ValueError(f"Unknown generation randomness level: {randomness_level}")
    low, high = ranges[randomness_level]
    rng = np.random.default_rng(random_seed)
    noise = rng.uniform(low, high, size=(*generation_data.shape, scenarios))
    return np.expand_dims(generation_data, axis=-1) * noise


def generate_rt_scenarios(
    scenarios: int,
    random_seed: int,
    root: Path | None = None,
) -> np.ndarray:
    root = repo_root() if root is None else Path(root)
    price_path = root / "data" / "price" / "20220718rt.csv"
    ny_rt = pd.read_csv(price_path)
    ny_rt["Time Stamp"] = pd.to_datetime(ny_rt["Time Stamp"])
    nyc_rt = ny_rt[ny_rt["Name"] == "MHK VL"].copy()

    start_of_day = nyc_rt["Time Stamp"].min().floor("D")
    end_of_day = start_of_day + pd.Timedelta(hours=23)
    nyc_rt = nyc_rt[
        (nyc_rt["Time Stamp"] >= start_of_day)
        & (nyc_rt["Time Stamp"] <= end_of_day)
    ]
    nyc_rt["Hour"] = pd.Series(nyc_rt["Time Stamp"]).dt.floor("h")
    price_hourly = (
        nyc_rt.groupby("Hour")["LBMP ($/MWHr)"]
        .mean()
        .reset_index()["LBMP ($/MWHr)"]
        .to_numpy()
    )

    rng = np.random.default_rng(random_seed)
    noise = rng.uniform(0.4, 1.6, size=(len(price_hourly), scenarios))
    return price_hourly[:, None] * noise


def load_price_data(
    p_rt: np.ndarray,
    scale_da: float = 1.5,
    scale_penalty: float = 2.0,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    root = repo_root() if root is None else Path(root)
    price_path = root / "data" / "price" / "20220718da.csv"
    ny_da = pd.read_csv(price_path)
    ny_da["Time Stamp"] = pd.to_datetime(ny_da["Time Stamp"])
    ny_da["Hour"] = pd.Series(ny_da["Time Stamp"]).dt.hour
    zone = ny_da[ny_da["Name"] == "MHK VL"]
    p_da = np.array(zone["LBMP ($/MWHr)"].astype(float)) * float(scale_da)
    p_pn = np.maximum(p_da[:, None], p_rt) * float(scale_penalty)
    return p_da, p_pn


def build_problem_data(
    n_der: int = 10,
    scenarios: int = 100,
    seed: int = 1,
    level: str = "high",
    date: str = "2022-07-18",
    ineff_batt: float = 0.95,
    ineff_ext_value: float = 0.99,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build one stochastic DER coordination instance."""

    root = repo_root() if root is None else Path(root)
    files = generation_files(root)
    if n_der > len(files):
        raise ValueError(f"Requested {n_der} DERs, but only {len(files)} files exist.")

    selected_files = files[:n_der]
    generation_data, i_count, t_count = load_generation_data(selected_files, date, root)
    r = generate_randomized_generation(generation_data, scenarios, level, seed)
    p_rt = generate_rt_scenarios(scenarios, seed, root)
    p_da, p_pn = load_price_data(p_rt, root=root)

    peak_generations = np.mean(generation_data, axis=1)
    k = (peak_generations * 2 // 100) * 100
    k0 = np.zeros(i_count)
    crate = k / 4
    drate = k / 4

    return {
        "R": r,
        "P_DA": p_da,
        "P_RT": p_rt,
        "P_PN": p_pn,
        "K": k,
        "K0": k0,
        "CRATE": crate,
        "DRATE": drate,
        "INEFF_BATT": ineff_batt,
        "INEFF_EXT": np.full(i_count, ineff_ext_value),
        "I": i_count,
        "T": t_count,
        "S": scenarios,
        "selected_files": selected_files,
        "date": date,
        "seed": seed,
        "level": level,
    }


def set_common_params(
    model: gp.Model,
    output_flag: int = 0,
    time_limit: float | None = None,
    method: int | None = None,
) -> None:
    model.setParam("OutputFlag", output_flag)
    model.setParam("MIPGap", 1e-6)
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if method is not None:
        model.setParam("Method", method)


def _add_terms_chunked(
    expr: gp.QuadExpr,
    coeffs: list[float],
    vars1: list[gp.Var],
    vars2: list[gp.Var] | None = None,
    chunk_size: int = 50_000,
) -> None:
    for start in range(0, len(coeffs), chunk_size):
        end = start + chunk_size
        if vars2 is None:
            expr.addTerms(coeffs[start:end], vars1[start:end])
        else:
            expr.addTerms(coeffs[start:end], vars1[start:end], vars2[start:end])


def expected_market_profit(
    p_da: np.ndarray,
    p_rt: np.ndarray,
    p_pn: np.ndarray,
    x: np.ndarray,
    yp: np.ndarray,
    ym: np.ndarray,
) -> float:
    """Expected profit in the same units as the notebook objective."""

    da = float(np.sum(p_da[None, :] * x))
    rt = float(np.mean(np.sum(p_rt[None, :, :] * yp, axis=(0, 1))))
    penalty = float(np.mean(np.sum(p_pn[None, :, :] * ym, axis=(0, 1))))
    return da + rt - penalty


def regularization_value(dp: np.ndarray, dm: np.ndarray, eps: float) -> float:
    s_count = dp.shape[2]
    return float(eps * np.sum(dp * dp + dm * dm) / s_count)


def internal_balance_residual(dp: np.ndarray, dm: np.ndarray) -> np.ndarray:
    return np.sum(dp - dm, axis=0)


def solve_centralized(
    data: dict[str, Any],
    eps: float = 1e-8,
    time_limit: float | None = 1200.0,
    output_flag: int = 0,
) -> dict[str, Any]:
    """Solve the centralized stochastic coordination QP from ``0527.ipynb``."""

    r = data["R"]
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    k = data["K"]
    k0 = data["K0"]
    crate = data["CRATE"]
    drate = data["DRATE"]
    ineff_batt = data["INEFF_BATT"]
    ineff_ext = data["INEFF_EXT"]

    i_count, t_count, s_count = r.shape
    build_start = time.perf_counter()
    model = gp.Model("centralized_stochastic_coordination")
    set_common_params(model, output_flag, time_limit)

    x = model.addVars(i_count, t_count, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="yp")
    ym = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="ym")
    dp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dp")
    dm = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dm")
    z = model.addVars(i_count, t_count + 1, s_count, vtype=GRB.CONTINUOUS, lb=0, name="z")
    zc = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zc")
    zd = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zd")

    obj = gp.QuadExpr()
    lin_coeffs: list[float] = []
    lin_vars: list[gp.Var] = []
    for i in range(i_count):
        for t in range(t_count):
            lin_coeffs.append(float(p_da[t]))
            lin_vars.append(x[i, t])
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        lin_coeffs.extend([float(p_rt[t, s] / s_count), float(-p_pn[t, s] / s_count)])
        lin_vars.extend([yp[i, t, s], ym[i, t, s]])
    _add_terms_chunked(obj, lin_coeffs, lin_vars)

    quad_coeffs: list[float] = []
    quad_vars: list[gp.Var] = []
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        coeff = float(-eps / s_count)
        quad_coeffs.extend([coeff, coeff])
        quad_vars.extend([dp[i, t, s], dm[i, t, s]])
    _add_terms_chunked(obj, quad_coeffs, quad_vars, quad_vars)
    model.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        eta = ineff_ext[i]
        model.addConstr(
            r[i, t, s] - (1 / eta) * x[i, t]
            == (1 / eta) * yp[i, t, s]
            - eta * ym[i, t, s]
            + (1 / eta) * dp[i, t, s]
            - eta * dm[i, t, s]
            + zc[i, t, s]
            - zd[i, t, s]
        )
        model.addConstr(zd[i, t, s] / ineff_batt <= z[i, t, s])
        model.addConstr(zd[i, t, s] / ineff_batt <= drate[i])
        model.addConstr(zc[i, t, s] * ineff_batt <= k[i] - z[i, t, s])
        model.addConstr(zc[i, t, s] * ineff_batt <= crate[i])
        model.addConstr(z[i, t, s] <= k[i])
        model.addConstr(
            z[i, t + 1, s]
            == z[i, t, s] + ineff_batt * zc[i, t, s] - zd[i, t, s] / ineff_batt
        )

    for i, s in product(range(i_count), range(s_count)):
        model.addConstr(z[i, 0, s] == k0[i])

    balance = {}
    for t, s in product(range(t_count), range(s_count)):
        balance[t, s] = model.addConstr(
            gp.quicksum(dp[i, t, s] for i in range(i_count))
            == gp.quicksum(dm[i, t, s] for i in range(i_count)),
            name=f"balance_{t}_{s}",
        )

    build_seconds = time.perf_counter() - build_start
    solve_start = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - solve_start

    ok_statuses = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
    if model.status not in ok_statuses or model.SolCount == 0:
        raise RuntimeError(f"Centralized solve failed with Gurobi status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array(
        [[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)]
    )
    ym_arr = np.array(
        [[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)]
    )
    dp_arr = np.array(
        [[[dp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)]
    )
    dm_arr = np.array(
        [[[dm[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)]
    )
    z_arr = np.array(
        [[[z[i, t, s].X for s in range(s_count)] for t in range(t_count + 1)] for i in range(i_count)]
    )
    zc_arr = np.array(
        [[[zc[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)]
    )
    zd_arr = np.array(
        [[[zd[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)]
    )
    lambda_dual = np.array([[balance[t, s].Pi for s in range(s_count)] for t in range(t_count)])
    profit = expected_market_profit(p_da, p_rt, p_pn, x_arr, yp_arr, ym_arr)
    reg = regularization_value(dp_arr, dm_arr, eps)

    return {
        "status": model.status,
        "gurobi_objective": model.ObjVal,
        "expected_profit": profit,
        "regularization": reg,
        "regularized_profit": profit - reg,
        "runtime": model.Runtime,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": build_seconds + solve_seconds,
        "x": x_arr,
        "yp": yp_arr,
        "ym": ym_arr,
        "dp": dp_arr,
        "dm": dm_arr,
        "z": z_arr,
        "zc": zc_arr,
        "zd": zd_arr,
        "lambda_dual": lambda_dual,
        "internal_price": -s_count * lambda_dual,
        "balance_residual": internal_balance_residual(dp_arr, dm_arr),
    }


@dataclass
class LocalDERModel:
    """One DER-local stochastic QP used inside ADMM."""

    index: int
    model: gp.Model
    x: gp.tupledict
    yp: gp.tupledict
    ym: gp.tupledict
    dp: gp.tupledict
    dm: gp.tupledict
    z: gp.tupledict
    zc: gp.tupledict
    zd: gp.tupledict
    p_da: np.ndarray
    p_rt: np.ndarray
    p_pn: np.ndarray
    eps: float
    rho: float
    t_count: int
    s_count: int

    def set_admm_objective(self, target: np.ndarray) -> None:
        """Set local objective for target ``z_i - u_i``.

        The local ADMM term is

            -(rho/2) || (dp_i - dm_i) - target_i ||_2^2.

        This is the separable augmented Lagrangian piece for the local copy
        q_i = dp_i - dm_i. The coordinator owns the consensus copy consensus_q_i and
        projects all consensus_q_i values onto the internal market balance hyperplane.
        """

        obj = gp.QuadExpr()
        lin_coeffs: list[float] = []
        lin_vars: list[gp.Var] = []
        for t in range(self.t_count):
            lin_coeffs.append(float(self.p_da[t]))
            lin_vars.append(self.x[t])

        quad_coeffs: list[float] = []
        quad_v1: list[gp.Var] = []
        quad_v2: list[gp.Var] = []

        for t, s in product(range(self.t_count), range(self.s_count)):
            lin_coeffs.extend(
                [
                    float(self.p_rt[t, s] / self.s_count),
                    float(-self.p_pn[t, s] / self.s_count),
                ]
            )
            lin_vars.extend([self.yp[t, s], self.ym[t, s]])

            # ADMM quadratic: -(rho/2) * (dp - dm - target)^2.
            # Quadratic part.
            quad_coeffs.extend(
                [
                    float(-self.rho / 2),
                    float(-self.rho / 2),
                    float(self.rho),
                ]
            )
            quad_v1.extend([self.dp[t, s], self.dm[t, s], self.dp[t, s]])
            quad_v2.extend([self.dp[t, s], self.dm[t, s], self.dm[t, s]])

            # Linear part induced by the target.
            lin_coeffs.extend([float(self.rho * target[t, s]), float(-self.rho * target[t, s])])
            lin_vars.extend([self.dp[t, s], self.dm[t, s]])

        _add_terms_chunked(obj, lin_coeffs, lin_vars)
        _add_terms_chunked(obj, quad_coeffs, quad_v1, quad_v2)
        self.model.setObjective(obj, GRB.MAXIMIZE)

    def optimize(self) -> dict[str, Any]:
        self.model.optimize()
        ok_statuses = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        if self.model.status not in ok_statuses or self.model.SolCount == 0:
            raise RuntimeError(
                f"Local DER {self.index} failed with Gurobi status {self.model.status}"
            )

        x = np.array([self.x[t].X for t in range(self.t_count)])
        yp = np.array(
            [[self.yp[t, s].X for s in range(self.s_count)] for t in range(self.t_count)]
        )
        ym = np.array(
            [[self.ym[t, s].X for s in range(self.s_count)] for t in range(self.t_count)]
        )
        dp = np.array(
            [[self.dp[t, s].X for s in range(self.s_count)] for t in range(self.t_count)]
        )
        dm = np.array(
            [[self.dm[t, s].X for s in range(self.s_count)] for t in range(self.t_count)]
        )
        z = np.array(
            [[self.z[t, s].X for s in range(self.s_count)] for t in range(self.t_count + 1)]
        )
        zc = np.array(
            [[self.zc[t, s].X for s in range(self.s_count)] for t in range(self.t_count)]
        )
        zd = np.array(
            [[self.zd[t, s].X for s in range(self.s_count)] for t in range(self.t_count)]
        )
        return {
            "x": x,
            "yp": yp,
            "ym": ym,
            "dp": dp,
            "dm": dm,
            "z": z,
            "zc": zc,
            "zd": zd,
            "q": dp - dm,
            "runtime": self.model.Runtime,
            "objective": self.model.ObjVal,
            "status": self.model.status,
        }


def build_local_der_model(
    i: int,
    data: dict[str, Any],
    eps: float,
    rho: float,
    time_limit: float | None,
    output_flag: int,
) -> LocalDERModel:
    r_i = data["R"][i]
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    k_i = data["K"][i]
    k0_i = data["K0"][i]
    crate_i = data["CRATE"][i]
    drate_i = data["DRATE"][i]
    ineff_batt = data["INEFF_BATT"]
    eta = data["INEFF_EXT"][i]
    t_count, s_count = r_i.shape

    model = gp.Model(f"admm_local_der_{i}")
    set_common_params(model, output_flag, time_limit)

    x = model.addVars(t_count, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = model.addVars(t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="yp")
    ym = model.addVars(t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="ym")
    dp = model.addVars(t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dp")
    dm = model.addVars(t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dm")
    z = model.addVars(t_count + 1, s_count, vtype=GRB.CONTINUOUS, lb=0, name="z")
    zc = model.addVars(t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zc")
    zd = model.addVars(t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zd")

    for t, s in product(range(t_count), range(s_count)):
        model.addConstr(
            r_i[t, s] - (1 / eta) * x[t]
            == (1 / eta) * yp[t, s]
            - eta * ym[t, s]
            + (1 / eta) * dp[t, s]
            - eta * dm[t, s]
            + zc[t, s]
            - zd[t, s]
        )
        model.addConstr(zd[t, s] / ineff_batt <= z[t, s])
        model.addConstr(zd[t, s] / ineff_batt <= drate_i)
        model.addConstr(zc[t, s] * ineff_batt <= k_i - z[t, s])
        model.addConstr(zc[t, s] * ineff_batt <= crate_i)
        model.addConstr(z[t, s] <= k_i)
        model.addConstr(
            z[t + 1, s] == z[t, s] + ineff_batt * zc[t, s] - zd[t, s] / ineff_batt
        )

    for s in range(s_count):
        model.addConstr(z[0, s] == k0_i)

    return LocalDERModel(
        index=i,
        model=model,
        x=x,
        yp=yp,
        ym=ym,
        dp=dp,
        dm=dm,
        z=z,
        zc=zc,
        zd=zd,
        p_da=p_da,
        p_rt=p_rt,
        p_pn=p_pn,
        eps=eps,
        rho=rho,
        t_count=t_count,
        s_count=s_count,
    )


def project_internal_balance_consensus(w: np.ndarray) -> np.ndarray:
    """Project local copies onto ``sum_i consensus_q_i(t,s)=0`` for every (t,s)."""

    return w - np.mean(w, axis=0, keepdims=True)


def stack_local_results(local_results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = ["x", "yp", "ym", "dp", "dm", "z", "zc", "zd", "q"]
    return {key: np.stack([res[key] for res in local_results], axis=0) for key in keys}


def run_admm(
    data: dict[str, Any],
    rho: float = 1e-3,
    eps: float = 1e-8,
    max_iter: int = 200,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-4,
    time_limit_per_local: float | None = None,
    output_flag: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run DER-wise stochastic consensus ADMM.

    One ADMM iteration updates all scenario-dependent price vectors together:
    the arrays ``q``, ``consensus_q``, ``u``, and ``lambda`` all have shape ``(I,T,S)``
    or ``(T,S)`` and are updated as one coupled stochastic object.
    """

    i_count = data["I"]
    t_count = data["T"]
    s_count = data["S"]
    start_total = time.perf_counter()
    build_start = time.perf_counter()
    local_models = [
        build_local_der_model(i, data, eps, rho, time_limit_per_local, output_flag)
        for i in range(i_count)
    ]
    build_seconds = time.perf_counter() - build_start

    q = np.zeros((i_count, t_count, s_count))
    consensus_q = np.zeros_like(q)
    u = np.zeros_like(q)
    history: list[dict[str, float]] = []
    local_solution: dict[str, np.ndarray] | None = None

    solve_seconds_total = 0.0
    converged = False
    for iteration in range(1, max_iter + 1):
        iter_start = time.perf_counter()
        consensus_q_old = consensus_q.copy()

        local_results = []
        for i, local in enumerate(local_models):
            # target = consensus_q_i - u_i in scaled ADMM form.
            local.set_admm_objective(consensus_q[i] - u[i])
            local_results.append(local.optimize())

        local_solution = stack_local_results(local_results)
        q = local_solution["q"]

        # Coordinator step: all DERs send q_i(t,s); coordinator projects onto
        # sum_i consensus_q_i(t,s)=0. This is where the coupling constraint is enforced.
        w = q + u
        consensus_q = project_internal_balance_consensus(w)

        # Dual update. Since consensus_q is the projection, u becomes common across DERs
        # for each (t,s), yielding scenario-dependent internal prices.
        u = u + q - consensus_q
        lambda_admm = rho * np.mean(u, axis=0)

        primal_residual = float(np.linalg.norm(q - consensus_q))
        dual_residual = float(rho * np.linalg.norm(consensus_q - consensus_q_old))
        eps_primal = math.sqrt(i_count * t_count * s_count) * abs_tol + rel_tol * max(
            float(np.linalg.norm(q)),
            float(np.linalg.norm(consensus_q)),
        )
        eps_dual = math.sqrt(i_count * t_count * s_count) * abs_tol + rel_tol * float(
            np.linalg.norm(rho * u)
        )

        iter_seconds = time.perf_counter() - iter_start
        solve_seconds_total += sum(float(res["runtime"]) for res in local_results)
        objective = expected_market_profit(
            data["P_DA"], data["P_RT"], data["P_PN"],
            local_solution["x"], local_solution["yp"], local_solution["ym"],
        )
        reg = 0.0
        balance = internal_balance_residual(local_solution["dp"], local_solution["dm"])
        row = {
            "iteration": float(iteration),
            "objective": objective,
            "regularized_profit": objective - reg,
            "regularization": reg,
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
            "eps_primal": eps_primal,
            "eps_dual": eps_dual,
            "max_balance_error": float(np.max(np.abs(balance))),
            "mean_abs_balance_error": float(np.mean(np.abs(balance))),
            "iter_wall_seconds": iter_seconds,
            "local_solver_runtime_sum": float(sum(res["runtime"] for res in local_results)),
            "communication_scalars_cumulative": float(2 * iteration * i_count * t_count * s_count),
            "internal_price_mean": float(np.mean(-s_count * lambda_admm)),
            "internal_price_std": float(np.std(-s_count * lambda_admm)),
        }
        history.append(row)

        if verbose and (iteration == 1 or iteration % 10 == 0):
            print(
                f"ADMM iter={iteration:03d} "
                f"obj={objective:,.3f} "
                f"r={primal_residual:.3e}/{eps_primal:.3e} "
                f"s={dual_residual:.3e}/{eps_dual:.3e}"
            )

        if primal_residual <= eps_primal and dual_residual <= eps_dual:
            converged = True
            break

    if local_solution is None:
        raise RuntimeError("ADMM produced no local solution.")

    total_seconds = time.perf_counter() - start_total
    lambda_admm = rho * np.mean(u, axis=0)
    balance = internal_balance_residual(local_solution["dp"], local_solution["dm"])
    objective = expected_market_profit(
        data["P_DA"], data["P_RT"], data["P_PN"],
        local_solution["x"], local_solution["yp"], local_solution["ym"],
    )
    reg = 0.0

    return {
        "converged": converged,
        "iterations": len(history),
        "rho": rho,
        "expected_profit": objective,
        "regularization": reg,
        "regularized_profit": objective - reg,
        "build_seconds": build_seconds,
        "local_solver_runtime_sum": solve_seconds_total,
        "total_seconds": total_seconds,
        "history": pd.DataFrame(history),
        "lambda_admm": lambda_admm,
        "internal_price": -s_count * lambda_admm,
        "u": u,
        "q": q,
        "consensus_q": consensus_q,
        "balance_residual": balance,
        **local_solution,
    }


def solve_individual_participation(
    data: dict[str, Any],
    time_limit: float | None = 1200.0,
    output_flag: int = 0,
) -> dict[str, Any]:
    """Solve the independent market participation benchmark with no internal market."""

    r = data["R"]
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    k = data["K"]
    k0 = data["K0"]
    crate = data["CRATE"]
    drate = data["DRATE"]
    ineff_batt = data["INEFF_BATT"]
    ineff_ext = data["INEFF_EXT"]
    i_count, t_count, s_count = r.shape

    build_start = time.perf_counter()
    model = gp.Model("individual_participation")
    set_common_params(model, output_flag, time_limit)

    x = model.addVars(i_count, t_count, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="yp")
    ym = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="ym")
    z = model.addVars(i_count, t_count + 1, s_count, vtype=GRB.CONTINUOUS, lb=0, name="z")
    zc = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zc")
    zd = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zd")

    obj = gp.QuadExpr()
    lin_coeffs: list[float] = []
    lin_vars: list[gp.Var] = []
    for i in range(i_count):
        for t in range(t_count):
            lin_coeffs.append(float(p_da[t]))
            lin_vars.append(x[i, t])
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        lin_coeffs.extend([float(p_rt[t, s] / s_count), float(-p_pn[t, s] / s_count)])
        lin_vars.extend([yp[i, t, s], ym[i, t, s]])
    _add_terms_chunked(obj, lin_coeffs, lin_vars)
    model.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        eta = ineff_ext[i]
        model.addConstr(
            r[i, t, s] - (1 / eta) * x[i, t]
            == (1 / eta) * yp[i, t, s]
            - eta * ym[i, t, s]
            + zc[i, t, s]
            - zd[i, t, s]
        )
        model.addConstr(zd[i, t, s] / ineff_batt <= z[i, t, s])
        model.addConstr(zd[i, t, s] / ineff_batt <= drate[i])
        model.addConstr(zc[i, t, s] * ineff_batt <= k[i] - z[i, t, s])
        model.addConstr(zc[i, t, s] * ineff_batt <= crate[i])
        model.addConstr(z[i, t, s] <= k[i])
        model.addConstr(
            z[i, t + 1, s]
            == z[i, t, s] + ineff_batt * zc[i, t, s] - zd[i, t, s] / ineff_batt
        )
    for i, s in product(range(i_count), range(s_count)):
        model.addConstr(z[i, 0, s] == k0[i])

    build_seconds = time.perf_counter() - build_start
    solve_start = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - solve_start

    ok_statuses = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
    if model.status not in ok_statuses or model.SolCount == 0:
        raise RuntimeError(f"Individual participation failed with Gurobi status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array([[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    ym_arr = np.array([[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    z_arr = np.array([[[z[i, t, s].X for s in range(s_count)] for t in range(t_count + 1)] for i in range(i_count)])
    zc_arr = np.array([[[zc[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    zd_arr = np.array([[[zd[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    zeros = np.zeros((i_count, t_count, s_count))
    profit = expected_market_profit(p_da, p_rt, p_pn, x_arr, yp_arr, ym_arr)

    return {
        "status": model.status,
        "expected_profit": profit,
        "runtime": model.Runtime,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": build_seconds + solve_seconds,
        "x": x_arr,
        "yp": yp_arr,
        "ym": ym_arr,
        "dp": zeros.copy(),
        "dm": zeros.copy(),
        "z": z_arr,
        "zc": zc_arr,
        "zd": zd_arr,
    }


def solve_price_replay(
    data: dict[str, Any],
    internal_price: np.ndarray,
    eps: float = 1e-8,
    time_limit: float | None = 1200.0,
    output_flag: int = 0,
) -> dict[str, Any]:
    """Solve decentralized replay against a fixed internal price matrix.

    This is the DAgg-style response in ``0527.ipynb``. There is no aggregate
    balance constraint. Each DER optimizes against the fixed internal price,
    and any remaining aggregate imbalance is evaluated afterward.
    """

    r = data["R"]
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    k = data["K"]
    k0 = data["K0"]
    crate = data["CRATE"]
    drate = data["DRATE"]
    ineff_batt = data["INEFF_BATT"]
    ineff_ext = data["INEFF_EXT"]
    i_count, t_count, s_count = r.shape

    build_start = time.perf_counter()
    model = gp.Model("price_replay_dagg")
    set_common_params(model, output_flag, time_limit)

    x = model.addVars(i_count, t_count, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="yp")
    ym = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="ym")
    dp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dp")
    dm = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dm")
    z = model.addVars(i_count, t_count + 1, s_count, vtype=GRB.CONTINUOUS, lb=0, name="z")
    zc = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zc")
    zd = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zd")

    obj = gp.QuadExpr()
    lin_coeffs: list[float] = []
    lin_vars: list[gp.Var] = []
    for i in range(i_count):
        for t in range(t_count):
            lin_coeffs.append(float(p_da[t]))
            lin_vars.append(x[i, t])
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        price_coeff = float(internal_price[t, s] / s_count)
        lin_coeffs.extend(
            [
                float(p_rt[t, s] / s_count),
                float(-p_pn[t, s] / s_count),
                price_coeff,
                -price_coeff,
            ]
        )
        lin_vars.extend([yp[i, t, s], ym[i, t, s], dp[i, t, s], dm[i, t, s]])
    _add_terms_chunked(obj, lin_coeffs, lin_vars)

    quad_coeffs: list[float] = []
    quad_vars: list[gp.Var] = []
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        coeff = float(-eps / s_count)
        quad_coeffs.extend([coeff, coeff])
        quad_vars.extend([dp[i, t, s], dm[i, t, s]])
    _add_terms_chunked(obj, quad_coeffs, quad_vars, quad_vars)
    model.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        eta = ineff_ext[i]
        model.addConstr(
            r[i, t, s] - (1 / eta) * x[i, t]
            == (1 / eta) * yp[i, t, s]
            - eta * ym[i, t, s]
            + (1 / eta) * dp[i, t, s]
            - eta * dm[i, t, s]
            + zc[i, t, s]
            - zd[i, t, s]
        )
        model.addConstr(zd[i, t, s] / ineff_batt <= z[i, t, s])
        model.addConstr(zd[i, t, s] / ineff_batt <= drate[i])
        model.addConstr(zc[i, t, s] * ineff_batt <= k[i] - z[i, t, s])
        model.addConstr(zc[i, t, s] * ineff_batt <= crate[i])
        model.addConstr(z[i, t, s] <= k[i])
        model.addConstr(
            z[i, t + 1, s]
            == z[i, t, s] + ineff_batt * zc[i, t, s] - zd[i, t, s] / ineff_batt
        )
    for i, s in product(range(i_count), range(s_count)):
        model.addConstr(z[i, 0, s] == k0[i])

    build_seconds = time.perf_counter() - build_start
    solve_start = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - solve_start

    ok_statuses = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
    if model.status not in ok_statuses or model.SolCount == 0:
        raise RuntimeError(f"Price replay failed with Gurobi status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array([[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    ym_arr = np.array([[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dp_arr = np.array([[[dp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dm_arr = np.array([[[dm[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    z_arr = np.array([[[z[i, t, s].X for s in range(s_count)] for t in range(t_count + 1)] for i in range(i_count)])
    zc_arr = np.array([[[zc[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    zd_arr = np.array([[[zd[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])

    market_profit = expected_market_profit(p_da, p_rt, p_pn, x_arr, yp_arr, ym_arr)
    internal_settlement = internal_settlement_value(internal_price, dp_arr, dm_arr)
    reg = regularization_value(dp_arr, dm_arr, eps)

    return {
        "status": model.status,
        "expected_profit": market_profit,
        "internal_settlement": internal_settlement,
        "regularization": reg,
        "objective_with_settlement": market_profit + internal_settlement - reg,
        "runtime": model.Runtime,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": build_seconds + solve_seconds,
        "internal_price": internal_price,
        "x": x_arr,
        "yp": yp_arr,
        "ym": ym_arr,
        "dp": dp_arr,
        "dm": dm_arr,
        "z": z_arr,
        "zc": zc_arr,
        "zd": zd_arr,
        "balance_residual": internal_balance_residual(dp_arr, dm_arr),
    }


def internal_settlement_value(internal_price: np.ndarray, dp: np.ndarray, dm: np.ndarray) -> float:
    """Aggregate expected internal settlement: average_s sum_i,t price*(dp-dm)."""

    return float(np.mean(np.sum(internal_price[None, :, :] * (dp - dm), axis=(0, 1))))


def market_profit_by_der(
    p_da: np.ndarray,
    p_rt: np.ndarray,
    p_pn: np.ndarray,
    x: np.ndarray,
    yp: np.ndarray,
    ym: np.ndarray,
) -> np.ndarray:
    da = np.sum(p_da[None, :] * x, axis=1)
    rt = np.mean(np.sum(p_rt[None, :, :] * yp, axis=1), axis=1)
    penalty = np.mean(np.sum(p_pn[None, :, :] * ym, axis=1), axis=1)
    return da + rt - penalty


def internal_settlement_by_der(internal_price: np.ndarray, dp: np.ndarray, dm: np.ndarray) -> np.ndarray:
    return np.mean(np.sum(internal_price[None, :, :] * (dp - dm), axis=1), axis=1)


def imbalance_summary(solution: dict[str, Any], internal_price: np.ndarray, p_rt: np.ndarray) -> dict[str, Any]:
    """Evaluate aggregate internal imbalance using the 0527.ipynb convention."""

    dp = solution["dp"]
    dm = solution["dm"]
    imbalance = np.sum(dm - dp, axis=0)
    by_ts_value = imbalance * (internal_price - p_rt)
    by_t_expected = np.mean(by_ts_value, axis=1)
    penalty = float(np.sum(by_t_expected))
    return {
        "imbalance_penalty": penalty,
        "imbalance_quantity_kwh": float(np.sum(np.mean(np.abs(imbalance), axis=1))),
        "imbalance_quantity_mwh": float(np.sum(np.mean(np.abs(imbalance), axis=1)) / 1000.0),
        "max_balance_error": float(np.max(np.abs(np.sum(dp - dm, axis=0)))),
        "mean_abs_balance_error": float(np.mean(np.abs(np.sum(dp - dm, axis=0)))),
        "imbalance_by_ts": imbalance,
        "imbalance_value_by_ts": by_ts_value,
    }


def allocate_imbalance_penalty(
    solution: dict[str, Any],
    internal_price: np.ndarray,
    imbalance_penalty: float,
) -> np.ndarray:
    """Allocate aggregate imbalance value to DERs using 0527-style price-weighted usage."""

    dp = solution["dp"]
    dm = solution["dm"]
    weights = np.sum(internal_price[None, :, :] * (dp + dm), axis=(1, 2))
    total_weight = float(np.sum(weights))
    if abs(total_weight) <= 1e-9:
        return np.full(dp.shape[0], imbalance_penalty / dp.shape[0])
    return imbalance_penalty * weights / total_weight


def individual_profit_summary(
    data: dict[str, Any],
    individual: dict[str, Any],
    centralized: dict[str, Any],
    dagg: dict[str, Any],
    admm: dict[str, Any],
    admm_replay: dict[str, Any],
) -> pd.DataFrame:
    """Compare DER-level profit against independent market participation."""

    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    c_price = centralized["internal_price"]
    d_price = dagg["internal_price"]
    a_price = admm["internal_price"]
    ar_price = admm_replay["internal_price"]

    ind_profit = market_profit_by_der(p_da, p_rt, p_pn, individual["x"], individual["yp"], individual["ym"])

    cagg_profit = market_profit_by_der(p_da, p_rt, p_pn, centralized["x"], centralized["yp"], centralized["ym"])
    cagg_profit += internal_settlement_by_der(c_price, centralized["dp"], centralized["dm"])

    dagg_profit = market_profit_by_der(p_da, p_rt, p_pn, dagg["x"], dagg["yp"], dagg["ym"])
    dagg_profit += internal_settlement_by_der(d_price, dagg["dp"], dagg["dm"])
    dagg_imb = imbalance_summary(dagg, d_price, p_rt)
    dagg_alloc = allocate_imbalance_penalty(dagg, d_price, dagg_imb["imbalance_penalty"])
    dagg_profit_with_imb = dagg_profit + dagg_alloc

    admm_profit = market_profit_by_der(p_da, p_rt, p_pn, admm["x"], admm["yp"], admm["ym"])
    admm_profit += internal_settlement_by_der(a_price, admm["dp"], admm["dm"])
    admm_imb = imbalance_summary(admm, a_price, p_rt)
    admm_alloc = allocate_imbalance_penalty(admm, a_price, admm_imb["imbalance_penalty"])
    admm_profit_with_imb = admm_profit + admm_alloc

    admm_replay_profit = market_profit_by_der(p_da, p_rt, p_pn, admm_replay["x"], admm_replay["yp"], admm_replay["ym"])
    admm_replay_profit += internal_settlement_by_der(ar_price, admm_replay["dp"], admm_replay["dm"])
    admm_replay_imb = imbalance_summary(admm_replay, ar_price, p_rt)
    admm_replay_alloc = allocate_imbalance_penalty(admm_replay, ar_price, admm_replay_imb["imbalance_penalty"])
    admm_replay_profit_with_imb = admm_replay_profit + admm_replay_alloc

    df = pd.DataFrame(
        {
            "der": np.arange(data["I"]),
            "individual_profit": ind_profit,
            "cagg_profit_at_central_price": cagg_profit,
            "dagg_profit_before_imbalance": dagg_profit,
            "dagg_allocated_imbalance": dagg_alloc,
            "dagg_profit_with_imbalance": dagg_profit_with_imb,
            "admm_final_profit_before_imbalance": admm_profit,
            "admm_final_allocated_imbalance": admm_alloc,
            "admm_final_profit_with_imbalance": admm_profit_with_imb,
            "admm_replay_profit_before_imbalance": admm_replay_profit,
            "admm_replay_allocated_imbalance": admm_replay_alloc,
            "admm_replay_profit_with_imbalance": admm_replay_profit_with_imb,
        }
    )
    df["cagg_gain_vs_individual"] = df["cagg_profit_at_central_price"] - df["individual_profit"]
    df["dagg_gain_vs_individual"] = df["dagg_profit_with_imbalance"] - df["individual_profit"]
    df["admm_final_gain_vs_individual"] = df["admm_final_profit_with_imbalance"] - df["individual_profit"]
    df["admm_replay_gain_vs_individual"] = df["admm_replay_profit_with_imbalance"] - df["individual_profit"]
    return df


def aggregate_comparison_summary(
    data: dict[str, Any],
    individual: dict[str, Any],
    centralized: dict[str, Any],
    dagg: dict[str, Any],
    admm: dict[str, Any],
    admm_replay: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate-level comparison: individual, CAGG, DAgg, and ADMM."""

    p_rt = data["P_RT"]
    rows = []
    rows.append(
        {
            "case": "individual_participation",
            "market_profit": individual["expected_profit"],
            "internal_settlement": 0.0,
            "imbalance_penalty": 0.0,
            "profit_with_imbalance": individual["expected_profit"],
            "imbalance_quantity_mwh": 0.0,
            "max_balance_error": 0.0,
            "runtime_seconds": individual["total_seconds"],
            "iterations": np.nan,
        }
    )
    c_imb = imbalance_summary(centralized, centralized["internal_price"], p_rt)
    rows.append(
        {
            "case": "cagg_centralized",
            "market_profit": centralized["expected_profit"],
            "internal_settlement": internal_settlement_value(centralized["internal_price"], centralized["dp"], centralized["dm"]),
            "imbalance_penalty": c_imb["imbalance_penalty"],
            "profit_with_imbalance": centralized["expected_profit"] + c_imb["imbalance_penalty"],
            "imbalance_quantity_mwh": c_imb["imbalance_quantity_mwh"],
            "max_balance_error": c_imb["max_balance_error"],
            "runtime_seconds": centralized["total_seconds"],
            "iterations": np.nan,
        }
    )
    d_imb = imbalance_summary(dagg, dagg["internal_price"], p_rt)
    rows.append(
        {
            "case": "dagg_replay_with_imbalance",
            "market_profit": dagg["expected_profit"],
            "internal_settlement": dagg["internal_settlement"],
            "imbalance_penalty": d_imb["imbalance_penalty"],
            "profit_with_imbalance": dagg["expected_profit"] + dagg["internal_settlement"] + d_imb["imbalance_penalty"],
            "imbalance_quantity_mwh": d_imb["imbalance_quantity_mwh"],
            "max_balance_error": d_imb["max_balance_error"],
            "runtime_seconds": dagg["total_seconds"],
            "iterations": np.nan,
        }
    )
    a_imb = imbalance_summary(admm, admm["internal_price"], p_rt)
    rows.append(
        {
            "case": "admm_final_iterate_with_imbalance",
            "market_profit": admm["expected_profit"],
            "internal_settlement": internal_settlement_value(admm["internal_price"], admm["dp"], admm["dm"]),
            "imbalance_penalty": a_imb["imbalance_penalty"],
            "profit_with_imbalance": admm["expected_profit"] + internal_settlement_value(admm["internal_price"], admm["dp"], admm["dm"]) + a_imb["imbalance_penalty"],
            "imbalance_quantity_mwh": a_imb["imbalance_quantity_mwh"],
            "max_balance_error": a_imb["max_balance_error"],
            "runtime_seconds": admm["total_seconds"],
            "iterations": admm["iterations"],
        }
    )
    ar_imb = imbalance_summary(admm_replay, admm_replay["internal_price"], p_rt)
    rows.append(
        {
            "case": "admm_price_replay_with_imbalance",
            "market_profit": admm_replay["expected_profit"],
            "internal_settlement": admm_replay["internal_settlement"],
            "imbalance_penalty": ar_imb["imbalance_penalty"],
            "profit_with_imbalance": admm_replay["expected_profit"] + admm_replay["internal_settlement"] + ar_imb["imbalance_penalty"],
            "imbalance_quantity_mwh": ar_imb["imbalance_quantity_mwh"],
            "max_balance_error": ar_imb["max_balance_error"],
            "runtime_seconds": admm_replay["total_seconds"],
            "iterations": np.nan,
        }
    )
    return pd.DataFrame(rows)


def price_comparison_summary(
    centralized: dict[str, Any],
    dagg: dict[str, Any],
    admm: dict[str, Any],
    admm_replay: dict[str, Any],
) -> pd.DataFrame:
    c_price = centralized["internal_price"]
    d_price = dagg["internal_price"]
    a_price = admm["internal_price"]
    ar_price = admm_replay["internal_price"]
    rows = []
    for name, price in [("cagg_centralized", c_price), ("dagg_replay", d_price), ("admm_final_iterate", a_price), ("admm_price_replay", ar_price)]:
        rows.append(
            {
                "case": name,
                "price_mean": float(np.mean(price)),
                "price_std": float(np.std(price)),
                "price_min": float(np.min(price)),
                "price_max": float(np.max(price)),
                "rmse_vs_cagg_price": float(np.sqrt(np.mean((price - c_price) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def build_full_comparison_from_solutions(
    data: dict[str, Any],
    centralized: dict[str, Any],
    admm: dict[str, Any],
    eps: float = 1e-8,
    time_limit: float | None = 1200.0,
    output_flag: int = 0,
) -> dict[str, Any]:
    """Build full comparison tables from existing CAGG and ADMM solves.

    This avoids rerunning the expensive centralized and ADMM benchmarks when a
    notebook has already computed them. It only solves the additional
    independent-participation and DAgg replay benchmarks needed for reporting.
    """

    individual = solve_individual_participation(
        data, time_limit=time_limit, output_flag=output_flag
    )
    dagg = solve_price_replay(
        data,
        centralized["internal_price"],
        eps=eps,
        time_limit=time_limit,
        output_flag=output_flag,
    )
    admm_replay = solve_price_replay(
        data,
        admm["internal_price"],
        eps=eps,
        time_limit=time_limit,
        output_flag=output_flag,
    )
    aggregate_df = aggregate_comparison_summary(data, individual, centralized, dagg, admm, admm_replay)
    individual_df = individual_profit_summary(data, individual, centralized, dagg, admm, admm_replay)
    price_df = price_comparison_summary(centralized, dagg, admm, admm_replay)
    return {
        "data": data,
        "individual": individual,
        "centralized": centralized,
        "dagg": dagg,
        "admm": admm,
        "admm_replay": admm_replay,
        "aggregate_df": aggregate_df,
        "individual_df": individual_df,
        "price_df": price_df,
    }


def run_full_comparison_case(
    n_der: int = 10,
    scenarios: int = 100,
    seed: int = 1,
    rho: float = 1.0,
    max_iter: int = 200,
    eps: float = 1e-8,
    level: str = "high",
    time_limit: float | None = 1200.0,
    output_flag: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run individual, CAGG, DAgg replay, and ADMM for one case."""

    data = build_problem_data(n_der=n_der, scenarios=scenarios, seed=seed, level=level)
    centralized = solve_centralized(data, eps=eps, time_limit=time_limit, output_flag=output_flag)
    admm = run_admm(
        data,
        rho=rho,
        eps=eps,
        max_iter=max_iter,
        output_flag=output_flag,
        verbose=verbose,
    )
    return build_full_comparison_from_solutions(
        data,
        centralized,
        admm,
        eps=eps,
        time_limit=time_limit,
        output_flag=output_flag,
    )


def compare_centralized_admm(
    centralized: dict[str, Any],
    admm: dict[str, Any],
) -> dict[str, Any]:
    central_profit = centralized["expected_profit"]
    admm_profit = admm["expected_profit"]
    gap = central_profit - admm_profit
    rel_gap = gap / abs(central_profit) if central_profit else np.nan
    price_rmse = float(np.sqrt(np.mean((centralized["internal_price"] - admm["internal_price"]) ** 2)))
    return {
        "centralized_expected_profit": central_profit,
        "admm_expected_profit": admm_profit,
        "optimality_gap": gap,
        "relative_gap": rel_gap,
        "centralized_runtime_seconds": centralized["total_seconds"],
        "admm_runtime_seconds": admm["total_seconds"],
        "admm_iterations": admm["iterations"],
        "admm_converged": admm["converged"],
        "communication_scalars": int(2 * admm["iterations"] * admm["q"].shape[0] * admm["q"].shape[1] * admm["q"].shape[2]),
        "max_balance_error": float(np.max(np.abs(admm["balance_residual"]))),
        "mean_abs_balance_error": float(np.mean(np.abs(admm["balance_residual"]))),
        "price_rmse": price_rmse,
    }


def run_one_experiment(
    n_der: int = 10,
    scenarios: int = 100,
    seed: int = 1,
    rho: float = 1e-3,
    max_iter: int = 200,
    eps: float = 1e-8,
    level: str = "high",
    time_limit: float | None = 1200.0,
    time_limit_per_local: float | None = None,
    output_flag: int = 0,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = build_problem_data(n_der=n_der, scenarios=scenarios, seed=seed, level=level)
    centralized = solve_centralized(data, eps=eps, time_limit=time_limit, output_flag=output_flag)
    admm = run_admm(
        data,
        rho=rho,
        eps=eps,
        max_iter=max_iter,
        time_limit_per_local=time_limit_per_local,
        output_flag=output_flag,
        verbose=verbose,
    )
    summary = compare_centralized_admm(centralized, admm)
    summary.update(
        {
            "n_der": n_der,
            "scenarios": scenarios,
            "seed": seed,
            "rho": rho,
            "eps": eps,
            "centralized_regularization_enabled": True,
            "admm_objective_regularization_enabled": False,
            "centralized_regularization": centralized["regularization"],
            "admm_regularization": admm["regularization"],
            "python_version": sys.version.split()[0],
            "selected_files_json": json.dumps(data["selected_files"], ensure_ascii=True),
        }
    )
    return data, centralized, admm, summary


def run_scalability_grid(
    n_values: list[int] | tuple[int, ...] = (10, 30, 50, 75, 100),
    scenario_values: list[int] | tuple[int, ...] = (25, 50, 100),
    seed: int = 1,
    rho: float = 1e-3,
    max_iter: int = 100,
    eps: float = 1e-8,
    level: str = "high",
    output_csv: str | Path | None = None,
    output_flag: int = 0,
) -> pd.DataFrame:
    """Run a scalability grid and return one row per (N,S) case."""

    rows = []
    for n_der in n_values:
        for scenarios in scenario_values:
            print(f"[start] N={n_der}, S={scenarios}, seed={seed}")
            wall_start = time.perf_counter()
            _, centralized, admm, summary = run_one_experiment(
                n_der=n_der,
                scenarios=scenarios,
                seed=seed,
                rho=rho,
                max_iter=max_iter,
                eps=eps,
                level=level,
                output_flag=output_flag,
                verbose=False,
            )
            summary["experiment_wall_seconds"] = time.perf_counter() - wall_start
            rows.append(summary)
            print(
                f"[done] N={n_der}, S={scenarios}, "
                f"gap={summary['relative_gap']:.3e}, "
                f"iters={summary['admm_iterations']}, "
                f"central={centralized['total_seconds']:.2f}s, "
                f"admm={admm['total_seconds']:.2f}s"
            )
            if output_csv is not None:
                output_path = Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    return pd.DataFrame(rows)


def _write_rows(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def _completed_cases(output_csv: str | Path) -> set[tuple[int, int, int]]:
    path = Path(output_csv)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    required = {"n_der", "scenarios", "seed"}
    if not required.issubset(df.columns):
        return set()
    return {
        (int(row.n_der), int(row.scenarios), int(row.seed))
        for row in df.itertuples(index=False)
    }


def summarize_scalability_results(df: pd.DataFrame) -> pd.DataFrame:
    """Average scalability results across seeds for paper tables/plots."""

    if df.empty:
        return df.copy()
    group_cols = ["n_der", "scenarios", "rho", "eps"]
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"seed", *group_cols}
    ]
    avg = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    seed_count = df.groupby(group_cols, as_index=False)["seed"].nunique()
    seed_count = seed_count.rename(columns={"seed": "seed_count"})
    return seed_count.merge(avg, on=group_cols, how="left")


def run_overnight_scalability(
    n_values: list[int] | tuple[int, ...] = (10, 30, 50, 75, 100),
    seeds: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 25),
    scenarios: int = 100,
    rho: float = 1.0,
    max_iter: int = 200,
    eps: float = 1e-8,
    level: str = "high",
    output_csv: str | Path = "admm_noreg_overnight_results.csv",
    average_csv: str | Path | None = "admm_noreg_overnight_seed_average.csv",
    skip_completed: bool = True,
    time_limit: float | None = 1200.0,
    time_limit_per_local: float | None = None,
    output_flag: int = 0,
) -> pd.DataFrame:
    """Run the paper-scale ADMM comparison grid with checkpointed CSV output.

    The intended revision experiment is fixed at ``scenarios=100`` and sweeps
    DER counts and random seeds. One row is saved after every completed case,
    so a long overnight run can be inspected or resumed.
    """

    output_path = Path(output_csv)
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
    completed = _completed_cases(output_path) if skip_completed else set()
    total_cases = len(n_values) * len(seeds)
    case_index = 0

    for seed in seeds:
        for n_der in n_values:
            case_index += 1
            key = (int(n_der), int(scenarios), int(seed))
            if key in completed:
                print(f"[skip {case_index}/{total_cases}] N={n_der}, S={scenarios}, seed={seed}")
                continue

            print(f"[start {case_index}/{total_cases}] N={n_der}, S={scenarios}, seed={seed}")
            wall_start = time.perf_counter()
            _, centralized, admm, summary = run_one_experiment(
                n_der=n_der,
                scenarios=scenarios,
                seed=seed,
                rho=rho,
                max_iter=max_iter,
                eps=eps,
                level=level,
                time_limit=time_limit,
                time_limit_per_local=time_limit_per_local,
                output_flag=output_flag,
                verbose=False,
            )
            summary["experiment_wall_seconds"] = time.perf_counter() - wall_start
            rows.append(summary)
            _write_rows(rows, output_path)

            if average_csv is not None:
                avg = summarize_scalability_results(pd.DataFrame(rows))
                avg_path = Path(average_csv)
                avg_path.parent.mkdir(parents=True, exist_ok=True)
                avg.to_csv(avg_path, index=False, encoding="utf-8")

            print(
                f"[done {case_index}/{total_cases}] N={n_der}, S={scenarios}, seed={seed}, "
                f"gap={summary['relative_gap']:.3e}, "
                f"iters={summary['admm_iterations']}, "
                f"central={centralized['total_seconds']:.2f}s, "
                f"admm={admm['total_seconds']:.2f}s, "
                f"comm={summary['communication_scalars']}"
            )

    result = pd.DataFrame(rows)
    if average_csv is not None and not result.empty:
        avg = summarize_scalability_results(result)
        avg_path = Path(average_csv)
        avg_path.parent.mkdir(parents=True, exist_ok=True)
        avg.to_csv(avg_path, index=False, encoding="utf-8")
    return result


def run_rho_sensitivity(
    n_der: int = 10,
    scenarios: int = 100,
    seed: int = 1,
    rho_values: list[float] | tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 2.0),
    max_iter: int = 200,
    eps: float = 1e-8,
    level: str = "high",
    output_csv: str | Path | None = None,
    time_limit: float | None = 1200.0,
    time_limit_per_local: float | None = None,
    output_flag: int = 0,
) -> pd.DataFrame:
    """Run ADMM with multiple rho values on the same stochastic instance.

    The centralized benchmark is solved once. Each rho value then runs a fresh
    ADMM solve against the same data so that convergence speed, balance error,
    price error, runtime, and objective gap can be compared.
    """

    data = build_problem_data(n_der=n_der, scenarios=scenarios, seed=seed, level=level)
    centralized = solve_centralized(
        data, eps=eps, time_limit=time_limit, output_flag=output_flag
    )

    rows: list[dict[str, Any]] = []
    for rho in rho_values:
        print(f"[rho sensitivity] N={n_der}, S={scenarios}, seed={seed}, rho={rho}")
        wall_start = time.perf_counter()
        admm = run_admm(
            data,
            rho=float(rho),
            eps=eps,
            max_iter=max_iter,
            time_limit_per_local=time_limit_per_local,
            output_flag=output_flag,
            verbose=False,
        )
        summary = compare_centralized_admm(centralized, admm)
        final_hist = admm["history"].iloc[-1].to_dict()
        summary.update(
            {
                "n_der": n_der,
                "scenarios": scenarios,
                "seed": seed,
                "rho": float(rho),
                "eps": eps,
                "experiment_wall_seconds": time.perf_counter() - wall_start,
                "final_primal_residual": final_hist["primal_residual"],
                "final_dual_residual": final_hist["dual_residual"],
                "final_eps_primal": final_hist["eps_primal"],
                "final_eps_dual": final_hist["eps_dual"],
                "final_internal_price_mean": final_hist["internal_price_mean"],
                "final_internal_price_std": final_hist["internal_price_std"],
            }
        )
        rows.append(summary)
        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
        print(
            f"[rho done] rho={rho}, "
            f"iters={summary['admm_iterations']}, "
            f"gap={summary['relative_gap']:.3e}, "
            f"balance={summary['max_balance_error']:.3e}, "
            f"price_rmse={summary['price_rmse']:.3e}"
        )

    return pd.DataFrame(rows)
