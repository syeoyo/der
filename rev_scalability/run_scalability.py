import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


DEFAULT_COUNTS = [10, 30, 50, 75, 100]
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 25]
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


def generation_files(root: Path) -> list[str]:
    data_dir = root / "data" / "generation"
    all_files = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.name != "past"
    )
    missing_initial = [name for name in INITIAL_SUBMISSION_FILES if name not in all_files]
    if missing_initial:
        raise ValueError(f"Missing initial-submission generation files: {missing_initial}")
    rest = [name for name in all_files if name not in INITIAL_SUBMISSION_FILES]
    return INITIAL_SUBMISSION_FILES + rest


def load_generation_data(root: Path, include_files: list[str], date_filter: str) -> tuple[np.ndarray, int, int]:
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

        # Preserve the notebook's legacy hour indexing for comparability.
        for t in range(24):
            if t in pd.Series(df["Time"]).values:
                generation_data[idx, t] = pd.Series(df[df["Time"] == t][gen_col]).values[0]

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
    np.random.seed(random_seed)
    noise = np.random.uniform(low, high, size=(*generation_data.shape, scenarios))
    return np.expand_dims(generation_data, axis=-1) * noise


def generate_rt_scenarios(root: Path, scenarios: int, random_seed: int) -> np.ndarray:
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
        nyc_rt.groupby("Hour")["LBMP ($/MWHr)"].mean().reset_index()["LBMP ($/MWHr)"].to_numpy()
    )

    np.random.seed(random_seed)
    noise = np.random.uniform(0.4, 1.6, size=(len(price_hourly), scenarios))
    return price_hourly[:, None] * noise


def load_price_data(root: Path, p_rt: np.ndarray, scale_da: float = 1.5, scale_penalty: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    price_path = root / "data" / "price" / "20220718da.csv"
    ny_da = pd.read_csv(price_path)
    ny_da["Time Stamp"] = pd.to_datetime(ny_da["Time Stamp"])
    ny_da["Hour"] = pd.Series(ny_da["Time Stamp"]).dt.hour
    zone = ny_da[ny_da["Name"] == "MHK VL"]
    p_da = np.array(zone["LBMP ($/MWHr)"].astype(float)) * float(scale_da)
    p_pn = np.maximum(p_da[:, None], p_rt) * float(scale_penalty)
    return p_da, p_pn


def load_parameters(
    generation_data: np.ndarray,
    scenarios: int,
    randomness_level: str,
    random_seed: int,
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = generate_randomized_generation(generation_data, scenarios, randomness_level, random_seed)
    p_rt = generate_rt_scenarios(root, scenarios, random_seed)

    peak_generations = np.mean(generation_data, axis=1)
    k = (peak_generations * 2 // 100) * 100
    k0 = np.zeros(len(generation_data))
    crate = k / 4
    drate = k / 4
    return r, p_rt, k, k0, crate, drate


def set_common_params(model: gp.Model, output_flag: int, time_limit: float | None) -> None:
    model.setParam("OutputFlag", output_flag)
    model.setParam("MIPGap", 1e-5)
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)


def add_terms_chunked(
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


def solve_cagg(
    r: np.ndarray,
    k: np.ndarray,
    k0: np.ndarray,
    crate: np.ndarray,
    drate: np.ndarray,
    p_da: np.ndarray,
    p_rt: np.ndarray,
    p_pn: np.ndarray,
    eps: float,
    ineff_batt: float,
    ineff_ext: np.ndarray,
    time_limit: float | None,
    output_flag: int,
) -> dict:
    i_count, t_count, s_count = r.shape
    build_start = time.perf_counter()
    model = gp.Model("REG_CAgg")
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
    lin_coeffs = []
    lin_vars = []
    for i in range(i_count):
        for t in range(t_count):
            lin_coeffs.append(float(p_da[t]))
            lin_vars.append(x[i, t])
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        lin_coeffs.extend([float(p_rt[t, s] / s_count), float(-p_pn[t, s] / s_count)])
        lin_vars.extend([yp[i, t, s], ym[i, t, s]])
    add_terms_chunked(obj, lin_coeffs, lin_vars)

    quad_coeffs = []
    quad_vars = []
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        coeff = float(-eps / s_count)
        quad_coeffs.extend([coeff, coeff])
        quad_vars.extend([dp[i, t, s], dm[i, t, s]])
    add_terms_chunked(obj, quad_coeffs, quad_vars, quad_vars)
    model.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        eta_e = ineff_ext[i]
        model.addConstr(
            r[i, t, s] - (1 / eta_e) * x[i, t]
            == (1 / eta_e) * yp[i, t, s]
            - eta_e * ym[i, t, s]
            + (1 / eta_e) * dp[i, t, s]
            - eta_e * dm[i, t, s]
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
        raise RuntimeError(f"REG-CAgg failed with status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array([[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    ym_arr = np.array([[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dp_arr = np.array([[[dp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dm_arr = np.array([[[dm[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    lambda_dual = np.array([[balance[t, s].Pi for s in range(s_count)] for t in range(t_count)])

    return {
        "status": model.status,
        "objective": model.ObjVal,
        "runtime": model.Runtime,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "x": x_arr,
        "yp": yp_arr,
        "ym": ym_arr,
        "dp": dp_arr,
        "dm": dm_arr,
        "lambda_dual": lambda_dual,
    }


def solve_dagg(
    r: np.ndarray,
    k: np.ndarray,
    k0: np.ndarray,
    crate: np.ndarray,
    drate: np.ndarray,
    p_da: np.ndarray,
    p_rt: np.ndarray,
    p_pn: np.ndarray,
    lambda_dual: np.ndarray,
    eps: float,
    ineff_batt: float,
    ineff_ext: np.ndarray,
    time_limit: float | None,
    output_flag: int,
) -> dict:
    i_count, t_count, s_count = r.shape
    build_start = time.perf_counter()
    model = gp.Model("DAgg_replay")
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
    lin_coeffs = []
    lin_vars = []
    for i in range(i_count):
        for t in range(t_count):
            lin_coeffs.append(float(p_da[t]))
            lin_vars.append(x[i, t])
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        lin_coeffs.extend(
            [
                float(p_rt[t, s] / s_count),
                float(-p_pn[t, s] / s_count),
                float(-lambda_dual[t, s]),
                float(lambda_dual[t, s]),
            ]
        )
        lin_vars.extend([yp[i, t, s], ym[i, t, s], dp[i, t, s], dm[i, t, s]])
    add_terms_chunked(obj, lin_coeffs, lin_vars)

    quad_coeffs = []
    quad_vars = []
    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        coeff = float(-eps / s_count)
        quad_coeffs.extend([coeff, coeff])
        quad_vars.extend([dp[i, t, s], dm[i, t, s]])
    add_terms_chunked(obj, quad_coeffs, quad_vars, quad_vars)
    model.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(i_count), range(t_count), range(s_count)):
        eta_e = ineff_ext[i]
        model.addConstr(
            r[i, t, s] - (1 / eta_e) * x[i, t]
            == (1 / eta_e) * yp[i, t, s]
            - eta_e * ym[i, t, s]
            + (1 / eta_e) * dp[i, t, s]
            - eta_e * dm[i, t, s]
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
        raise RuntimeError(f"DAgg replay failed with status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array([[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    ym_arr = np.array([[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dp_arr = np.array([[[dp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dm_arr = np.array([[[dm[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])

    return {
        "status": model.status,
        "objective": model.ObjVal,
        "runtime": model.Runtime,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "x": x_arr,
        "yp": yp_arr,
        "ym": ym_arr,
        "dp": dp_arr,
        "dm": dm_arr,
    }


def market_terms_usd(p_da: np.ndarray, p_rt: np.ndarray, p_pn: np.ndarray, x: np.ndarray, yp: np.ndarray, ym: np.ndarray) -> dict:
    da = float(np.sum(p_da[None, :] * x)) / 1000.0
    rt = float(np.mean(np.sum(p_rt[None, :, :] * yp, axis=(0, 1)))) / 1000.0
    penalty = float(np.mean(np.sum(p_pn[None, :, :] * ym, axis=(0, 1)))) / 1000.0
    return {
        "da_revenue_usd": da,
        "rt_revenue_usd": rt,
        "penalty_cost_usd": penalty,
        "market_profit_usd": da + rt - penalty,
    }


def imbalance_summary_usd(dp: np.ndarray, dm: np.ndarray, lambda_dual: np.ndarray, p_rt: np.ndarray) -> dict:
    i_count, t_count, s_count = dp.shape
    pin = -lambda_dual * s_count
    imbalance_kwh = np.sum(dm - dp, axis=0)
    imbalance_quantity_mwh = float(np.sum(np.mean(np.abs(imbalance_kwh), axis=1))) / 1000.0
    signed_by_ts_usd = imbalance_kwh * (pin - p_rt) / 1000.0
    signed_value_usd = float(np.sum(np.mean(signed_by_ts_usd, axis=1)))
    absolute_value_usd = float(np.sum(np.mean(np.abs(signed_by_ts_usd), axis=1)))
    cost_only_usd = float(np.sum(np.mean(np.maximum(-signed_by_ts_usd, 0.0), axis=1)))
    return {
        "imbalance_quantity_mwh": imbalance_quantity_mwh,
        "imbalance_penalty_usd": signed_value_usd,
        "imbalance_value_signed_usd": signed_value_usd,
        "imbalance_value_abs_usd": absolute_value_usd,
        "imbalance_penalty_cost_usd": cost_only_usd,
    }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(count: int, seed: int, files: list[str], args: argparse.Namespace, root: Path) -> dict:
    selected_files = files[:count]
    generation_data, i_count, t_count = load_generation_data(root, selected_files, args.date)
    r, p_rt, k, k0, crate, drate = load_parameters(
        generation_data, args.scenarios, args.level, seed, root
    )
    p_da, p_pn = load_price_data(root, p_rt)
    ineff_ext = np.full(i_count, args.ineff_ext)

    cagg = solve_cagg(
        r, k, k0, crate, drate, p_da, p_rt, p_pn, args.eps, args.ineff_batt,
        ineff_ext, args.time_limit, args.output_flag,
    )
    dagg = solve_dagg(
        r, k, k0, crate, drate, p_da, p_rt, p_pn, cagg["lambda_dual"], args.eps,
        args.ineff_batt, ineff_ext, args.time_limit, args.output_flag,
    )

    dec_terms = market_terms_usd(p_da, p_rt, p_pn, dagg["x"], dagg["yp"], dagg["ym"])
    cagg_terms = market_terms_usd(p_da, p_rt, p_pn, cagg["x"], cagg["yp"], cagg["ym"])
    imb = imbalance_summary_usd(dagg["dp"], dagg["dm"], cagg["lambda_dual"], p_rt)
    dec_profit_with_imbalance = dec_terms["market_profit_usd"] + imb["imbalance_penalty_usd"]

    return {
        "participant_count": i_count,
        "date": args.date,
        "seed": seed,
        "scenarios": args.scenarios,
        "level": args.level,
        "eps": args.eps,
        "python_version": sys.version.split()[0],
        "cagg_status": cagg["status"],
        "dagg_status": dagg["status"],
        "decentralized_aggregate_profit_usd": dec_terms["market_profit_usd"],
        "decentralized_market_profit_usd": dec_terms["market_profit_usd"],
        "decentralized_profit_with_imbalance_usd": dec_profit_with_imbalance,
        "decentralized_da_revenue_usd": dec_terms["da_revenue_usd"],
        "decentralized_rt_revenue_usd": dec_terms["rt_revenue_usd"],
        "decentralized_penalty_cost_usd": dec_terms["penalty_cost_usd"],
        "aggregate_imbalance_penalty_usd": imb["imbalance_penalty_usd"],
        "aggregate_imbalance_penalty_cost_usd": imb["imbalance_penalty_cost_usd"],
        "aggregate_imbalance_value_signed_usd": imb["imbalance_value_signed_usd"],
        "aggregate_imbalance_value_abs_usd": imb["imbalance_value_abs_usd"],
        "aggregate_imbalance_quantity_mwh": imb["imbalance_quantity_mwh"],
        "cagg_market_profit_usd": cagg_terms["market_profit_usd"],
        "cagg_solve_seconds": cagg["runtime"],
        "cagg_wall_solve_seconds": cagg["solve_seconds"],
        "cagg_build_seconds": cagg["build_seconds"],
        "cagg_total_seconds": cagg["build_seconds"] + cagg["solve_seconds"],
        "dagg_solve_seconds": dagg["runtime"],
        "dagg_wall_solve_seconds": dagg["solve_seconds"],
        "dagg_build_seconds": dagg["build_seconds"],
        "dagg_total_seconds": dagg["build_seconds"] + dagg["solve_seconds"],
        "selected_files_json": json.dumps(selected_files, ensure_ascii=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run REG-CAgg/DAgg scalability experiments.")
    parser.add_argument("--counts", nargs="+", type=int, default=DEFAULT_COUNTS)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None, help="Single seed override. Use --seeds for multiple runs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--level", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--date", default="2022-07-18")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--ineff-batt", type=float, default=0.95)
    parser.add_argument("--ineff-ext", type=float, default=0.99)
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--output", default="scalability_results.csv")
    parser.add_argument("--per-seed-dir", default="seed_results")
    parser.add_argument("--average-output", default="scalability_results_seed_average.csv")
    parser.add_argument("--summary-output", default="scalability_results_mean_std_summary.csv")
    parser.add_argument("--summary-markdown", default="scalability_results_mean_std_summary.md")
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--require-python", default=None, help="Fail unless the Python version matches this value, e.g. 3.12.2.")
    parser.add_argument("--output-flag", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def write_average(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        return

    group_cols = ["participant_count", "scenarios", "level", "eps"]
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"seed", *group_cols}
    ]
    avg = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    seed_count = df.groupby(group_cols, as_index=False)["seed"].nunique()
    seed_count = seed_count.rename(columns={"seed": "seed_count"})
    avg = seed_count.merge(avg, on=group_cols, how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg.to_csv(output_path, index=False, encoding="utf-8")


def mean_std_text(series: pd.Series, digits: int = 3) -> str:
    mean = series.mean()
    std = series.std(ddof=1)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_mean_std_summary(input_path: Path, csv_path: Path, markdown_path: Path) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        return

    metrics = {
        "Cagg (mean ± std)": "cagg_market_profit_usd",
        "Dagg (mean ± std)": "decentralized_market_profit_usd",
        "Imb Qty (mean ± std)": "aggregate_imbalance_quantity_mwh",
        "Pricing time (secs) (mean ± std)": "cagg_total_seconds",
        "Decentralized Solving time (secs) (mean ± std)": "dagg_total_seconds",
    }
    required = {"participant_count", "seed", *metrics.values()}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cannot write mean/std summary; missing columns: {sorted(missing)}")

    rows = []
    for participant_count, group in df.groupby("participant_count", sort=True):
        row = {"": int(participant_count), "replications": group["seed"].nunique()}
        for label, col in metrics.items():
            row[label] = mean_std_text(group[col])
        rows.append(row)

    summary = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    markdown_df = summary.drop(columns=["replications"])
    headers = list(markdown_df.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in markdown_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    markdown_path.write_text(
        f"Replications: {int(summary['replications'].min())}\n\n" + "\n".join(table_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.require_python and sys.version.split()[0] != args.require_python:
        raise RuntimeError(
            f"Python {args.require_python} is required, but this interpreter is {sys.version.split()[0]}."
        )

    seeds = [args.seed] if args.seed is not None else args.seeds
    root = repo_root()
    files = generation_files(root)
    if max(args.counts) > len(files):
        raise ValueError(f"Requested {max(args.counts)} participants, but only {len(files)} CSV files are available.")

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / args.output
    average_path = script_dir / args.average_output
    summary_path = script_dir / args.summary_output
    summary_markdown_path = script_dir / args.summary_markdown
    per_seed_dir = script_dir / args.per_seed_dir

    if args.reset_output:
        for path in [output_path, average_path, summary_path, summary_markdown_path]:
            if path.exists():
                path.unlink()
        if per_seed_dir.exists():
            for path in per_seed_dir.glob("*.csv"):
                path.unlink()

    print(f"Python {sys.version.split()[0]}")
    print(f"Using {len(files)} generation files; writing combined results to {output_path}")
    print(f"Seeds: {seeds}")

    for seed in seeds:
        seed_output = per_seed_dir / f"scalability_seed_{seed}.csv"
        if args.reset_output and seed_output.exists():
            seed_output.unlink()

        for count in args.counts:
            start = time.perf_counter()
            print(f"[start] seed={seed}, participant_count={count}")
            row = run_one(count, seed, files, args, root)
            row["experiment_wall_seconds"] = time.perf_counter() - start
            append_row(output_path, row)
            append_row(seed_output, row)
            print(
                "[done] "
                f"seed={seed}, N={count}, "
                f"dec_profit=${row['decentralized_market_profit_usd']:.2f}, "
                f"dec_profit_with_imb=${row['decentralized_profit_with_imbalance_usd']:.2f}, "
                f"cen_profit=${row['cagg_market_profit_usd']:.2f}, "
                f"imb_qty={row['aggregate_imbalance_quantity_mwh']:.3f}MWh, "
                f"imb_penalty=${row['aggregate_imbalance_penalty_usd']:.2f}, "
                f"cagg={row['cagg_total_seconds']:.2f}s, dagg={row['dagg_total_seconds']:.2f}s"
            )

    write_average(output_path, average_path)
    print(f"[average] wrote {average_path}")
    write_mean_std_summary(output_path, summary_path, summary_markdown_path)
    print(f"[summary] wrote {summary_path}")
    print(f"[summary] wrote {summary_markdown_path}")


if __name__ == "__main__":
    main()
