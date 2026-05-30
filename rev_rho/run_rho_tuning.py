"""Rho tuning experiment for the original CAgg/DAgg pricing model.

This script varies the quadratic regularization coefficient rho in paper
equation (3b), then reports:
- the regularization magnitude added to REG-CAgg,
- CAgg profit measured by the original paper objective (1a), and
- DAgg profit measured by the expected sum of individual profits in (5c).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB


DEFAULT_FILES = [
    "545.csv",
    "665.csv",
    "690.csv",
    "1033.csv",
    "1818.csv",
    "2502.csv",
    "2503.csv",
    "2634.csv",
    "2698.csv",
    "2816.csv",
]
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 25]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_generation_data(root: Path, include_files: list[str], date_filter: str) -> tuple[np.ndarray, int, int]:
    data_dir = root / "data" / "generation"
    t_count = 24
    generation_data = np.zeros((len(include_files), t_count))

    for idx, filename in enumerate(include_files):
        df = pd.read_csv(data_dir / filename)
        df.columns = df.columns.str.strip()
        df = df[df["Date"] == date_filter].copy()
        df = df[df["Hour (Eastern Time, Daylight-Adjusted)"].astype(str).str.match(r"^\d+$")]
        df["Time"] = df["Hour (Eastern Time, Daylight-Adjusted)"].astype(int)
        df = df[df["Time"].between(0, 23)]
        for t in range(t_count):
            vals = df.loc[df["Time"] == t, "Electricity Generated"].to_numpy()
            if vals.size:
                generation_data[idx, t] = float(vals[0])

    return generation_data, len(include_files), t_count


def generate_rt_scenarios(root: Path, scenarios: int, level: str, seed: int) -> np.ndarray:
    ny_rt = pd.read_csv(root / "data" / "price" / "20220718rt.csv")
    ny_rt["Time Stamp"] = pd.to_datetime(ny_rt["Time Stamp"])
    ny_rt = ny_rt[ny_rt["Name"] == "MHK VL"].copy()
    start = ny_rt["Time Stamp"].min().floor("D")
    ny_rt = ny_rt[(ny_rt["Time Stamp"] >= start) & (ny_rt["Time Stamp"] <= start + pd.Timedelta(hours=23))]
    ny_rt["Hour"] = ny_rt["Time Stamp"].dt.floor("h")
    price_hourly = ny_rt.groupby("Hour")["LBMP ($/MWHr)"].mean().to_numpy()

    ranges = {"low": (0.95, 1.05), "medium": (0.85, 1.15), "high": (0.4, 1.6)}
    rng = np.random.default_rng(seed)
    low, high = ranges[level]
    return price_hourly[:, None] * rng.uniform(low, high, size=(len(price_hourly), scenarios))


def generate_randomized_generation(
    generation_data: np.ndarray,
    scenarios: int,
    level: str,
    seed: int,
) -> np.ndarray:
    ranges = {"low": (0.8, 1.2), "medium": (0.5, 1.5), "high": (0.2, 1.8)}
    rng = np.random.default_rng(seed)
    low, high = ranges[level]
    return generation_data[:, :, None] * rng.uniform(low, high, size=(*generation_data.shape, scenarios))


def load_price_data(root: Path, p_rt: np.ndarray, scale_da: float = 1.5, scale_penalty: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    ny_da = pd.read_csv(root / "data" / "price" / "20220718da.csv")
    ny_da["Time Stamp"] = pd.to_datetime(ny_da["Time Stamp"])
    ny_da = ny_da[ny_da["Name"] == "MHK VL"].copy()
    p_da = ny_da["LBMP ($/MWHr)"].astype(float).to_numpy() * scale_da
    p_pn = np.maximum(p_da[:, None], p_rt) * scale_penalty
    return p_da, p_pn


def build_problem_data(
    scenarios: int,
    level: str,
    seed: int,
    date: str,
    include_files: list[str] | None = None,
) -> dict[str, Any]:
    root = repo_root()
    files = include_files or DEFAULT_FILES
    generation_data, i_count, t_count = load_generation_data(root, files, date)
    r = generate_randomized_generation(generation_data, scenarios, level, seed)
    p_rt = generate_rt_scenarios(root, scenarios, level, seed)
    p_da, p_pn = load_price_data(root, p_rt)

    peak_generations = np.mean(generation_data, axis=1)
    k = (peak_generations * 2 // 100) * 100
    return {
        "R": r,
        "P_DA": p_da,
        "P_RT": p_rt,
        "P_PN": p_pn,
        "K": k,
        "K0": np.zeros(i_count),
        "CRATE": k / 4,
        "DRATE": k / 4,
        "INEFF_BATT": 0.95,
        "INEFF_EXT": np.full(i_count, 0.99),
        "files": files,
        "date": date,
        "level": level,
        "seed": seed,
    }


def set_params(model: gp.Model, output_flag: int, time_limit: float | None) -> None:
    model.setParam("OutputFlag", output_flag)
    model.setParam("MIPGap", 1e-7)
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)


def expected_market_profit(data: dict[str, Any], x: np.ndarray, yp: np.ndarray, ym: np.ndarray) -> float:
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    da = float(np.sum(p_da[None, :] * x))
    rt = float(np.mean(np.sum(p_rt[None, :, :] * yp, axis=(0, 1))))
    penalty = float(np.mean(np.sum(p_pn[None, :, :] * ym, axis=(0, 1))))
    return da + rt - penalty


def quad_raw(dp: np.ndarray, dm: np.ndarray) -> float:
    return float(np.sum(dp * dp + dm * dm) / dp.shape[2])


def internal_settlement(internal_price: np.ndarray, dp: np.ndarray, dm: np.ndarray) -> float:
    return float(np.mean(np.sum(internal_price[None, :, :] * (dp - dm), axis=(0, 1))))


def add_der_constraints(
    model: gp.Model,
    data: dict[str, Any],
    x: gp.tupledict,
    yp: gp.tupledict,
    ym: gp.tupledict,
    dp: gp.tupledict,
    dm: gp.tupledict,
    z: gp.tupledict,
    zc: gp.tupledict,
    zd: gp.tupledict,
) -> None:
    r = data["R"]
    k = data["K"]
    k0 = data["K0"]
    crate = data["CRATE"]
    drate = data["DRATE"]
    ineff_batt = data["INEFF_BATT"]
    ineff_ext = data["INEFF_EXT"]
    i_count, t_count, s_count = r.shape

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
        model.addConstr(z[i, t + 1, s] == z[i, t, s] + ineff_batt * zc[i, t, s] - zd[i, t, s] / ineff_batt)

    for i, s in product(range(i_count), range(s_count)):
        model.addConstr(z[i, 0, s] == k0[i])


def solve_cagg(data: dict[str, Any], rho: float, output_flag: int, time_limit: float | None) -> dict[str, Any]:
    r = data["R"]
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    i_count, t_count, s_count = r.shape

    start = time.perf_counter()
    model = gp.Model("REG_CAgg")
    set_params(model, output_flag, time_limit)
    x = model.addVars(i_count, t_count, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="yp")
    ym = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="ym")
    dp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dp")
    dm = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dm")
    z = model.addVars(i_count, t_count + 1, s_count, vtype=GRB.CONTINUOUS, lb=0, name="z")
    zc = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zc")
    zd = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zd")

    obj = gp.quicksum(p_da[t] * x[i, t] for i in range(i_count) for t in range(t_count))
    obj += gp.quicksum(
        (p_rt[t, s] * yp[i, t, s] - p_pn[t, s] * ym[i, t, s]) / s_count
        for i, t, s in product(range(i_count), range(t_count), range(s_count))
    )
    if rho:
        obj -= gp.quicksum(
            (rho / s_count) * (dp[i, t, s] * dp[i, t, s] + dm[i, t, s] * dm[i, t, s])
            for i, t, s in product(range(i_count), range(t_count), range(s_count))
        )
    model.setObjective(obj, GRB.MAXIMIZE)
    add_der_constraints(model, data, x, yp, ym, dp, dm, z, zc, zd)

    balance = {}
    for t, s in product(range(t_count), range(s_count)):
        balance[t, s] = model.addConstr(
            gp.quicksum(dp[i, t, s] for i in range(i_count)) == gp.quicksum(dm[i, t, s] for i in range(i_count)),
            name=f"balance_{t}_{s}",
        )

    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"REG-CAgg failed for rho={rho} with status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array([[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    ym_arr = np.array([[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dp_arr = np.array([[[dp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dm_arr = np.array([[[dm[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    lambda_dual = np.array([[balance[t, s].Pi for s in range(s_count)] for t in range(t_count)])
    internal_price = -lambda_dual * s_count
    q_raw = quad_raw(dp_arr, dm_arr)
    profit_1a = expected_market_profit(data, x_arr, yp_arr, ym_arr)

    return {
        "status": model.status,
        "objective_3a": model.ObjVal,
        "profit_1a": profit_1a,
        "quad_raw": q_raw,
        "quad_reg_value": rho * q_raw,
        "internal_price": internal_price,
        "runtime_seconds": time.perf_counter() - start,
        "x": x_arr,
        "yp": yp_arr,
        "ym": ym_arr,
        "dp": dp_arr,
        "dm": dm_arr,
    }


def solve_dagg(data: dict[str, Any], internal_price: np.ndarray, rho: float, output_flag: int, time_limit: float | None) -> dict[str, Any]:
    r = data["R"]
    p_da = data["P_DA"]
    p_rt = data["P_RT"]
    p_pn = data["P_PN"]
    i_count, t_count, s_count = r.shape

    start = time.perf_counter()
    model = gp.Model("DAgg_replay")
    set_params(model, output_flag, time_limit)
    x = model.addVars(i_count, t_count, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="yp")
    ym = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="ym")
    dp = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dp")
    dm = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="dm")
    z = model.addVars(i_count, t_count + 1, s_count, vtype=GRB.CONTINUOUS, lb=0, name="z")
    zc = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zc")
    zd = model.addVars(i_count, t_count, s_count, vtype=GRB.CONTINUOUS, lb=0, name="zd")

    obj = gp.quicksum(p_da[t] * x[i, t] for i in range(i_count) for t in range(t_count))
    obj += gp.quicksum(
        (
            p_rt[t, s] * yp[i, t, s]
            - p_pn[t, s] * ym[i, t, s]
            + internal_price[t, s] * (dp[i, t, s] - dm[i, t, s])
        )
        / s_count
        for i, t, s in product(range(i_count), range(t_count), range(s_count))
    )
    if rho:
        obj -= gp.quicksum(
            (rho / s_count) * (dp[i, t, s] * dp[i, t, s] + dm[i, t, s] * dm[i, t, s])
            for i, t, s in product(range(i_count), range(t_count), range(s_count))
        )
    model.setObjective(obj, GRB.MAXIMIZE)
    add_der_constraints(model, data, x, yp, ym, dp, dm, z, zc, zd)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"DAgg replay failed for rho={rho} with status {model.status}")

    x_arr = np.array([[x[i, t].X for t in range(t_count)] for i in range(i_count)])
    yp_arr = np.array([[[yp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    ym_arr = np.array([[[ym[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dp_arr = np.array([[[dp[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    dm_arr = np.array([[[dm[i, t, s].X for s in range(s_count)] for t in range(t_count)] for i in range(i_count)])
    q_raw = quad_raw(dp_arr, dm_arr)
    market_profit = expected_market_profit(data, x_arr, yp_arr, ym_arr)
    settlement = internal_settlement(internal_price, dp_arr, dm_arr)
    allocation = imbalance_allocation_total(data, internal_price, dp_arr, dm_arr)

    return {
        "status": model.status,
        "objective_regularized": model.ObjVal,
        "market_profit": market_profit,
        "internal_settlement": settlement,
        "imbalance_allocation": allocation,
        "profit_sum_5c": market_profit + settlement + allocation,
        "quad_raw": q_raw,
        "quad_reg_value": rho * q_raw,
        "runtime_seconds": time.perf_counter() - start,
        "balance_abs_mwh": float(np.sum(np.mean(np.abs(np.sum(dm_arr - dp_arr, axis=0)), axis=1)) / 1000.0),
    }


def imbalance_allocation_total(data: dict[str, Any], internal_price: np.ndarray, dp: np.ndarray, dm: np.ndarray) -> float:
    p_rt = data["P_RT"]
    _, t_count, s_count = dp.shape
    allocation = 0.0
    for t, s in product(range(t_count), range(s_count)):
        imbalance = float(np.sum(dm[:, t, s]) - np.sum(dp[:, t, s]))
        loss = imbalance * float(internal_price[t, s] - p_rt[t, s])
        weights = internal_price[t, s] * (dp[:, t, s] + dm[:, t, s])
        denom = float(np.sum(weights))
        if abs(denom) > 1e-9:
            allocation += loss
    return allocation / s_count


def run_experiment(
    rho_values: list[float],
    scenarios: int,
    level: str,
    seeds: list[int],
    date: str,
    output_csv: Path,
    summary_md: Path,
    output_plot: Path | None,
    output_flag: int,
    time_limit: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        data = build_problem_data(scenarios=scenarios, level=level, seed=seed, date=date)
        baseline_cagg = None
        baseline_dagg = None

        for rho in rho_values:
            print(f"[rho tuning] seed={seed}, rho={rho:g}")
            cagg = solve_cagg(data, rho=rho, output_flag=output_flag, time_limit=time_limit)
            dagg = solve_dagg(data, cagg["internal_price"], rho=rho, output_flag=output_flag, time_limit=time_limit)
            if baseline_cagg is None:
                baseline_cagg = cagg["profit_1a"]
                baseline_dagg = dagg["profit_sum_5c"]

            row = {
                "rho": rho,
                "cagg_profit_1a": cagg["profit_1a"],
                "cagg_quad_raw": cagg["quad_raw"],
                "cagg_quad_reg_value": cagg["quad_reg_value"],
                "reg_cagg_objective_3a": cagg["objective_3a"],
                "dagg_profit_sum_5c": dagg["profit_sum_5c"],
                "dagg_market_profit": dagg["market_profit"],
                "dagg_internal_settlement": dagg["internal_settlement"],
                "dagg_imbalance_allocation": dagg["imbalance_allocation"],
                "dagg_quad_raw": dagg["quad_raw"],
                "dagg_quad_reg_value": dagg["quad_reg_value"],
                "cagg_profit_delta_vs_first": cagg["profit_1a"] - baseline_cagg,
                "dagg_profit_delta_vs_first": dagg["profit_sum_5c"] - baseline_dagg,
                "cagg_minus_dagg_profit": cagg["profit_1a"] - dagg["profit_sum_5c"],
                "dagg_balance_abs_mwh": dagg["balance_abs_mwh"],
                "internal_price_mean": float(np.mean(cagg["internal_price"])),
                "internal_price_std": float(np.std(cagg["internal_price"])),
                "internal_price_min": float(np.min(cagg["internal_price"])),
                "internal_price_max": float(np.max(cagg["internal_price"])),
                "cagg_status": cagg["status"],
                "dagg_status": dagg["status"],
                "cagg_runtime_seconds": cagg["runtime_seconds"],
                "dagg_runtime_seconds": dagg["runtime_seconds"],
                "scenarios": scenarios,
                "level": level,
                "seed": seed,
                "date": date,
                "files_json": json.dumps(data["files"]),
            }
            rows.append(row)
            df = pd.DataFrame(rows)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            write_mean_std_markdown(df, summary_md, seeds=seeds)
            print(
                f"  CAgg 1a={row['cagg_profit_1a']:.3f}, "
                f"DAgg sum5c={row['dagg_profit_sum_5c']:.3f}, "
                f"CAgg quad={row['cagg_quad_reg_value']:.6g}"
            )

    result = pd.DataFrame(rows)
    write_mean_std_markdown(result, summary_md, seeds=seeds)
    if output_plot is not None:
        write_plot(summarize_by_rho(result), output_plot)
    return result


def mean_std_text(series: pd.Series, digits: int = 3) -> str:
    return f"{series.mean():.{digits}f} ± {series.std(ddof=1):.{digits}f}"


def summarize_by_rho(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "cagg_quad_reg_value",
        "cagg_profit_1a",
        "dagg_profit_sum_5c",
        "cagg_profit_delta_vs_first",
        "dagg_profit_delta_vs_first",
        "cagg_minus_dagg_profit",
        "dagg_balance_abs_mwh",
        "internal_price_mean",
        "internal_price_std",
    ]
    rows = []
    for rho, group in df.groupby("rho", sort=True):
        row = {"rho": rho, "replications": group["seed"].nunique()}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_mean_std_markdown(df: pd.DataFrame, output_path: Path, seeds: list[int]) -> None:
    if df.empty:
        return

    metrics = {
        "Quad Reg (mean ± std)": "cagg_quad_reg_value",
        "CAgg Profit 1a (mean ± std)": "cagg_profit_1a",
        "DAgg Profit sum 5c (mean ± std)": "dagg_profit_sum_5c",
        "CAgg Δ vs rho0 (mean ± std)": "cagg_profit_delta_vs_first",
        "DAgg Δ vs rho0 (mean ± std)": "dagg_profit_delta_vs_first",
        "CAgg - DAgg (mean ± std)": "cagg_minus_dagg_profit",
        "DAgg Imb Qty MWh (mean ± std)": "dagg_balance_abs_mwh",
        "PIN std (mean ± std)": "internal_price_std",
    }
    rows = []
    for rho, group in df.groupby("rho", sort=True):
        row = {
            "rho": f"{rho:g}",
            "replications": group["seed"].nunique(),
        }
        for label, col in metrics.items():
            row[label] = mean_std_text(group[col])
        rows.append(row)

    summary = pd.DataFrame(rows)
    headers = list(summary.columns)
    lines = [
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(df: pd.DataFrame, output_plot: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_df = df.sort_values("rho")
    x = plot_df["rho"]

    axes[0].plot(x, plot_df["cagg_quad_reg_value_mean"], marker="o")
    axes[0].set_xscale("symlog", linthresh=1e-10)
    axes[0].set_title("Quadratic regularization")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("mean rho / S * sum(d^2)")

    axes[1].plot(x, plot_df["cagg_profit_1a_mean"], marker="o", label="CAgg 1a")
    axes[1].plot(x, plot_df["dagg_profit_sum_5c_mean"], marker="s", label="DAgg sum 5c")
    axes[1].set_xscale("symlog", linthresh=1e-10)
    axes[1].set_title("Mean economic profit")
    axes[1].set_xlabel("rho")
    axes[1].legend()

    axes[2].plot(x, plot_df["internal_price_std_mean"], marker="o")
    axes[2].set_xscale("symlog", linthresh=1e-10)
    axes[2].set_title("Internal price dispersion")
    axes[2].set_xlabel("rho")
    axes[2].set_ylabel("std(PIN)")

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-values", nargs="+", type=float, default=[0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--level", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--date", default="2022-07-18")
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).resolve().parent / "rho_tuning_results.csv")
    parser.add_argument("--summary-md", type=Path, default=Path(__file__).resolve().parent / "rho_tuning_results_mean_std.md")
    parser.add_argument("--output-plot", type=Path, default=Path(__file__).resolve().parent / "rho_tuning_results.png")
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--output-flag", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        rho_values=args.rho_values,
        scenarios=args.scenarios,
        level=args.level,
        seeds=args.seeds,
        date=args.date,
        output_csv=args.output_csv,
        summary_md=args.summary_md,
        output_plot=args.output_plot,
        output_flag=args.output_flag,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()
