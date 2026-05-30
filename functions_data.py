import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import os
from itertools import product


def load_parameters(I, T, generation_data, S, randomness_level, random_seed):
    R = generate_randomized_generation(
        I, T, S, generation_data, randomness_level, random_seed
    )
    P_RT = generate_rt_scenarios(S, randomness_level, random_seed)

    peak_generations = np.mean(generation_data, axis=1)
    storage_hours_factor = 2
    K = peak_generations * storage_hours_factor
    K = (K // 100) * 100

    K0 = np.full(I, 0)
    M1 = np.maximum(R, K[:, None, None]).max()
    M2 = max(R.sum(axis=0).max(), K.sum())
    CRATE = K / 4
    DRATE = K / 4

    print(
        f"✅ 시뮬레이션 초기화 완료: S={S}, Randomness='{randomness_level}', Random Seed={random_seed}, M1={M1:.2f}, M2={M2:.2f}"
    )
    print(f"   - 개별 K 값: {K}")

    return R, P_RT, K, K0, M1, M2, CRATE, DRATE


def load_generation_data(include_files=None, date_filter=None):
    if include_files is None:
        include_files = [
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
    data_dir = "/Users/jangseohyun/SynologyDrive/workspace/symply/DER/data/generation"
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    if include_files is not None:
        for file in include_files:
            if file not in all_files:
                raise ValueError(f"파일을 찾을 수 없습니다: {file}")
        files_to_load = [f for f in all_files if f in include_files]
    else:
        files_to_load = all_files

    I = len(files_to_load)
    T = 24
    generation_data = np.zeros((I, T))

    loaded_files = []

    for idx, file in enumerate(files_to_load):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        date_col = "Date"
        hour_col = "Hour (Eastern Time, Daylight-Adjusted)"
        gen_col = "Electricity Generated"

        if any(col not in df.columns for col in [date_col, hour_col, gen_col]):
            print(f"{file}: 필요한 컬럼 없음. 스킵됨.")
            continue

        if date_filter:
            df = df[df[date_col] == date_filter]
            if df.empty:
                print(f"{file}: {date_filter} 데이터 없음. 스킵됨.")
                continue

        df = df[pd.Series(df[hour_col]).astype(str).str.match(r"^\d+$")]
        df["Time"] = pd.Series(df[hour_col]).astype(int)
        df = df[pd.Series(df["Time"]).between(0, 23)]

        for t in range(T):
            if t in pd.Series(df["Time"]).values:
                generation_data[idx, t] = pd.Series(
                    df[df["Time"] == t][gen_col]
                ).values[0]

        loaded_files.append(file)

    print(f"✅ 총 {I}개 파일을 불러왔습니다: {', '.join(loaded_files)}")

    return generation_data, I, T


def load_price_data(P_RT, scale_da=1.5, scale_penalty=2, region="MHK VL"):
    ny_da = pd.read_csv("data/price/20220718da.csv")
    ny_da["Time Stamp"] = pd.to_datetime(ny_da["Time Stamp"])
    ny_da["Hour"] = pd.Series(ny_da["Time Stamp"]).dt.hour
    nyc_data = ny_da[ny_da["Name"] == region]

    P_DA = np.array(nyc_data["LBMP ($/MWHr)"].astype(float)) * float(scale_da)
    P_PN = np.maximum(P_DA[:, None], P_RT) * float(scale_penalty)
    return P_DA, P_PN


def generate_rt_scenarios(S, randomness_level, random_seed):
    ny_rt = pd.read_csv("data/price/20220718rt.csv")
    ny_rt["Time Stamp"] = pd.to_datetime(ny_rt["Time Stamp"])
    nyc_rt = ny_rt[ny_rt["Name"] == "MHK VL"].copy()

    start_of_day = nyc_rt["Time Stamp"].min().floor("D")
    end_of_day = start_of_day + pd.Timedelta(hours=23)
    nyc_rt = nyc_rt[
        (nyc_rt["Time Stamp"] >= start_of_day) & (nyc_rt["Time Stamp"] <= end_of_day)
    ]

    nyc_rt = pd.DataFrame(nyc_rt)
    nyc_rt["Hour"] = pd.Series(nyc_rt["Time Stamp"]).dt.floor("h")
    hourly_avg = nyc_rt.groupby("Hour")["LBMP ($/MWHr)"].mean().reset_index()
    price_hourly = hourly_avg["LBMP ($/MWHr)"].to_numpy()
    T = len(price_hourly)

    np.random.seed(random_seed)
    noise_ranges = {
        "low": (0.95, 1.05),
        "medium": (0.85, 1.15),
        "high": (0.4, 1.6),
    }

    if randomness_level not in noise_ranges:
        raise ValueError(
            "Invalid randomness level. Choose from 'low', 'medium', 'high'."
        )

    low, high = noise_ranges[randomness_level]
    noise_factors = np.random.uniform(low, high, size=(T, S))

    P_RT = np.expand_dims(price_hourly, axis=-1) * noise_factors

    return P_RT


def generate_randomized_generation(I, T, S, data, randomness_level, random_seed):
    np.random.seed(random_seed)

    noise_ranges = {
        "low": (0.8, 1.2),
        "medium": (0.5, 1.5),
        "high": (0.2, 1.8),
    }

    if randomness_level not in noise_ranges:
        raise ValueError(
            "Invalid randomness level. Please choose 'low', 'medium', or 'high'."
        )

    low, high = noise_ranges[randomness_level]
    noise_factors = np.random.uniform(low, high, size=(I, T, S))

    generation_r = np.expand_dims(data, axis=-1) * noise_factors
    # generation_r = np.round(generation_r).astype(int)

    print(f"📊 데이터 Shape: I={I}, T={T}, S={S}")
    return generation_r


def plot_generation_data(generation_data, I):
    hours = np.arange(24)
    plt.figure(figsize=(15, 9))

    for i in range(I):
        plt.plot(
            hours,
            generation_data[i],
            marker="o",
            linestyle="-",
            alpha=0.7,
            label=f"Generator {i}",
        )

    plt.xlabel("Hour")
    plt.ylabel("Electricity Generated (kWh)")
    plt.title("Hourly Electricity Generation for All Generators")
    plt.xticks(hours)  # 0~23 시간 설정
    plt.legend(loc="upper left", fontsize="small")

    plt.show()


def plot_randomized_generation(R, I, T, S):
    # plot_randomized_generation(R,1,T,7)
    hours = np.arange(T)

    plt.figure(figsize=(15, 9))

    for i in range(I):
        plt.plot(
            hours,
            R[i, :, S],
            marker="o",
            linestyle="-",
            alpha=0.7,
            label=f"Generator {i}",
        )

    plt.xlabel("Hour")
    plt.ylabel("Electricity Generated (kWh)")
    plt.title(f"Randomized Hourly Generation for Scenario {S}")
    plt.xticks(hours)
    plt.legend(loc="upper left")

    plt.show()


def plot_rt_scenarios(P_RT):
    # # plot_scenarios_for_generator(R,1)
    T, S = P_RT.shape
    hours = np.arange(0.5, T + 0.5)  # 0.5~23.5

    plt.figure(figsize=(10, 5))

    for s in range(S):
        plt.plot(hours, P_RT[:, s], linestyle="-", alpha=0.1)

    mean_curve = np.mean(P_RT, axis=1)
    plt.plot(hours, mean_curve, color="#142755", linewidth=3.5, label="Original Data")

    plt.xlabel("Hour", fontsize=14)
    plt.ylabel("Price ($/MWh)", fontsize=14)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25))
    plt.legend(loc="upper left", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_scenarios_for_generator(R, i):
    T = R.shape[1]
    S = R.shape[2]
    hours = np.arange(0.5, T + 0.5)  # 0.5~23.5

    plt.figure(figsize=(10, 5))

    for s in range(S):
        plt.plot(hours, R[i, :, s] * 1000, linestyle="-", alpha=0.1)
    mean_curve = np.mean(R[i, :, :] * 1000, axis=1)
    plt.plot(hours, mean_curve, color="#142755", linewidth=3.5, label="Original Data")

    plt.xlabel("Hour", fontsize=14)
    plt.ylabel("Electricity Generated (kW)", fontsize=14)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25))
    plt.legend(loc="upper left", fontsize=14)
    plt.tight_layout()
    plt.show()


# 시간별로 정규화
# hourly_contribution(x_vals)
# return : normalized data
def hourly_contribution(data):

    if len(data.shape) == 2:  # Case 1: [i,t]
        I, T = data.shape
        normalized_data = np.zeros((I, T))

        for t in range(T):
            total = np.sum(data[:, t])  # 각 시간별
            if total > 0:
                normalized_data[:, t] = data[:, t] / total  # 비율 계산
            else:
                normalized_data[:, t] = 1 / I

    elif len(data.shape) == 3:  # Case 2: [i,t,s]
        I, T, S = data.shape
        normalized_data = np.zeros((I, T))

        for t in range(T):
            scenario_mean = np.mean(data[:, t, :], axis=1)  # 각 발전기의 시나리오 평균
            total = np.sum(scenario_mean)  # 각 시간별 (시나리오 평균 기준)

            if total > 0:
                normalized_data[:, t] = scenario_mean / total  # 비율 계산
            else:
                normalized_data[:, t] = 1 / I

    else:
        raise ValueError("Input data must be of shape (I, T) or (I, T, S).")

    return normalized_data


# 정규화한걸 더해서 정규화하는게 아니라, raw data의 하루동안 합으로 정규화
# daily_contribution(x_vals)
# return : normalized data
def daily_contribution(data):

    if len(data.shape) == 2:  # Case 1: [i,t]
        I, T = data.shape
        daily_total = np.sum(data)  # 하루 동안 전체
        normalized_data = np.zeros(I)

        if daily_total > 0:
            normalized_data = np.sum(data, axis=1) / daily_total  # 하루 기여도 계산
        else:
            normalized_data[:] = 1 / I

    elif len(data.shape) == 3:  # Case 2: [i,t,s]
        I, T, S = data.shape
        normalized_data = np.zeros(I)

        scenario_mean = np.mean(data, axis=2)  # 시나리오 평균 계산 (I, T)
        daily_total = np.sum(scenario_mean)  # 하루 동안 합 (시나리오 평균 기준)

        if daily_total > 0:
            normalized_data = (
                np.sum(scenario_mean, axis=1) / daily_total
            )  # 하루 기여도 계산
        else:
            normalized_data[:] = 1 / I

    else:
        raise ValueError("Input data must be of shape (I, T) or (I, T, S).")

    return normalized_data


# remuneration_hourly, hourly_total = remuneration(hourly_contribution(x_vals), hourly_system_profit)
# (hourly) return: remuneration_amount(시간별), total_remuneration(시간별 합)
# (daily) return: remuneration_amount(하루치), total_remuneration(같음)
def remuneration(contribution, amount):

    # **Case 1: 시간별 정산**
    if len(contribution.shape) == 2 and len(amount.shape) == 1:
        I, T = contribution.shape
        if amount.shape[0] != T:
            raise ValueError(
                "Hourly amount (T,) should match contribution shape (I, T)."
            )

        # 시간별 정산액 = 시간별 기여도 * 시간별 총 분배금액
        remuneration_amount = contribution * amount.reshape(1, T)  # (I, T)

        # 각 발전기의 총 정산액 (t에 대한 합)
        total_remuneration = np.sum(remuneration_amount, axis=1)  # (I,)

    # **Case 2: 하루 단위 정산**
    elif len(contribution.shape) == 1 and len(amount.shape) == 0:
        I = contribution.shape[0]

        # 하루 단위 정산액 = 하루 기여도 * 하루 총 분배금액
        remuneration_amount = contribution * amount  # (I,)

        # 하루 단위 정산에서는 총합이 원래 하루 정산액과 동일
        total_remuneration = remuneration_amount.copy()  # (I,)

    else:
        raise ValueError(
            "Invalid input shapes. Expected (I,T) with (T,) or (I,) with (1,)."
        )

    return remuneration_amount, total_remuneration


# plot_hourly_contribution(hourly_contribution(x_vals), hourly_contribution(given_vals), labels=["x", "d"], selected_hours=[6, 7, 8, 9, 10, 11])
def plot_hourly_contribution(*hourly_contributions, labels=None, selected_hours=None):

    I, T = hourly_contributions[0].shape

    # 선택한 시간이 없으면 전체 24시간 사용
    if selected_hours is None:
        selected_hours = list(range(T))

    num_selected = len(selected_hours)
    num_rows = (num_selected // 6) + (
        1 if num_selected % 6 != 0 else 0
    )  # 필요한 행 개수 계산

    fig, axes = plt.subplots(
        num_rows,
        min(6, num_selected),
        figsize=(18, num_rows * 3),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).flatten()

    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(hourly_contributions))]

    # 선택한 시간대만 플롯
    for idx, t in enumerate(selected_hours):
        for i, data in enumerate(hourly_contributions):
            axes[idx].plot(
                range(I), data[:, t] * 100, marker="o", linestyle="-", label=labels[i]
            )
            axes[idx].set_title(f"Hour {t}")
            axes[idx].set_xticks(range(I))
            axes[idx].set_ylabel("Contribution (%)")

    # Add a single legend for all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1))

    plt.tight_layout(rect=(0, 0, 0.95, 1))  # Adjust layout to fit legend
    plt.show()


# plot_daily_contribution(daily_contribution(x_vals), daily_contribution(given_vals), labels=["x", "d"])
def plot_daily_contribution(*daily_contributions, labels=None):
    I = len(daily_contributions[0])
    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(daily_contributions))]

    for i, data in enumerate(daily_contributions):
        plt.plot(range(I), data * 100, marker="o", linestyle="-", label=labels[i])

    plt.xlabel("Generator Index")
    plt.ylabel("Daily Contribution (%)")
    plt.title("Daily Contribution Rate")
    plt.xticks(range(I))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


# plot_hourly_remuneration(remuneration_hourly, remuneration_hourly1, labels=["x", "d"], selected_hours=[6, 7, 8, 9, 10, 11])
def plot_hourly_remuneration(*hourly_remunerations, labels=None, selected_hours=None):

    only_hourly_remuneration_df = pd.read_csv(
        "/Users/jangseohyun/Documents/workspace/symply/DER/result/result_only_hourly_profit.csv"
    )
    only_hourly_remuneration = only_hourly_remuneration_df.pivot(
        index="DER", columns="Hour", values="hourly_total"
    ).values

    I, T = hourly_remunerations[0].shape

    # 선택한 시간이 없으면 전체 24시간 사용
    if selected_hours is None:
        selected_hours = list(range(T))

    num_selected = len(selected_hours)
    num_rows = (num_selected // 6) + (
        1 if num_selected % 6 != 0 else 0
    )  # 필요한 행 개수 계산

    fig, axes = plt.subplots(
        num_rows,
        min(6, num_selected),
        figsize=(18, num_rows * 3),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).flatten()

    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(hourly_remunerations))]

    # 선택한 시간대만 플롯
    for idx, t in enumerate(selected_hours):
        for i, data in enumerate(hourly_remunerations):
            axes[idx].plot(
                range(I), data[:, t], marker="o", linestyle="-", label=labels[i]
            )
        axes[idx].plot(
            range(I),
            only_hourly_remuneration[:, t],
            marker="s",
            linestyle="-",
            label="Base",
            color="#3e3e3e",
        )
        axes[idx].set_title(f"Hour {t}")
        axes[idx].set_xticks(range(I))
        axes[idx].set_ylabel("Remuneration ($)")

    # Add a single legend for all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1))

    plt.tight_layout(rect=(0, 0, 0.95, 1))  # Adjust layout to fit legend
    plt.show()


# plot_daily_remuneration(remuneration_daily, remuneration_daily1, labels=["x", "d"])
def plot_daily_remuneration(*daily_remunerations, labels=None):
    """
    하루별 정산액을 선 그래프로 플랏 (하나의 plot)
    x축: 발전기 index, y축: daily remuneration amount ($)
    """
    I = len(daily_remunerations[0])
    only_daily_remuneration = pd.read_csv(
        "/Users/jangseohyun/Documents/workspace/symply/DER/result/result_only_profit.csv"
    )
    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(daily_remunerations))]

    for i, data in enumerate(daily_remunerations):
        plt.plot(range(I), data, marker="o", linestyle="-", label=labels[i])

    plt.plot(
        range(I),
        only_daily_remuneration,
        marker="s",
        linestyle="--",
        color="#3e3e3e",
        label="Base",
    )
    plt.xlabel("Generator Index")
    plt.ylabel("Daily Remuneration ($)")
    plt.title("Daily Remuneration Amount")
    plt.xticks(range(I))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))  # 오른쪽 바깥에 배치
    plt.tight_layout(rect=(0, 0, 1.11, 1))
    plt.show()


def plot_summary(model, K, P_DA, P_RT, P_PN, a_vals, bp_vals, bm_vals, g_vals, s=0):
    T = len(P_DA)
    S = P_RT.shape[1]

    da_profit = sum(P_DA[t] * a_vals[t] for t in range(T))
    rt_profit = sum(
        P_RT[t, s_] * bp_vals[t, s_] / S for t in range(T) for s_ in range(S)
    )
    pn_cost = sum(P_PN[t] * bm_vals[t, s_] / S for t in range(T) for s_ in range(S))
    total_profit = da_profit + rt_profit - pn_cost

    print(f"DA Profit      = {da_profit:.2f}")
    print(f"RT Profit      = {rt_profit:.2f}")
    print(f"Penalty Cost   = {pn_cost:.2f}")
    print(f"Total Profit   = {total_profit:.2f}, Objective Val  = {model.ObjVal:.2f}")

    hours = np.arange(T)
    hours_g = np.arange(T + 1)

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    total_commitment = np.sum(a_vals)
    axs[0].bar(
        hours, a_vals, color="#6DE9FF", label=f"α (Total: {total_commitment:.2f})"
    )
    axs[0].set_title("Total Day-Ahead Commitment Over Time")
    axs[0].set_xlabel("Hour")
    axs[0].set_ylabel("Total x")
    axs[0].set_ylim(0, 2000)
    axs[0].legend()
    axs[0].grid(True, axis="y", ls="--")

    axs[1].step(
        hours_g,
        g_vals[: T + 1, s],
        where="post",
        label=f"SoC (Scen {s})",
        color="#00FF88",
        linewidth=2,
    )
    axs[1].set_title(f"Battery Charging/Discharging & SoC (Scenario {s})")
    axs[1].set_xlabel("Hour")
    axs[1].set_ylabel("Energy (kWh)")
    axs[1].set_xticks(np.arange(T + 1))
    axs[1].set_ylim(-10, sum(K) + 30)
    axs[1].legend()
    axs[1].grid(True, linestyle="--")
    axs[1].fill_between(
        hours_g, g_vals[: T + 1, s], alpha=0.3, step="post", color="#00FF88"
    )

    plt.tight_layout()
    plt.show()


def save_scenario_data(R, P_DA, P_RT, P_PN, I, T, S, seed, randomness_level, base_dir):
    save_dir = os.path.join(
        base_dir, f"i_{I}_s_{S}", f"seed_{seed}_level_{randomness_level}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"🔄 시나리오 데이터 저장 중... (폴더: {save_dir})")

    R_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        R_data.append({"i": i, "t": t, "s": s, "R": R[i, t, s]})
    pd.DataFrame(R_data).to_csv(f"{save_dir}/R.csv", index=False)
    print("✅ R.csv 저장 완료")

    P_DA_data = []
    for t in range(T):
        P_DA_data.append({"t": t, "P_DA": P_DA[t]})
    pd.DataFrame(P_DA_data).to_csv(f"{save_dir}/P_DA.csv", index=False)
    print("✅ P_DA.csv 저장 완료")

    P_RT_data = []
    for t, s in product(range(T), range(S)):
        P_RT_data.append({"t": t, "s": s, "P_RT": P_RT[t, s]})
    pd.DataFrame(P_RT_data).to_csv(f"{save_dir}/P_RT.csv", index=False)
    print("✅ P_RT.csv 저장 완료")

    P_PN_data = []
    for t, s in product(range(T), range(S)):
        P_PN_data.append({"t": t, "s": s, "P_PN": P_PN[t, s]})
    pd.DataFrame(P_PN_data).to_csv(f"{save_dir}/P_PN.csv", index=False)
    print("✅ P_PN.csv 저장 완료")

    print(f"🎉 모든 시나리오 데이터가 '{save_dir}' 폴더에 저장되었습니다!")
    print(f"📁 총 4개 파일 생성")

    return save_dir


def save_holistic_results(
    x_hol,
    a_hol,
    yp_hol,
    ym_hol,
    z_hol,
    zc_hol,
    zd_hol,
    ep_hol,
    bp_hol,
    em_hol,
    bm_hol,
    d_hol,
    dp_hol,
    dm_hol,
    obj_hol,
    I,
    T,
    S,
    seed,
    randomness_level,
    base_dir,
):

    save_dir = os.path.join(
        base_dir, f"i_{I}_s_{S}", f"seed_{seed}_level_{randomness_level}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"🔄 최적화 결과 저장 중... (폴더: {save_dir})")

    x_data = []
    for i, t in product(range(I), range(T)):
        x_data.append({"i": i, "t": t, "x_hol": x_hol[i, t]})
    pd.DataFrame(x_data).to_csv(f"{save_dir}/x_hol.csv", index=False)
    print("✅ x_hol.csv 저장 완료")

    a_data = []
    for t in range(T):
        a_data.append({"t": t, "a_hol": a_hol[t]})
    pd.DataFrame(a_data).to_csv(f"{save_dir}/a_hol.csv", index=False)
    print("✅ a_hol.csv 저장 완료")

    yp_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        yp_data.append({"i": i, "t": t, "s": s, "yp_hol": yp_hol[i, t, s]})
    pd.DataFrame(yp_data).to_csv(f"{save_dir}/yp_hol.csv", index=False)
    print("✅ yp_hol.csv 저장 완료")

    ym_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        ym_data.append({"i": i, "t": t, "s": s, "ym_hol": ym_hol[i, t, s]})
    pd.DataFrame(ym_data).to_csv(f"{save_dir}/ym_hol.csv", index=False)
    print("✅ ym_hol.csv 저장 완료")

    z_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        z_data.append({"i": i, "t": t, "s": s, "z_hol": z_hol[i, t, s]})
    pd.DataFrame(z_data).to_csv(f"{save_dir}/z_hol.csv", index=False)
    print("✅ z_hol.csv 저장 완료")

    zc_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        zc_data.append({"i": i, "t": t, "s": s, "zc_hol": zc_hol[i, t, s]})
    pd.DataFrame(zc_data).to_csv(f"{save_dir}/zc_hol.csv", index=False)
    print("✅ zc_hol.csv 저장 완료")

    zd_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        zd_data.append({"i": i, "t": t, "s": s, "zd_hol": zd_hol[i, t, s]})
    pd.DataFrame(zd_data).to_csv(f"{save_dir}/zd_hol.csv", index=False)
    print("✅ zd_hol.csv 저장 완료")

    ep_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        ep_data.append({"i": i, "t": t, "s": s, "ep_hol": ep_hol[i, t, s]})
    pd.DataFrame(ep_data).to_csv(f"{save_dir}/ep_hol.csv", index=False)
    print("✅ ep_hol.csv 저장 완료")

    bp_data = []
    for t, s in product(range(T), range(S)):
        bp_data.append({"t": t, "s": s, "bp_hol": bp_hol[t, s]})
    pd.DataFrame(bp_data).to_csv(f"{save_dir}/bp_hol.csv", index=False)
    print("✅ bp_hol.csv 저장 완료")

    em_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        em_data.append({"i": i, "t": t, "s": s, "em_hol": em_hol[i, t, s]})
    pd.DataFrame(em_data).to_csv(f"{save_dir}/em_hol.csv", index=False)
    print("✅ em_hol.csv 저장 완료")

    bm_data = []
    for t, s in product(range(T), range(S)):
        bm_data.append({"t": t, "s": s, "bm_hol": bm_hol[t, s]})
    pd.DataFrame(bm_data).to_csv(f"{save_dir}/bm_hol.csv", index=False)
    print("✅ bm_hol.csv 저장 완료")

    d_data = []
    for i, j, t, s in product(range(I), range(I), range(T), range(S)):
        if i != j:
            d_data.append({"i": i, "j": j, "t": t, "s": s, "d_hol": d_hol[i, j, t, s]})
    pd.DataFrame(d_data).to_csv(f"{save_dir}/d_hol.csv", index=False)
    print("✅ d_hol.csv 저장 완료")

    dp_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        dp_data.append({"i": i, "t": t, "s": s, "dp_hol": dp_hol[i, t, s]})
    pd.DataFrame(dp_data).to_csv(f"{save_dir}/dp_hol.csv", index=False)
    print("✅ dp_hol.csv 저장 완료")

    dm_data = []
    for i, t, s in product(range(I), range(T), range(S)):
        dm_data.append({"i": i, "t": t, "s": s, "dm_hol": dm_hol[i, t, s]})
    pd.DataFrame(dm_data).to_csv(f"{save_dir}/dm_hol.csv", index=False)
    print("✅ dm_hol.csv 저장 완료")

    obj_data = [
        {
            "obj_hol": obj_hol,
            "I": I,
            "T": T,
            "S": S,
            "seed": seed,
            "randomness_level": randomness_level,
        }
    ]
    pd.DataFrame(obj_data).to_csv(f"{save_dir}/obj_hol.csv", index=False)
    print("✅ obj_hol.csv 저장 완료")

    print(f"🎉 모든 변수가 '{save_dir}' 폴더에 저장되었습니다!")
    print(f"📁 총 {15}개 파일 생성")

    return save_dir


def load_scenario_data(I, T, S, seed, randomness_level, base_dir):
    save_dir = os.path.join(
        base_dir, f"i_{I}_s_{S}", f"seed_{seed}_level_{randomness_level}"
    )
    print(f"🔄 시나리오 데이터 불러오는 중... (폴더: {save_dir})")

    results = {}

    # R 불러오기 (I × T × S)
    R_df = pd.read_csv(f"{save_dir}/R.csv")
    R = np.zeros((I, T, S))
    for _, row in R_df.iterrows():
        R[int(row["i"]), int(row["t"]), int(row["s"])] = row["R"]
    results["R"] = R
    print("✅ R 불러오기 완료")

    # P_DA 불러오기 (T,)
    P_DA_df = pd.read_csv(f"{save_dir}/P_DA.csv")
    P_DA = np.zeros(T)
    for _, row in P_DA_df.iterrows():
        P_DA[int(row["t"])] = row["P_DA"]
    results["P_DA"] = P_DA
    print("✅ P_DA 불러오기 완료")

    # P_RT 불러오기 (T × S)
    P_RT_df = pd.read_csv(f"{save_dir}/P_RT.csv")
    P_RT = np.zeros((T, S))
    for _, row in P_RT_df.iterrows():
        P_RT[int(row["t"]), int(row["s"])] = row["P_RT"]
    results["P_RT"] = P_RT
    print("✅ P_RT 불러오기 완료")

    # P_PN 불러오기 (T × S)
    P_PN_df = pd.read_csv(f"{save_dir}/P_PN.csv")
    P_PN = np.zeros((T, S))
    for _, row in P_PN_df.iterrows():
        P_PN[int(row["t"]), int(row["s"])] = row["P_PN"]
    results["P_PN"] = P_PN
    print("✅ P_PN 불러오기 완료")

    print(f"🎉 모든 시나리오 데이터 불러오기 완료!")
    return results


def load_single_batch(I, T, S, seed, randomness_level, base_dir):
    save_dir = os.path.join(
        base_dir, f"i_{I}_s_{S}", f"seed_{seed}_level_{randomness_level}"
    )
    print(f"🔄 단일 배치 불러오는 중... (폴더: {save_dir})")

    results = {}

    # Scenario data
    R_df = pd.read_csv(f"{save_dir}/R.csv")
    R = np.zeros((I, T, S))
    for _, row in R_df.iterrows():
        R[int(row["i"]), int(row["t"]), int(row["s"])] = row["R"]
    results["R"] = R

    P_DA_df = pd.read_csv(f"{save_dir}/P_DA.csv")
    P_DA = np.zeros(T)
    for _, row in P_DA_df.iterrows():
        P_DA[int(row["t"])] = row["P_DA"]
    results["P_DA"] = P_DA

    P_RT_df = pd.read_csv(f"{save_dir}/P_RT.csv")
    P_RT = np.zeros((T, S))
    for _, row in P_RT_df.iterrows():
        P_RT[int(row["t"]), int(row["s"])] = row["P_RT"]
    results["P_RT"] = P_RT

    P_PN_df = pd.read_csv(f"{save_dir}/P_PN.csv")
    P_PN = np.zeros((T, S))
    for _, row in P_PN_df.iterrows():
        P_PN[int(row["t"]), int(row["s"])] = row["P_PN"]
    results["P_PN"] = P_PN

    # Optimization results
    x_df = pd.read_csv(f"{save_dir}/x_hol.csv")
    x_hol = np.zeros((I, T))
    for _, row in x_df.iterrows():
        x_hol[int(row["i"]), int(row["t"])] = row["x_hol"]
    results["x_hol"] = x_hol

    a_df = pd.read_csv(f"{save_dir}/a_hol.csv")
    a_hol = np.zeros(T)
    for _, row in a_df.iterrows():
        a_hol[int(row["t"])] = row["a_hol"]
    results["a_hol"] = a_hol

    yp_df = pd.read_csv(f"{save_dir}/yp_hol.csv")
    yp_hol = np.zeros((I, T, S))
    for _, row in yp_df.iterrows():
        yp_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["yp_hol"]
    results["yp_hol"] = yp_hol

    ym_df = pd.read_csv(f"{save_dir}/ym_hol.csv")
    ym_hol = np.zeros((I, T, S))
    for _, row in ym_df.iterrows():
        ym_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["ym_hol"]
    results["ym_hol"] = ym_hol

    z_df = pd.read_csv(f"{save_dir}/z_hol.csv")
    z_hol = np.zeros((I, T, S))
    for _, row in z_df.iterrows():
        z_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["z_hol"]
    results["z_hol"] = z_hol

    zc_df = pd.read_csv(f"{save_dir}/zc_hol.csv")
    zc_hol = np.zeros((I, T, S))
    for _, row in zc_df.iterrows():
        zc_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["zc_hol"]
    results["zc_hol"] = zc_hol

    zd_df = pd.read_csv(f"{save_dir}/zd_hol.csv")
    zd_hol = np.zeros((I, T, S))
    for _, row in zd_df.iterrows():
        zd_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["zd_hol"]
    results["zd_hol"] = zd_hol

    ep_df = pd.read_csv(f"{save_dir}/ep_hol.csv")
    ep_hol = np.zeros((I, T, S))
    for _, row in ep_df.iterrows():
        ep_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["ep_hol"]
    results["ep_hol"] = ep_hol

    bp_df = pd.read_csv(f"{save_dir}/bp_hol.csv")
    bp_hol = np.zeros((T, S))
    for _, row in bp_df.iterrows():
        bp_hol[int(row["t"]), int(row["s"])] = row["bp_hol"]
    results["bp_hol"] = bp_hol

    em_df = pd.read_csv(f"{save_dir}/em_hol.csv")
    em_hol = np.zeros((I, T, S))
    for _, row in em_df.iterrows():
        em_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["em_hol"]
    results["em_hol"] = em_hol

    bm_df = pd.read_csv(f"{save_dir}/bm_hol.csv")
    bm_hol = np.zeros((T, S))
    for _, row in bm_df.iterrows():
        bm_hol[int(row["t"]), int(row["s"])] = row["bm_hol"]
    results["bm_hol"] = bm_hol

    d_df = pd.read_csv(f"{save_dir}/d_hol.csv")
    d_hol = np.zeros((I, I, T, S))
    for _, row in d_df.iterrows():
        d_hol[int(row["i"]), int(row["j"]), int(row["t"]), int(row["s"])] = row["d_hol"]
    results["d_hol"] = d_hol

    dp_df = pd.read_csv(f"{save_dir}/dp_hol.csv")
    dp_hol = np.zeros((I, T, S))
    for _, row in dp_df.iterrows():
        dp_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["dp_hol"]
    results["dp_hol"] = dp_hol

    dm_df = pd.read_csv(f"{save_dir}/dm_hol.csv")
    dm_hol = np.zeros((I, T, S))
    for _, row in dm_df.iterrows():
        dm_hol[int(row["i"]), int(row["t"]), int(row["s"])] = row["dm_hol"]
    results["dm_hol"] = dm_hol

    obj_df = pd.read_csv(f"{save_dir}/obj_hol.csv")
    obj_hol = obj_df["obj_hol"].iloc[0]
    results["obj_hol"] = obj_hol

    print(f"🎉 단일 배치 불러오기 완료!")
    return results


def load_batches(I, S, seed_list=None, level_list=None, base_dir=None):
    parent_dir = os.path.join(base_dir, f"i_{I}_s_{S}")
    print(f"🔄 필터링된 배치 불러오는 중... (폴더: {parent_dir})")

    filtered_results = {}

    if not os.path.exists(parent_dir):
        print(f"❌ 폴더가 존재하지 않습니다: {parent_dir}")
        return filtered_results

    for subdir in os.listdir(parent_dir):
        if subdir.startswith("seed_") and "_level_" in subdir:
            parts = subdir.split("_")
            seed = int(parts[1])
            level = parts[3]

            if seed_list is not None and seed not in seed_list:
                continue
            if level_list is not None and level not in level_list:
                continue

            print(f"  📂 {subdir} 로딩 중...")
            batch_results = load_single_batch(I, 24, S, seed, level, base_dir)
            filtered_results[f"seed_{seed}_level_{level}"] = batch_results

    print(f"🎉 총 {len(filtered_results)}개 배치 불러오기 완료!")
    return filtered_results


def get_available_batches(I, S, base_dir):
    parent_dir = os.path.join(base_dir, f"i_{I}_s_{S}")
    available_batches = []

    if not os.path.exists(parent_dir):
        return available_batches

    for subdir in os.listdir(parent_dir):
        if subdir.startswith("seed_") and "_level_" in subdir:
            parts = subdir.split("_")
            seed = int(parts[1])
            level = parts[3]
            available_batches.append((seed, level))

    return available_batches


def check_batch_exists(I, S, seed, randomness_level, base_dir):
    save_dir = os.path.join(
        base_dir, f"i_{I}_s_{S}", f"seed_{seed}_level_{randomness_level}"
    )
    if not os.path.exists(save_dir):
        return False, f"폴더가 존재하지 않습니다: {save_dir}"

    required_files = [
        "x_hol.csv",
        "a_hol.csv",
        "yp_hol.csv",
        "ym_hol.csv",
        "z_hol.csv",
        "zc_hol.csv",
        "zd_hol.csv",
        "ep_hol.csv",
        "bp_hol.csv",
        "em_hol.csv",
        "bm_hol.csv",
        "d_hol.csv",
        "dp_hol.csv",
        "dm_hol.csv",
        "obj_hol.csv",
        "R.csv",
        "P_DA.csv",
        "P_RT.csv",
        "P_PN.csv",
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(save_dir, file)):
            missing_files.append(file)

    if missing_files:
        return False, f"누락된 파일: {missing_files}"
    else:
        return True, "모든 파일이 존재합니다"
