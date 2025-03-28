import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import os

def load_parameters(I, T, generation_data):
    S=5
    randomness_level="high"
    R = generate_randomized_generation(I, T, S, generation_data, randomness_level)
    P_RT = generate_rt_scenarios(S, randomness_level)
    K = np.full(I, 100)
    K0 = np.full(I, 10)
    M1 = np.maximum(R, K[:, None, None]).max()
    M2 = max(R.sum(axis=0).max(), K.sum())

    print(f"✅ 시뮬레이션 초기화 완료: S={S}, Randomness='{randomness_level}', M1={M1:.2f}, M2={M2:.2f}")
    return S, R, P_RT, K, K0, M1, M2

def load_generation_data(include_files=None, date_filter=None):
    if include_files is None:
        include_files = ['1201.csv', '137.csv', '401.csv', '89.csv']
        # include_files = ['1201.csv', '137.csv', '401.csv', '524.csv', '89.csv']
        # include_files = ['1201.csv', '137.csv', '281.csv', '397.csv', '401.csv', '430.csv', '514.csv', '524.csv', '775.csv', '89.csv']        
    data_dir = "/Users/jangseohyun/Documents/workspace/symply/DER/data/generation"
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

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

        df = df[df[hour_col].astype(str).str.match(r'^\d+$')]
        df["Time"] = df[hour_col].astype(int)
        df = df[df["Time"].between(0, 23)]

        for t in range(T):
            if t in df["Time"].values:
                generation_data[idx, t] = df[df["Time"] == t][gen_col].values[0]

        loaded_files.append(file)

    print(f"✅ 총 {I}개 파일을 불러왔습니다: {', '.join(loaded_files)}")

    return generation_data, I, T

def load_price_data(scale_da=1.3, scale_penalty=1.5, region="N.Y.C."):
    ny_da = pd.read_csv("data/price/20220718da.csv")
    ny_da["Time Stamp"] = pd.to_datetime(ny_da["Time Stamp"])
    ny_da["Hour"] = ny_da["Time Stamp"].dt.hour
    nyc_data = ny_da[ny_da["Name"] == region]
    
    P_DA = nyc_data["LBMP ($/MWHr)"].astype(float).to_numpy() * float(scale_da)
    P_PN = P_DA * float(scale_penalty)
    return P_DA, P_PN

def generate_rt_scenarios(S, randomness_level):
    ny_rt = pd.read_csv("data/price/20220718rt.csv")
    ny_rt["Time Stamp"] = pd.to_datetime(ny_rt["Time Stamp"])
    nyc_rt = ny_rt[ny_rt["Name"] == "N.Y.C."].copy() 

    # Extract the start of the day and filter only the first 24 hours
    start_of_day = nyc_rt["Time Stamp"].min().floor("D")
    end_of_day = start_of_day + pd.Timedelta(hours=23)
    nyc_rt = nyc_rt[(nyc_rt["Time Stamp"] >= start_of_day) & (nyc_rt["Time Stamp"] <= end_of_day)]

    nyc_rt["Hour"] = nyc_rt["Time Stamp"].dt.floor("H")
    hourly_avg = nyc_rt.groupby("Hour")["LBMP ($/MWHr)"].mean().reset_index()
    price_hourly = hourly_avg["LBMP ($/MWHr)"].to_numpy()
    T = len(price_hourly)

    np.random.seed(11)
    noise_ranges = {
        "low": (0.95, 1.05),
        "medium": (0.85, 1.15),
        "high": (0.7, 1.3),
    }

    if randomness_level not in noise_ranges:
        raise ValueError("Invalid randomness level. Choose from 'low', 'medium', 'high'.")

    low, high = noise_ranges[randomness_level]
    noise_factors = np.random.uniform(low, high, size=(T, S))

    P_RT = np.expand_dims(price_hourly, axis=-1) * noise_factors

    return P_RT

def generate_randomized_generation(I, T, S, data, randomness_level):
    np.random.seed(1)

    noise_ranges = {
        "low": (0.8, 1.2),
        "medium": (0.5, 1.5),
        "high": (0.2, 1.8),
    }

    if randomness_level not in noise_ranges:
        raise ValueError("Invalid randomness level. Please choose 'low', 'medium', or 'high'.")

    low, high = noise_ranges[randomness_level]
    noise_factors = np.random.uniform(low, high, size=(I, T, S))

    generation_r = np.expand_dims(data, axis=-1) * noise_factors
    
    print(f"📊 데이터 Shape: I={I}, T={T}, S={S}")
    return generation_r

def plot_generation_data(generation_data, I):
    hours = np.arange(24)
    plt.figure(figsize=(15, 9))

    for i in range(I):
        plt.plot(hours, generation_data[i], marker='o', linestyle='-', alpha=0.7, label=f'Generator {i}')

    plt.xlabel("Hour")
    plt.ylabel("Electricity Generated (kWh)")
    plt.title("Hourly Electricity Generation for All Generators")
    plt.xticks(hours)  # 0~23 시간 설정
    plt.legend(loc="upper left", fontsize='small')

    plt.show()

def plot_randomized_generation(R, I, T, S):
# plot_randomized_generation(R,1,T,7)
    hours = np.arange(T)
    
    plt.figure(figsize=(15, 9))

    for i in range(I):
        plt.plot(hours, R[i, :, S], marker='o', linestyle='-', alpha=0.7, label=f'Generator {i}')

    plt.xlabel("Hour")
    plt.ylabel("Electricity Generated (kWh)")
    plt.title(f"Randomized Hourly Generation for Scenario {S}")
    plt.xticks(hours) 
    plt.legend(loc="upper left") 

    plt.show()
       
def plot_scenarios_for_generator(R, i):
# plot_scenarios_for_generator(R,1)
    T = R.shape[1]
    S = R.shape[2] 
    hours = np.arange(T) 

    plt.figure(figsize=(15, 9))

    for s in range(S):
        plt.plot(hours, R[i, :, s], linestyle='-', alpha=0.6, label=f'Scenario {s+1}')

    plt.xlabel("Hour")
    plt.ylabel("Electricity Generated (kWh)")
    plt.title(f"Hourly Electricity Generation for Generator {i} Across All Scenarios")
    plt.xticks(hours)
    plt.legend(loc="upper left", fontsize='small', ncol=2)
    plt.show()

def plot_rt_scenarios(P_RT):
    T, S = P_RT.shape
    hours = np.arange(T)

    plt.figure(figsize=(15, 8))

    for s in range(S):
        plt.plot(hours, P_RT[:, s], linestyle='-', alpha=0.6, label=f"Scenario {s+1}")

    plt.xlabel("Hour")
    plt.ylabel("Price ($/MWHr)")
    plt.title("Real-Time Price Scenarios (Hourly Averaged)")
    plt.xticks(hours)
    plt.legend(loc="upper left", fontsize="small", ncol=2)

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
            normalized_data = np.sum(scenario_mean, axis=1) / daily_total  # 하루 기여도 계산
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
            raise ValueError("Hourly amount (T,) should match contribution shape (I, T).")

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
        raise ValueError("Invalid input shapes. Expected (I,T) with (T,) or (I,) with (1,).")

    return remuneration_amount, total_remuneration

# plot_hourly_contribution(hourly_contribution(x_vals), hourly_contribution(given_vals), labels=["x", "d"], selected_hours=[6, 7, 8, 9, 10, 11])
def plot_hourly_contribution(*hourly_contributions, labels=None, selected_hours=None):

    I, T = hourly_contributions[0].shape

    # 선택한 시간이 없으면 전체 24시간 사용
    if selected_hours is None:
        selected_hours = list(range(T))

    num_selected = len(selected_hours)
    num_rows = (num_selected // 6) + (1 if num_selected % 6 != 0 else 0)  # 필요한 행 개수 계산

    fig, axes = plt.subplots(num_rows, min(6, num_selected), figsize=(18, num_rows * 3), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(hourly_contributions))]

    # 선택한 시간대만 플롯
    for idx, t in enumerate(selected_hours):
        for i, data in enumerate(hourly_contributions):
            axes[idx].plot(range(I), data[:, t] * 100, marker='o', linestyle='-', label=labels[i])
            axes[idx].set_title(f'Hour {t}')
            axes[idx].set_xticks(range(I))
            axes[idx].set_ylabel('Contribution (%)')

    # Add a single legend for all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.02, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to fit legend
    plt.show()

# plot_daily_contribution(daily_contribution(x_vals), daily_contribution(given_vals), labels=["x", "d"])
def plot_daily_contribution(*daily_contributions, labels=None):
    I = len(daily_contributions[0])
    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(daily_contributions))]

    for i, data in enumerate(daily_contributions):
        plt.plot(range(I), data * 100, marker='o', linestyle='-', label=labels[i])

    plt.xlabel('Generator Index')
    plt.ylabel('Daily Contribution (%)')
    plt.title('Daily Contribution Rate')
    plt.xticks(range(I))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# plot_hourly_remuneration(remuneration_hourly, remuneration_hourly1, labels=["x", "d"], selected_hours=[6, 7, 8, 9, 10, 11])
def plot_hourly_remuneration(*hourly_remunerations, labels=None, selected_hours=None):
    
    only_hourly_remuneration_df = pd.read_csv("/Users/jangseohyun/Documents/workspace/symply/DER/result/result_only_hourly_profit.csv")
    only_hourly_remuneration = only_hourly_remuneration_df.pivot(index="DER", columns="Hour", values="hourly_total").values

    I, T = hourly_remunerations[0].shape

    # 선택한 시간이 없으면 전체 24시간 사용
    if selected_hours is None:
        selected_hours = list(range(T))

    num_selected = len(selected_hours)
    num_rows = (num_selected // 6) + (1 if num_selected % 6 != 0 else 0)  # 필요한 행 개수 계산

    fig, axes = plt.subplots(num_rows, min(6, num_selected), figsize=(18, num_rows * 3), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(hourly_remunerations))]

    # 선택한 시간대만 플롯
    for idx, t in enumerate(selected_hours):
        for i, data in enumerate(hourly_remunerations):
            axes[idx].plot(range(I), data[:, t], marker='o', linestyle='-', label=labels[i])
        axes[idx].plot(range(I), only_hourly_remuneration[:, t], marker='s', linestyle='-', label="Base", color="#3e3e3e")
        axes[idx].set_title(f'Hour {t}')
        axes[idx].set_xticks(range(I))
        axes[idx].set_ylabel('Remuneration ($)')

    # Add a single legend for all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.02, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to fit legend
    plt.show()

# plot_daily_remuneration(remuneration_daily, remuneration_daily1, labels=["x", "d"])
def plot_daily_remuneration(*daily_remunerations, labels=None):
    """
    하루별 정산액을 선 그래프로 플랏 (하나의 plot)
    x축: 발전기 index, y축: daily remuneration amount ($)
    """
    I = len(daily_remunerations[0])
    only_daily_remuneration = pd.read_csv("/Users/jangseohyun/Documents/workspace/symply/DER/result/result_only_profit.csv")
    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(daily_remunerations))]

    for i, data in enumerate(daily_remunerations):
        plt.plot(range(I), data, marker='o', linestyle='-', label=labels[i])

    plt.plot(range(I), only_daily_remuneration, marker='s', linestyle='--', color="#3e3e3e", label="Base")
    plt.xlabel('Generator Index')
    plt.ylabel('Daily Remuneration ($)')
    plt.title('Daily Remuneration Amount')
    plt.xticks(range(I))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 오른쪽 바깥에 배치
    plt.tight_layout(rect=[0, 0, 1.11, 1])
    plt.show()

def plot_summary(model, K, P_DA, P_RT, P_PN, a_vals, bp_vals, bm_vals, g_vals, s=0):
    T = len(P_DA)
    S = P_RT.shape[1]

    da_profit = sum(P_DA[t] * a_vals[t] for t in range(T))
    rt_profit = sum(P_RT[t, s_] * bp_vals[t, s_] / S for t in range(T) for s_ in range(S))
    pn_cost   = sum(P_PN[t]   * bm_vals[t, s_] / S for t in range(T) for s_ in range(S))
    total_profit = da_profit + rt_profit - pn_cost

    print(f"DA Profit      = {da_profit:.2f}")
    print(f"RT Profit      = {rt_profit:.2f}")
    print(f"Penalty Cost   = {pn_cost:.2f}")
    print(f"Total Profit   = {total_profit:.2f}, Objective Val  = {model.ObjVal:.2f}")

    hours = np.arange(T)
    hours_g = np.arange(T + 1)

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    total_commitment = np.sum(a_vals)
    axs[0].plot(hours, a_vals, marker='o', linewidth=2, color='#0096EB', label=f"α (Total: {total_commitment:.2f})")
    axs[0].set_title("Total Day-Ahead Commitment Over Time")
    axs[0].set_xlabel("Hour")
    axs[0].set_ylabel("Total x")
    axs[0].set_ylim(0, 2000)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].step(hours_g, g_vals[:T+1, s], where='post', label=f"SoC (Scen {s})", color='#00821E', linewidth=2)
    axs[1].set_title(f"Battery Charging/Discharging & SoC (Scenario {s})")
    axs[1].set_xlabel("Hour")
    axs[1].set_ylabel("Energy (kWh)")
    axs[1].set_xticks(np.arange(T+1))
    axs[1].set_ylim(-10, sum(K)+30)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()