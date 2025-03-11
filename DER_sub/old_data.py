generation_q = pd.read_csv(
    "/Users/jangseohyun/Documents/workspace/symply/DER/DATA_generation.csv"
)
generation_q["Time"] = pd.to_datetime(generation_q["Time"], format="%Y-%m-%d %H:%M")
generation_q["Hour"] = generation_q["Time"].dt.floor("h")
generation_h = generation_q.groupby("Hour").sum(numeric_only=True)

demand_q = pd.read_csv(
    "/Users/jangseohyun/Documents/workspace/symply/DER/DATA_demand.csv"
)
demand_q["Time"] = pd.to_datetime(demand_q["Time"], format="%Y-%m-%d %H:%M")
demand_q["Hour"] = demand_q["Time"].dt.floor("h")
demand_h = demand_q.groupby("Hour").sum(numeric_only=True)

price_q = pd.read_csv(
    "/Users/jangseohyun/Documents/workspace/symply/DER/DATA_price.csv"
)
price_q["Time"] = pd.to_datetime(price_q["Time"], format="%Y-%m-%d %H:%M")
I = list(range(len(generation_q.columns) - 11))
T = list(generation_h.index.hour.unique())
S = list(range(20))
prob = np.array([1 / len(S) for s in S])
# <!-- #### Generation -->
generation_avg = np.array(
    [[generation_h[generation_h.index.hour == t].mean().iloc[i] for t in T] for i in I]
)


def generate_randomized_generation(I, T, S, generation_avg, randomness_level):
    np.random.seed(7)
    if randomness_level == "low":
        noise_factors = np.random.uniform(0.8, 1.2, size=(len(I), len(T), len(S)))
    elif randomness_level == "medium":
        noise_factors = np.random.uniform(0.5, 1.5, size=(len(I), len(T), len(S)))
    elif randomness_level == "high":
        noise_factors = np.random.uniform(0.2, 1.8, size=(len(I), len(T), len(S)))
    elif randomness_level == "none":
        noise_factors = np.random.uniform(1, 1.1, size=(len(I), len(T), len(S)))
    else:
        raise ValueError(
            "Invalid randomness level. Please choose 'low', 'medium', or 'high'."
        )

    generation_r = np.zeros((len(I), len(T), len(S)))
    for i in range(len(I)):
        for t in range(len(T)):
            for s in range(len(S)):
                generation_r[i, t, s] = generation_avg[i, t] * noise_factors[i, t, s]

    return generation_r
# <!-- #### Demand -->
demand_avg = np.array(
    [[demand_h[demand_h.index.hour == t].mean().iloc[i] for t in T] for i in I]
)


def generate_randomized_demand(I, T, S, demand_avg, randomness_level):
    np.random.seed(17)
    if randomness_level == "low":
        noise_factors = np.random.uniform(0.8, 1.2, size=(len(I), len(T), len(S)))
    elif randomness_level == "medium":
        noise_factors = np.random.uniform(0.5, 1.5, size=(len(I), len(T), len(S)))
    elif randomness_level == "high":
        noise_factors = np.random.uniform(0.2, 1.8, size=(len(I), len(T), len(S)))
    elif randomness_level == "none":
        noise_factors = np.random.uniform(1, 1.1, size=(len(I), len(T), len(S)))
    else:
        raise ValueError(
            "Invalid randomness level. Please choose 'low', 'medium', or 'high'."
        )

    demand_r = np.zeros((len(I), len(T), len(S)))
    for i in range(len(I)):
        for t in range(len(T)):
            for s in range(len(S)):
                demand_r[i, t, s] = demand_avg[i, t] * noise_factors[i, t, s]

    return demand_r
# <!-- #### Randomize -->
random_key = "high"
generation_r = generate_randomized_generation(I, T, S, generation_avg, random_key)
demand_r = generate_randomized_demand(I, T, S, demand_avg, random_key)
# <!-- #### Residual = generation - demand -->
residual = np.zeros((len(I), len(T), len(S)))
for i in range(len(I)):
    for t in range(len(T)):
        for s in range(len(S)):
            residual[i, t, s] = generation_r[i, t, s] - demand_r[i, t, s]

R = np.zeros((len(I), len(T), len(S)))
for i in range(len(I)):
    for t in range(len(T)):
        for s in range(len(S)):
            R[i, t, s] = max(0, residual[i, t, s])
index = pd.MultiIndex.from_product([range(len(I)), range(len(T)), range(len(S))],
                                 names=['generator', 'time', 'scenario'])
df = pd.DataFrame({'value': R.flatten()}, index=index).reset_index()
df.to_csv('result/result_R.csv', index=False)
# <!-- #### Price 설정 -->
price_q["Hour"] = price_q["Time"].dt.floor("h")
price_h = price_q.groupby("Hour").mean(numeric_only=True)

price = price_h.iloc[: len(S) * len(T)]

P_DA = np.array(
    [sum(price["Price"].iloc[t + s * len(T)] for s in S) / len(S) * 1.2 for t in T]
)
P_RT = np.array([[price["Price"].iloc[t + s * len(T)] for s in S] for t in T])
P_PN = np.array(
    [sum(price["Price"].iloc[t + s * len(T)] for s in S) / len(S) * 2 for t in T]
)