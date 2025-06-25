import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Holistic Model

def optimize_hol(R, K, K0, P_DA, P_RT, P_PN, I, T, S, M1, M2):
    set = gp.Model("set")
    set.setParam("MIPGap", 1e-7)
    set.setParam("OutputFlag", 0)

    x_hol = set.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0, name="x")
    ep_hol = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="e_plus")
    em_hol = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="e_minus")

    yp_hol = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_plus")
    ym_hol = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_minus")
    z_hol = set.addVars(I, T + 1, S, vtype=GRB.CONTINUOUS, name="z")
    zc_hol = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="z_charge")
    zd_hol = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="z_discharge")
    d_hol = set.addVars(I, I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="d")

    p1_hol = set.addVars(I, T, S, vtype=GRB.BINARY, name="p1")
    p2_hol = set.addVars(I, T, S, vtype=GRB.BINARY, name="p2")
    p3_hol = set.addVars(I, T, S, vtype=GRB.BINARY, name="p3")
    p4_hol = set.addVars(I, T, S, vtype=GRB.BINARY, name="p4")

    set.update()

    obj_hol = gp.quicksum(P_DA[t] * gp.quicksum(x_hol[i, t] for i in range(I)) for t in range(T)) + gp.quicksum((1 / S) * (P_RT[t, s] * gp.quicksum(ep_hol[i, t, s] for i in range(I)) - P_PN[t] * gp.quicksum(em_hol[i, t, s] for i in range(I))) for t in range(T) for s in range(S))

    set.setObjective(obj_hol, GRB.MAXIMIZE)

    for i, t, s in product(range(I), range(T), range(S)):
        set.addConstr(R[i, t, s] - x_hol[i, t] == yp_hol[i, t, s] - ym_hol[i, t, s] + zc_hol[i, t, s] - zd_hol[i, t, s])
        set.addConstr(yp_hol[i, t, s] + zc_hol[i, t, s] <= R[i, t, s] + zd_hol[i, t, s])
        set.addConstr(zd_hol[i, t, s] <= z_hol[i, t, s])
        set.addConstr(zc_hol[i, t, s] <= K[i] - z_hol[i, t, s])
        set.addConstr(yp_hol[i, t, s] <= M1 * p3_hol[i, t, s])
        set.addConstr(ym_hol[i, t, s] <= M1 * (1 - p3_hol[i, t, s]))
        set.addConstr(ym_hol[i, t, s] <= M1 * p2_hol[i, t, s])
        set.addConstr(zc_hol[i, t, s] <= M1 * (1 - p2_hol[i, t, s]))
        set.addConstr(zc_hol[i, t, s] <= M1 * p1_hol[i, t, s])
        set.addConstr(zd_hol[i, t, s] <= M1 * (1 - p1_hol[i, t, s]))
        set.addConstr(z_hol[i, t, s] <= K[i])
        set.addConstr(z_hol[i, t + 1, s] == z_hol[i, t, s] + zc_hol[i, t, s] - zd_hol[i, t, s])
    for i, s in product(range(I), range(S)):
        set.addConstr(z_hol[i, 0, s] == K0[i])

    for i, t, s in product(range(I), range(T), range(S)):
        set.addConstr(ep_hol[i, t, s] == yp_hol[i, t, s] - gp.quicksum(d_hol[i, j, t, s] for j in range(I)))
        set.addConstr(em_hol[i, t, s] == ym_hol[i, t, s] - gp.quicksum(d_hol[j, i, t, s] for j in range(I)))
        set.addConstr(gp.quicksum(ep_hol[i, t, s] for i in range(I)) <= M2 * p4_hol[i, t, s])
        set.addConstr(gp.quicksum(em_hol[i, t, s] for i in range(I)) <= M2 * (1 - p4_hol[i, t, s]))
        set.addConstr(d_hol[i, i, t, s] == 0)

    set.optimize()

    if set.status == GRB.OPTIMAL:
        print(f"Optimal solution found! Objective value: {set.objVal}")
    else:
        print("No optimal solution found.")
        
    x_hol = np.array([[x_hol[i, t].X for t in range(T)] for i in range(I)])
    a_hol = np.sum(x_hol, axis=0)
    yp_hol = np.array([[[yp_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]) 
    ym_hol = np.array([[[ym_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    z_hol = np.array([[[z_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    zc_hol = np.array([[[zc_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    zd_hol = np.array([[[zd_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    ep_hol = np.array([[[ep_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    bp_hol = np.sum(ep_hol, axis=0) 
    em_hol = np.array([[[em_hol[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    bm_hol = np.sum(em_hol, axis=0) 
    d_hol = np.array([[[[d_hol[i, j, t, s].X for s in range(S)] for t in range(T)] for j in range(I)] for i in range(I)])
    dp_hol = np.sum(d_hol, axis=1)
    dm_hol = np.sum(d_hol, axis=0)
    
    return {
        'x_hol': x_hol, 'a_hol': a_hol, 'yp_hol': yp_hol, 'ym_hol': ym_hol,
        'z_hol': z_hol, 'zc_hol': zc_hol, 'zd_hol': zd_hol, 'ep_hol': ep_hol,
        'bp_hol': bp_hol, 'em_hol': em_hol, 'bm_hol': bm_hol, 'd_hol': d_hol,
        'dp_hol': dp_hol, 'dm_hol': dm_hol, 'obj_hol': set.objVal
    }


# Holistic Optimization without target DER
def optimize_without(target_i, R, K, K0, P_DA, P_RT, P_PN, I, T, S):
    I_set = [i for i in range(I) if i != target_i]
    M1 = np.maximum(R[I_set], K[I_set, None, None]).max()
    M2 = max(R[I_set].sum(axis=0).max(), K[I_set].sum())

    set_wo = gp.Model(f"set_without_{target_i}")
    set_wo.setParam("MIPGap", 1e-7)
    set_wo.setParam("OutputFlag", 0)

    x_wo = set_wo.addVars(I_set, T, vtype=GRB.CONTINUOUS, lb=0, name="x")
    ep_wo = set_wo.addVars(I_set, T, S, vtype=GRB.CONTINUOUS, name="e_plus")
    em_wo = set_wo.addVars(I_set, T, S, vtype=GRB.CONTINUOUS, name="e_minus")
    yp_wo = set_wo.addVars(I_set, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_plus")
    ym_wo = set_wo.addVars(I_set, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_minus")
    z_wo = set_wo.addVars(I_set, T + 1, S, vtype=GRB.CONTINUOUS, name="z")
    zc_wo = set_wo.addVars(I_set, T, S, vtype=GRB.CONTINUOUS, name="z_charge")
    zd_wo = set_wo.addVars(I_set, T, S, vtype=GRB.CONTINUOUS, name="z_discharge")
    d_wo = set_wo.addVars(I_set, I_set, T, S, vtype=GRB.CONTINUOUS, lb=0, name="d")

    p1_wo = set_wo.addVars(I_set, T, S, vtype=GRB.BINARY, name="p1")
    p2_wo = set_wo.addVars(I_set, T, S, vtype=GRB.BINARY, name="p2")
    p3_wo = set_wo.addVars(I_set, T, S, vtype=GRB.BINARY, name="p3")
    p4_wo = set_wo.addVars(I_set, T, S, vtype=GRB.BINARY, name="p4")

    obj = gp.quicksum(P_DA[t] * x_wo[i, t] for i in I_set for t in range(T)) + gp.quicksum(
        (1 / S) * (
            P_RT[t, s] * gp.quicksum(ep_wo[i, t, s] for i in I_set) -
            P_PN[t] * gp.quicksum(em_wo[i, t, s] for i in I_set)
        )
        for t in range(T) for s in range(S)
    )
    set_wo.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(I_set, range(T), range(S)):
        set_wo.addConstr(R[i, t, s] - x_wo[i, t] == yp_wo[i, t, s] - ym_wo[i, t, s] + zc_wo[i, t, s] - zd_wo[i, t, s])
        set_wo.addConstr(yp_wo[i, t, s] + zc_wo[i, t, s] <= R[i, t, s] + zd_wo[i, t, s])
        set_wo.addConstr(zd_wo[i, t, s] <= z_wo[i, t, s])
        set_wo.addConstr(zc_wo[i, t, s] <= K[i] - z_wo[i, t, s])
        set_wo.addConstr(yp_wo[i, t, s] <= M1 * p3_wo[i, t, s])
        set_wo.addConstr(ym_wo[i, t, s] <= M1 * (1 - p3_wo[i, t, s]))
        set_wo.addConstr(ym_wo[i, t, s] <= M1 * p2_wo[i, t, s])
        set_wo.addConstr(zc_wo[i, t, s] <= M1 * (1 - p2_wo[i, t, s]))
        set_wo.addConstr(zc_wo[i, t, s] <= M1 * p1_wo[i, t, s])
        set_wo.addConstr(zd_wo[i, t, s] <= M1 * (1 - p1_wo[i, t, s]))
        set_wo.addConstr(z_wo[i, t, s] <= K[i])
        set_wo.addConstr(z_wo[i, t + 1, s] == z_wo[i, t, s] + zc_wo[i, t, s] - zd_wo[i, t, s])

    for i, s in product(I_set, range(S)):
        set_wo.addConstr(z_wo[i, 0, s] == K0[i])

    for i, t, s in product(I_set, range(T), range(S)):
        set_wo.addConstr(ep_wo[i, t, s] == yp_wo[i, t, s] - gp.quicksum(d_wo[i, j, t, s] for j in I_set if j != i))
        set_wo.addConstr(em_wo[i, t, s] == ym_wo[i, t, s] - gp.quicksum(d_wo[j, i, t, s] for j in I_set if j != i))
        set_wo.addConstr(gp.quicksum(ep_wo[i, t, s] for i in I_set) <= M2 * p4_wo[i, t, s])
        set_wo.addConstr(gp.quicksum(em_wo[i, t, s] for i in I_set) <= M2 * (1 - p4_wo[i, t, s]))
        set_wo.addConstr(d_wo[i, i, t, s] == 0)

    set_wo.optimize()

    i_map = {i: idx for idx, i in enumerate(I_set)}

    x_wo = np.array([[x_wo[i, t].X for t in range(T)] for i in I_set])
    ep_wo = np.array([[[ep_wo[i, t, s].X for s in range(S)] for t in range(T)] for i in I_set])
    em_wo = np.array([[[em_wo[i, t, s].X for s in range(S)] for t in range(T)] for i in I_set])
    yp_wo = np.array([[[yp_wo[i, t, s].X for s in range(S)] for t in range(T)] for i in I_set])
    ym_wo = np.array([[[ym_wo[i, t, s].X for s in range(S)] for t in range(T)] for i in I_set])
    d_wo = np.array([[[[d_wo[i, j, t, s].X for s in range(S)] for t in range(T)] for j in I_set] for i in I_set])
    dp_wo = np.sum(d_wo, axis=1)
    dm_wo = np.sum(d_wo, axis=0)

    return x_wo, ep_wo, em_wo, yp_wo, ym_wo, d_wo, dp_wo, dm_wo, i_map


# Holistic Optimization without target DER Loop
def optimize_without_loop(R, K, K0, P_DA, P_RT, P_PN, I, T, S):
    x_without = {}; ep_without = {}; em_without = {}
    yp_without = {}; ym_without = {}; d_without = {}; dp_without = {}; dm_without = {}; i_map_without = {}

    for target_i in tqdm(range(I), desc="Solving settlement model for each target DER"):
        x_vals, ep_vals, em_vals, yp_vals, ym_vals, d_vals, dp_vals, dm_vals, i_map = optimize_without(
            target_i, R, K, K0, P_DA, P_RT, P_PN, I, T, S
        )
        x_without[target_i] = x_vals
        ep_without[target_i] = ep_vals
        em_without[target_i] = em_vals
        yp_without[target_i] = yp_vals
        ym_without[target_i] = ym_vals
        d_without[target_i] = d_vals
        dp_without[target_i] = dp_vals
        dm_without[target_i] = dm_vals
        i_map_without[target_i] = i_map

    return x_without, ep_without, em_without, yp_without, ym_without, d_without, dp_without, dm_without, i_map_without

# RDC & RSC -> Price Functions
def compute_price_functions(yp_without, ym_without, dp_without, dm_without, i_map_without, P_RT, P_PN, T, S, I):
    rdc_coefficients_all = np.full((I, T, S, 2), np.nan)
    rsc_coefficients_all = np.full((I, T, S, 2), np.nan)
    rho_plus_func_all = np.full((I, T, S, 2), np.nan)   # RDC
    rho_minus_func_all = np.full((I, T, S, 2), np.nan)  # RSC
    
    total_demand_without = np.zeros((I, T, S))
    total_supply_without = np.zeros((I, T, S))

    for target_i in range(I):
        yp_vals = yp_without[target_i]
        ym_vals = ym_without[target_i]
        dp_vals = dp_without[target_i]
        dm_vals = dm_without[target_i]
        i_map = i_map_without[target_i]

        for t in range(T):
            for s in range(S):
                total_supply = sum(yp_vals[i_map[i], t, s] for i in i_map)
                total_demand = sum(ym_vals[i_map[i], t, s] for i in i_map)
                
                total_supply_without[target_i, t, s] = total_supply
                total_demand_without[target_i, t, s] = total_demand

                given_profit = received_profit = realized_supply = realized_demand = 0

                for i in i_map:
                    given_profit += dp_vals[i_map[i], t, s] * P_PN[t]
                    received_profit += dm_vals[i_map[i], t, s] * P_RT[t, s]
                    realized_supply += dp_vals[i_map[i], t, s]
                    realized_demand += dm_vals[i_map[i], t, s]

                BIG_M_POS = 1e10
                BIG_M_NEG = -1e10

                if realized_demand <= 1e-4 or realized_supply <= 1e-4:
                    rho_plus_func_all[target_i, t, s, :] = [BIG_M_NEG, 0.0]
                    rho_minus_func_all[target_i, t, s, :] = [BIG_M_POS, 0.0]
                    continue

                a_d = P_PN[t]
                b_d = 2 * (a_d * realized_demand - received_profit) / (realized_demand ** 2)
                a_s = P_RT[t, s]
                b_s = 2 * (given_profit - a_s * realized_supply) / (realized_supply ** 2)

                denom = b_d + b_s
                if abs(denom) < 1e-6: continue

                # === RDC: q0 → 공급 증가 ===
                q0_list_rdc = np.linspace(-5, 1.1 * total_supply, 10)
                prices_rdc = []
                for q0 in q0_list_rdc:
                    q_cleared = (a_d - a_s + b_s * q0) / denom
                    p_cleared = a_d - b_d * q_cleared
                    prices_rdc.append(p_cleared)
                q_rdc = np.array(q0_list_rdc).reshape(-1, 1)
                p_rdc = np.array(prices_rdc)

                # === RSC: q0 → 수요 증가 ===
                q0_list_rsc = np.linspace(-5, 1.1 * total_demand, 10)
                prices_rsc = []
                for q0 in q0_list_rsc:
                    q_cleared = (a_d + b_d * q0 - a_s) / denom
                    p_cleared = a_s + b_s * q_cleared
                    prices_rsc.append(p_cleared)
                q_rsc = np.array(q0_list_rsc).reshape(-1, 1)
                p_rsc = np.array(prices_rsc)

                # 선형 회귀 근사 (1차)
                X_poly_rdc = PolynomialFeatures(degree=1).fit_transform(q_rdc)
                X_poly_rsc = PolynomialFeatures(degree=1).fit_transform(q_rsc)

                model_rdc = LinearRegression().fit(X_poly_rdc, p_rdc)
                model_rsc = LinearRegression().fit(X_poly_rsc, p_rsc)

                a_rdc, b_rdc = model_rdc.intercept_, model_rdc.coef_[1]
                a_rsc, b_rsc = model_rsc.intercept_, model_rsc.coef_[1]

                rho_plus_func_all[target_i, t, s, :] = [a_rdc, b_rdc]
                rho_minus_func_all[target_i, t, s, :] = [a_rsc, b_rsc]
                rdc_coefficients_all[target_i, t, s, :] = [a_rdc, b_rdc]
                rsc_coefficients_all[target_i, t, s, :] = [a_rsc, b_rsc]

    return rdc_coefficients_all, rsc_coefficients_all, rho_plus_func_all, rho_minus_func_all, total_demand_without, total_supply_without


def transform_step(rho_plus, rho_minus, total_demand_without, total_supply_without):

    I, T, S, _ = rho_plus.shape
    rho_plus_step = np.full((I, T, S), np.nan, dtype=object)
    rho_minus_step = np.full((I, T, S), np.nan, dtype=object)

    B_map_plus = np.zeros((I, T, S), dtype=int)
    B_map_minus = np.zeros((I, T, S), dtype=int)

    for target_i, t, s in product(range(I), range(T), range(S)):
        # ➤ RDC 
        ap, bp = rho_plus[target_i, t, s]  
        Bp = max(1, int(np.ceil(total_supply_without[target_i, t, s])))

        if total_supply_without[target_i, t, s] <= 0:
            rho_plus_step[target_i][t][s] = np.array([[0, ap + bp * 0]])
            B_map_plus[target_i, t, s] = 1
        else:
            qp_org = np.linspace(0, total_supply_without[target_i, t, s], Bp)
            step_width = total_supply_without[target_i, t, s] / Bp
            qp_bound = np.linspace(-step_width/2, total_supply_without[target_i, t, s] + step_width/2, Bp + 1)
            qp_bound = np.clip(qp_bound, 0, total_supply_without[target_i, t, s])
            rho_plus_step[target_i][t][s] = np.array([[qp, ap + bp * qp] for qp in qp_bound[:-1]])  # 마지막 점 제외
            B_map_plus[target_i, t, s] = Bp

        # ➤ RSC
        am, bm = rho_minus[target_i, t, s]
        Bm = max(1, int(np.ceil(total_demand_without[target_i, t, s])))

        if total_demand_without[target_i, t, s] <= 0:
            rho_minus_step[target_i][t][s] = np.array([[0, am + bm * 0]])
            B_map_minus[target_i, t, s] = 1
        else:
            qm_org = np.linspace(0, total_demand_without[target_i, t, s], Bm)
            step_width = total_demand_without[target_i, t, s] / Bm
            qm_bound = np.linspace(-step_width/2, total_demand_without[target_i, t, s] + step_width/2, Bm + 1)
            qm_bound = np.clip(qm_bound, 0, total_demand_without[target_i, t, s])
            rho_minus_step[target_i][t][s] = np.array([[qm, am + bm * qm] for qm in qm_bound[:-1]])
            B_map_minus[target_i, t, s] = Bm

    return rho_plus_step, rho_minus_step, B_map_plus, B_map_minus

