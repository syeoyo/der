import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def extracted_target(I, T, S, x_hol, ep_hol, em_hol, yp_hol, ym_hol, d_hol, dp_hol, dm_hol):
    x_extracted = {}
    ep_extracted = {}
    em_extracted = {}
    yp_extracted = {}
    ym_extracted = {}
    d_extracted = {}
    dp_extracted = {}
    dm_extracted = {}
    i_map_extracted = {}

    for target_i in range(I):
        I_set = [i for i in range(I) if i != target_i]
        i_map = {i: idx for idx, i in enumerate(I_set)}

        x_extracted[target_i] = np.array([[x_hol[i, t] for t in range(T)] for i in I_set])
        ep_extracted[target_i] = np.array([[[ep_hol[i, t, s] for s in range(S)] for t in range(T)] for i in I_set])
        em_extracted[target_i] = np.array([[[em_hol[i, t, s] for s in range(S)] for t in range(T)] for i in I_set])
        yp_extracted[target_i] = np.array([[[yp_hol[i, t, s] for s in range(S)] for t in range(T)] for i in I_set])
        ym_extracted[target_i] = np.array([[[ym_hol[i, t, s] for s in range(S)] for t in range(T)] for i in I_set])
        d_extracted[target_i] = np.array([[[[d_hol[i, j, t, s] for s in range(S)] for t in range(T)] for j in I_set] for i in I_set])
        dp_extracted[target_i] = np.array([[[dp_hol[i, t, s] for s in range(S)] for t in range(T)] for i in I_set])
        dm_extracted[target_i] = np.array([[[dm_hol[i, t, s] for s in range(S)] for t in range(T)] for i in I_set])
        i_map_extracted[target_i] = i_map

    return x_extracted, ep_extracted, em_extracted, yp_extracted, ym_extracted, d_extracted, dp_extracted, dm_extracted, i_map_extracted


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
                    # given_profit += dp_vals[i_map[i], t, s] * P_RT[t, s]
                    # received_profit += dm_vals[i_map[i], t, s] * P_PN[t]
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
                q0_list_rdc = np.linspace(-1, 1.1 * total_supply, 10)
                prices_rdc = []
                for q0 in q0_list_rdc:
                    q_cleared = (a_d - a_s + b_s * q0) / denom
                    p_cleared = a_d - b_d * q_cleared
                    prices_rdc.append(p_cleared)
                q_rdc = np.array(q0_list_rdc).reshape(-1, 1)
                p_rdc = np.array(prices_rdc)

                # === RSC: q0 → 수요 증가 ===
                q0_list_rsc = np.linspace(-1, 1.1 * total_demand, 10)
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


def transform_step(rho_plus, rho_minus, total_demand_without, total_supply_without, step_width_multiplier=1.0):

    I, T, S, _ = rho_plus.shape
    rho_plus_step = np.full((I, T, S), np.nan, dtype=object)
    rho_minus_step = np.full((I, T, S), np.nan, dtype=object)

    B_map_plus = np.zeros((I, T, S), dtype=int)
    B_map_minus = np.zeros((I, T, S), dtype=int)

    for target_i, t, s in product(range(I), range(T), range(S)):
        # ➤ RDC 
        ap, bp = rho_plus[target_i, t, s]  
        Bp = max(1, int(np.ceil(total_demand_without[target_i, t, s] / step_width_multiplier)))

        if total_demand_without[target_i, t, s] <= 0:
            rho_plus_step[target_i][t][s] = np.array([[0, ap + bp * 0]])
            B_map_plus[target_i, t, s] = 1
        else:
            qp_org = np.linspace(0, total_demand_without[target_i, t, s], Bp+1)
            step_width = total_demand_without[target_i, t, s] / (Bp)
            qp_bound = np.linspace(qp_org[0]-step_width/2, qp_org[-1]+step_width/2, Bp + 2)
            qp_bound = np.clip(qp_bound, 0, total_demand_without[target_i, t, s])
            rho_plus_step[target_i][t][s] = np.array([[qp, ap + bp * qp] for qp in qp_bound[:-1]])
            B_map_plus[target_i, t, s] = Bp

        # ➤ RSC
        am, bm = rho_minus[target_i, t, s]
        Bm = max(1, int(np.ceil(total_supply_without[target_i, t, s] / step_width_multiplier)))

        if total_supply_without[target_i, t, s] <= 0:
            rho_minus_step[target_i][t][s] = np.array([[0, am + bm * 0]])
            B_map_minus[target_i, t, s] = 1
        else:
            qm_org = np.linspace(0, total_supply_without[target_i, t, s], Bm+1)
            step_width = total_supply_without[target_i, t, s] / (Bm)
            qm_bound = np.linspace(qm_org[0]-step_width/2, qm_org[-1]+step_width/2, Bm + 2)
            qm_bound = np.clip(qm_bound, 0, total_supply_without[target_i, t, s])
            rho_minus_step[target_i][t][s] = np.array([[qm, am + bm * qm] for qm in qm_bound[:-1]])
            B_map_minus[target_i, t, s] = Bm

    return rho_plus_step, rho_minus_step, B_map_plus, B_map_minus


def compare_allstep(a_hol, bp_hol, bm_hol, dp_hol, dm_hol, x_part, yp_part, ym_part, dp_part, dm_part, T):

    x_pwl_sum = np.sum(np.array(x_part), axis=0)                 
    yp_pwl_avg = np.mean(np.sum(np.array(yp_part), axis=0), axis=1)
    ym_pwl_avg = np.mean(np.sum(np.array(ym_part), axis=0), axis=1)
    dp_pwl_avg = np.mean(np.sum(np.array(dp_part), axis=0), axis=1)
    dm_pwl_avg = np.mean(np.sum(np.array(dm_part), axis=0), axis=1)

    print(f"=== [시나리오 평균] set 모델 vs 전체 PWL 결과 비교 ===\n")
    print(f"{'t':>2} | {'a_opt':>10} {'x_pwl_sum':>10} || {'bp_opt':>10} {'yp_avg':>10} || {'bm_opt':>10} {'ym_avg':>10} || {'dp_opt':>10} {'dp_avg':>10} || {'dm_opt':>10} {'dm_avg':>10}")
    print("-" * 125)

    for t in range(T):
        a_ = a_hol[t]
        x_ = x_pwl_sum[t]

        bp_ = np.mean(bp_hol[t, :])
        yp_ = yp_pwl_avg[t]

        bm_ = np.mean(bm_hol[t, :])
        ym_ = ym_pwl_avg[t]

        dp_ = np.mean(dp_hol[:, t, :].sum(axis=0))
        dpp_ = dp_pwl_avg[t]

        dm_ = np.mean(dm_hol[:, t, :].sum(axis=0))
        dmm_ = dm_pwl_avg[t]

        print(f"{t:>2} | {a_:>10.3f} {x_:>10.3f} || {bp_:>10.3f} {yp_:>10.3f} || {bm_:>10.3f} {ym_:>10.3f} || {dp_:>10.3f} {dpp_:>10.3f} || {dm_:>10.3f} {dmm_:>10.3f}")

def compare_onestep(a_hol, bp_hol, bm_hol, dp_hol, dm_hol, x_part, yp_part, ym_part, dp_part, dm_part, x_without, ep_without, em_without, dp_without, dm_without, T, target_i):
    
    print(f"=== [시나리오 평균] set 모델 vs (PWL + without) 결과 비교 (target_i={target_i}) ===\n")
    print(f"{'t':>2} | {'a_opt':>10} {'x_sum':>10} || {'bp_opt':>10} {'yp_avg':>10} || {'bm_opt':>10} {'ym_avg':>10} || {'dp_opt':>10} {'dp_avg':>10} || {'dm_opt':>10} {'dm_avg':>10}")
    print("-" * 125)

    for t in range(T):
        a_ = a_hol[t]
        x_ = x_part[target_i][t] + x_without[target_i][:, t].sum()

        bp_ = np.mean(bp_hol[t, :])
        yp_ = np.mean(yp_part[target_i][t, :] + ep_without[target_i][:, t, :].sum(axis=0))

        bm_ = np.mean(bm_hol[t, :])
        ym_ = np.mean(ym_part[target_i][t, :] + em_without[target_i][:, t, :].sum(axis=0))

        dp_ = np.mean(dp_hol[:, t, :].sum(axis=0))
        dpp_ = np.mean(dp_part[target_i][t, :] + dp_without[target_i][:, t, :].sum(axis=0))

        dm_ = np.mean(dm_hol[:, t, :].sum(axis=0))
        dmm_ = np.mean(dm_part[target_i][t, :] + dm_without[target_i][:, t, :].sum(axis=0))

        print(f"{t:>2} | {a_:>10.3f} {x_:>10.3f} || {bp_:>10.3f} {yp_:>10.3f} || {bm_:>10.3f} {ym_:>10.3f} || {dp_:>10.3f} {dpp_:>10.3f} || {dm_:>10.3f} {dmm_:>10.3f}")

def compare_onestep_detailed(a_hol, bp_hol, bm_hol, dp_hol, dm_hol, x_part, yp_part, ym_part, dp_part, dm_part, x_without, ep_without, em_without, dp_without, dm_without, T, target_i):
    print(f"=== [시나리오 평균] PWL(target_i) + without(target_i) vs set 모델 비교 (target_i={target_i}) ===\n")
    print(f"{'t':>2} | {'a_opt':>10} {'x_pwl':>10} {'x_wo':>10} {'x_sum':>10} || "
          f"{'bp_hol':>10} {'yp_pwl':>10} {'ep_wo':>10} {'yp_sum':>10} || "
          f"{'bm_hol':>10} {'ym_pwl':>10} {'em_wo':>10} {'ym_sum':>10} || "
          f"{'dp_hol':>10} {'dp_pwl':>10} {'d_wo':>10} {'dp_sum':>10} || "
          f"{'dm_hol':>10} {'dm_pwl':>10} {'d_wo':>10} {'dm_sum':>10}")
    print("-" * 210)

    for t in range(T):
        a_ = a_hol[t]
        x_step = x_part[target_i][t]
        x_wo = x_without[target_i][:, t].sum()
        x_sum = x_step + x_wo

        bp_hol_avg = np.mean(bp_hol[t, :])
        yp_step = np.mean(yp_part[target_i][t, :])
        ep_wo = np.mean(ep_without[target_i][:, t, :].sum(axis=0))
        yp_sum = yp_step + ep_wo

        bm_hol_avg = np.mean(bm_hol[t, :])
        ym_step = np.mean(ym_part[target_i][t, :])
        em_wo = np.mean(em_without[target_i][:, t, :].sum(axis=0))
        ym_sum = ym_step + em_wo

        dp_hol_avg = np.mean(dp_hol[:, t, :].sum(axis=0))
        dp_step = np.mean(dp_part[target_i][t, :])
        d_wo = np.mean(dp_without[target_i][:, t, :].sum(axis=0))
        dp_sum = dp_step + d_wo

        dm_hol_avg = np.mean(dm_hol[:, t, :].sum(axis=0))
        dm_step = np.mean(dm_part[target_i][t, :])
        d_wo_m = np.mean(dm_without[target_i][:, t, :].sum(axis=0))
        dm_sum = dm_step + d_wo_m

        print(f"{t:>2} | {a_:>10.3f} {x_step:>10.3f} {x_wo:>10.3f} {x_sum:>10.3f} || "
              f"{bp_hol_avg:>10.3f} {yp_step:>10.3f} {ep_wo:>10.3f} {yp_sum:>10.3f} || "
              f"{bm_hol_avg:>10.3f} {ym_step:>10.3f} {em_wo:>10.3f} {ym_sum:>10.3f} || "
              f"{dp_hol_avg:>10.3f} {dp_step:>10.3f} {d_wo:>10.3f} {dp_sum:>10.3f} || "
              f"{dm_hol_avg:>10.3f} {dm_step:>10.3f} {d_wo_m:>10.3f} {dm_sum:>10.3f}")
