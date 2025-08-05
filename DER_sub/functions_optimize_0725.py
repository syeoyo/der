import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Individual Participation Model (only for target DER)
def optimize_individually(R, K, K0, P_DA, P_RT, P_PN, T, S, M1, target_i):
    only = gp.Model("only")
    only.setParam("MIPGap", 1e-7)
    only.setParam("OutputFlag", 0)

    x_ind = only.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp_ind = only.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_plus")
    ym_ind = only.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_minus")
    z_ind = only.addVars(T+1, S, vtype=GRB.CONTINUOUS, name="z")
    zc_ind = only.addVars(T, S, vtype=GRB.CONTINUOUS, name="z_charge")
    zd_ind = only.addVars(T, S, vtype=GRB.CONTINUOUS, name="z_discharge")
    zeta = only.addVars(T, S, vtype=GRB.BINARY, name="zeta")
    delta = only.addVars(T, S, vtype=GRB.BINARY, name="delta")
    rho = only.addVars(T, S, vtype=GRB.BINARY, name="rho")

    only.update()

    obj = gp.quicksum(P_DA[t] * x_ind[t] for t in range(T)) \
        + gp.quicksum(1 / S * (P_RT[t, s] * yp_ind[t, s] - P_PN[t] * ym_ind[t, s]) for t, s in product(range(T), range(S)))

    only.setObjective(obj, GRB.MAXIMIZE)

    for t, s in product(range(T), range(S)):
        only.addConstr(R[target_i, t, s] - x_ind[t] == yp_ind[t, s] - ym_ind[t, s] + zc_ind[t, s] - zd_ind[t, s])
        only.addConstr(yp_ind[t, s] <= R[target_i, t, s])
        only.addConstr(z_ind[t + 1, s] == z_ind[t, s] + zc_ind[t, s] - zd_ind[t, s])
        only.addConstr(zd_ind[t, s] <= z_ind[t, s])
        only.addConstr(zc_ind[t, s] <= K[target_i] - z_ind[t, s])
        only.addConstr(yp_ind[t, s] <= M1 * rho[t, s])
        only.addConstr(ym_ind[t, s] <= M1 * (1 - rho[t, s]))
        only.addConstr(ym_ind[t, s] <= M1 * delta[t, s])
        only.addConstr(zc_ind[t, s] <= M1 * (1 - delta[t, s]))
        only.addConstr(zc_ind[t, s] <= M1 * zeta[t, s])
        only.addConstr(zd_ind[t, s] <= M1 * (1 - zeta[t, s]))
    
    for s in range(S):
        only.addConstr(z_ind[0, s] == K0[target_i])

    only.optimize()

    if only.status == GRB.OPTIMAL:
        print(f"Optimal solution found for target_i={target_i}! Objective value: {only.objVal}")
        obj_val = only.objVal
    else:
        print(f"No optimal solution found for target_i={target_i}.")
        obj_val = None

    x_ind = np.array([x_ind[t].X for t in range(T)])
    yp_ind = np.array([[yp_ind[t, s].X for s in range(S)] for t in range(T)]) 
    ym_ind = np.array([[ym_ind[t, s].X for s in range(S)] for t in range(T)])
    z_ind  = np.array([[z_ind[t, s].X for s in range(S)] for t in range(T)])
    zc_ind = np.array([[zc_ind[t, s].X for s in range(S)] for t in range(T)])
    zd_ind = np.array([[zd_ind[t, s].X for s in range(S)] for t in range(T)])
    
    return x_ind, yp_ind, ym_ind, z_ind, zc_ind, zd_ind, obj_val


# Individual Participation Model LOOP
def optimize_individually_forall(R, K, K0, P_DA, P_RT, P_PN, I, T, S, M1):
    x_ind = {} ; yp_ind = {} ; ym_ind = {}; z_ind = {} ; zc_ind = {} ; zd_ind = {}; obj_ind = {}
    
    for target_i in tqdm(range(I), desc="Optimizing individually for each target_i"):
        x_individual, yp_individual, ym_individual, z_individual, zc_individual, zd_individual, obj_individual = optimize_individually(
            R, K, K0, P_DA, P_RT, P_PN, T, S, M1, target_i
        )
        
        x_ind[target_i] = x_individual
        yp_ind[target_i] = yp_individual
        ym_ind[target_i] = ym_individual
        z_ind[target_i] = z_individual
        zc_ind[target_i] = zc_individual
        zd_ind[target_i] = zd_individual
        obj_ind[target_i] = obj_individual
    
    return x_ind, yp_ind, ym_ind, z_ind, zc_ind, zd_ind, obj_ind


# Holistic Model (including all DERs)
def optimize_hol(R, K, K0, P_DA, P_RT, P_PN, I, T, S, M1, M2):
    set = gp.Model("set")
    set.setParam("MIPGap", 1e-3)
    # set.setParam("OutputFlag", 0)

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
    
    return x_hol, a_hol, yp_hol, ym_hol, z_hol, zc_hol, zd_hol, ep_hol, bp_hol, em_hol, bm_hol, d_hol, dp_hol, dm_hol, set.objVal


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


# Holistic Optimization without target DER LOOP
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


# Stepwise Optimization LOOP
def stepwise_optimize_forall(I, T, S, R, K, K0, P_DA, P_RT, P_PN, RP, RM, BP, BM, total_demand_without, total_supply_without, M1):

    x_part, yp_part, ym_part, z_part, zc_part, zd_part, dp_part, dm_part, up_part, um_part, wp_part, wm_part, obj_part = [], [], [], [], [], [], [], [], [], [], [], [], []

    for target_i in tqdm(range(I), desc="Optimizing Stepwise for each target_i"):
        model = gp.Model(f"Stepwise_Internal_Optimization_{target_i}")
        model.setParam("OutputFlag", 0)
        model.setParam("MIPGap", 1e-7)

        x = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="x")
        yp = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_plus")
        ym = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_minus")
        z = model.addVars(T + 1, S, vtype=GRB.CONTINUOUS, lb=0, name="z")
        zc = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="z_charge")
        zd = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="z_discharge")
        dp = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="d_plus")
        dm = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name="d_minus")

        p1  = model.addVars(T, S, vtype=GRB.BINARY, name="p1")
        p2  = model.addVars(T, S, vtype=GRB.BINARY, name="p2")
        p3  = model.addVars(T, S, vtype=GRB.BINARY, name="p3")
        p4  = model.addVars(T, S, vtype=GRB.BINARY, name="p4")
        
        wp, up = {}, {}
        wm, um = {}, {}
        for t, s in product(range(T), range(S)):
            Bp = BP[target_i, t, s]
            Bm = BM[target_i, t, s]
            for b in range(Bp):
                wp[t, s, b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_plus_{t}_{s}_{b}")
                up[t, s, b] = model.addVar(vtype=GRB.BINARY, name=f"u_plus_{t}_{s}_{b}")
            for b in range(Bm):
                wm[t, s, b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_minus_{t}_{s}_{b}")
                um[t, s, b] = model.addVar(vtype=GRB.BINARY, name=f"u_minus_{t}_{s}_{b}")

        model.update()

        obj = gp.quicksum(P_DA[t] * x[t] for t in range(T)) \
            + gp.quicksum((1/S) * (P_RT[t, s] * yp[t, s] - P_PN[t] * ym[t, s]) for t in range(T) for s in range(S)) \
            + gp.quicksum((1/S) * (
                gp.quicksum(
                    RP[target_i, t, s][b][1] * (wp[t, s, b] + up[t, s, b] * RP[target_i, t, s][b][0])
                    for b in range(BP[target_i, t, s])
                ) -
                gp.quicksum(
                    RM[target_i, t, s][b][1] * (wm[t, s, b] + um[t, s, b] * RM[target_i, t, s][b][0])
                    for b in range(BM[target_i, t, s])
                )
            ) for t in range(T) for s in range(S))

        model.setObjective(obj, GRB.MAXIMIZE)

        for t, s in product(range(T), range(S)):
            model.addConstr(R[target_i, t, s] - x[t] == yp[t, s] - ym[t, s] + zc[t, s] - zd[t, s] + dp[t, s] - dm[t, s])
            model.addConstr(yp[t, s] + dp[t, s] + zc[t, s] <= R[target_i, t, s] + zd[t, s])
            model.addConstr(z[t + 1, s] == z[t, s] + zc[t, s] - zd[t, s])
            model.addConstr(zd[t, s] <= z[t, s])
            model.addConstr(zc[t, s] <= K[target_i] - z[t, s])
            model.addConstr(z[t, s] <= K[target_i])
            model.addConstr(z[t, s] >= 0)
            model.addConstr(yp[t, s] <= M1 * p1[t, s])
            model.addConstr(ym[t, s] <= M1 * (1 - p1[t, s]))
            model.addConstr(ym[t, s] <= M1 * p2[t, s])
            model.addConstr(zc[t, s] <= M1 * (1 - p2[t, s]))
            model.addConstr(zc[t, s] <= M1 * p3[t, s])
            model.addConstr(zd[t, s] <= M1 * (1 - p3[t, s]))
            model.addConstr(dp[t, s] <= M1 * p4[t, s])
            model.addConstr(dm[t, s] <= M1 * (1 - p4[t, s]))

        for s in range(S):
            model.addConstr(z[0, s] == K0[target_i])

        for t, s in product(range(T), range(S)):
            model.addConstr(dp[t, s] == gp.quicksum(wp[t, s, bp] + up[t, s, bp] * RP[target_i, t, s][bp][0]
                                                    for bp in range(BP[target_i, t, s])))
            model.addConstr(dm[t, s] == gp.quicksum(wm[t, s, bm] + um[t, s, bm] * RM[target_i, t, s][bm][0]
                                                    for bm in range(BM[target_i, t, s])))

            model.addConstr(gp.quicksum(up[t, s, bp] for bp in range(BP[target_i, t, s])) <= 1)
            model.addConstr(gp.quicksum(um[t, s, bm] for bm in range(BM[target_i, t, s])) <= 1)
                
            for bp in range(BP[target_i, t, s]):
                if bp < BP[target_i, t, s] - 1:
                    WIDTH = RP[target_i, t, s][bp + 1][0] - RP[target_i, t, s][bp][0]
                else:
                    WIDTH = max(1e-6, total_demand_without[target_i, t, s] - RP[target_i, t, s][bp][0])
                model.addConstr(wp[t, s, bp] <= up[t, s, bp] * WIDTH)

            for bm in range(BM[target_i, t, s]):
                if bm < BM[target_i, t, s] - 1:
                    WIDTH = RM[target_i, t, s][bm + 1][0] - RM[target_i, t, s][bm][0]
                else:
                    WIDTH = max(1e-6, total_supply_without[target_i, t, s] - RM[target_i, t, s][bm][0])
                model.addConstr(wm[t, s, bm] <= um[t, s, bm] * WIDTH)

        model.optimize()

        x_part.append(np.array([x[t].X for t in range(T)]))
        yp_part.append(np.array([[yp[t, s].X for s in range(S)] for t in range(T)]))
        ym_part.append(np.array([[ym[t, s].X for s in range(S)] for t in range(T)]))
        z_part.append(np.array([[z[t, s].X for s in range(S)] for t in range(T + 1)]))
        zc_part.append(np.array([[zc[t, s].X for s in range(S)] for t in range(T)]))
        zd_part.append(np.array([[zd[t, s].X for s in range(S)] for t in range(T)]))
        dp_part.append(np.array([[dp[t, s].X for s in range(S)] for t in range(T)]))
        dm_part.append(np.array([[dm[t, s].X for s in range(S)] for t in range(T)]))
        obj_part.append(model.objVal)
        
        MAX_BP = np.max(BP[target_i])
        MAX_BM = np.max(BM[target_i])

        up_array = np.full((T, S, MAX_BP), np.nan)
        um_array = np.full((T, S, MAX_BM), np.nan)
        wp_array = np.full((T, S, MAX_BP), np.nan)
        wm_array = np.full((T, S, MAX_BM), np.nan)

        for t, s in product(range(T), range(S)):
            for bp in range(BP[target_i, t, s]):
                up_array[t, s, bp] = up[t, s, bp].X
                wp_array[t, s, bp] = wp[t, s, bp].X
            for bm in range(BM[target_i, t, s]):
                um_array[t, s, bm] = um[t, s, bm].X
                wm_array[t, s, bm] = wm[t, s, bm].X

        up_part.append(up_array)
        um_part.append(um_array)
        wp_part.append(wp_array)
        wm_part.append(wm_array)

    RP_CLEARED = np.full((I, T, S), np.nan)
    RM_CLEARED = np.full((I, T, S), np.nan)

    for target_i in range(I):
        for s, t in product(range(S), range(T)):
            for bp in range(BP[target_i, t, s]):
                if round(up_part[target_i][t, s, bp]) == 1:
                    RP_CLEARED[target_i, t, s] = RP[target_i, t, s][bp][1]
                    break
            for bm in range(BM[target_i, t, s]):
                if round(um_part[target_i][t, s, bm]) == 1:
                    RM_CLEARED[target_i, t, s] = RM[target_i, t, s][bm][1]
                    break
    return x_part, yp_part, ym_part, z_part, zc_part, zd_part, dp_part, dm_part, up_part, um_part, wp_part, wm_part, obj_part, RP_CLEARED, RM_CLEARED


