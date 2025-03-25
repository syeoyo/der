import gurobipy as gp
from gurobipy import GRB
from itertools import product
import numpy as np

def only(I, T, S, R, P_DA, P_RT, P_PN, K, K0, M1):
    
    only = gp.Model("only")
    # only.setParam("MIPGap", 1e-7)

    x = only.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0, name="x")
    yp = only.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_plus")
    ym = only.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_minus")
    z = only.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="z")
    zc = only.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="z_charge")
    zd = only.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="z_discharge")
    zeta = only.addVars(I, T, S, vtype=GRB.BINARY, name="zeta")
    delta = only.addVars(I, T, S, vtype=GRB.BINARY, name="delta")
    only.update()

    obj = gp.quicksum(P_DA[t] * x[i, t] for i, t in product(range(I), range(T))) \
        + gp.quicksum(1 / S * (P_RT[t, s] * yp[i, t, s] - P_PN[t] * ym[i, t, s])
                      for i, t, s in product(range(I), range(T), range(S)))
    only.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(I), range(T), range(S)):
        only.addConstr(R[i, t, s] - x[i, t] == yp[i, t, s] - ym[i, t, s] + zc[i, t, s] - zd[i, t, s])
        only.addConstr(R[i, t, s] + zd[i, t, s] >= yp[i, t, s] + zc[i, t, s])
        only.addConstr(ym[i, t, s] <= M1 * delta[i, t, s])
        only.addConstr(zc[i, t, s] <= M1 * (1 - delta[i, t, s]))
        only.addConstr(zc[i, t, s] <= M1 * zeta[i, t, s])
        only.addConstr(zd[i, t, s] <= M1 * (1 - zeta[i, t, s]))
        only.addConstr(z[i, t, s] <= K[i])
        only.addConstr(z[i, t, s] >= 0)

    for i, s in product(range(I), range(S)):
        only.addConstr(z[i, 0, s] == K0[i] + zc[i, 0, s] - zd[i, 0, s])
        only.addConstr(zd[i, 0, s] <= K0[i])
    for i, t, s in product(range(I), range(1, T), range(S)):
        only.addConstr(z[i, t, s] == z[i, t - 1, s] + zc[i, t, s] - zd[i, t, s])
        only.addConstr(z[i, t - 1, s] >= zd[i, t, s])

    only.optimize()

    if only.status == GRB.OPTIMAL:
        print(f"✅ Optimal solution found! Objective value: {only.objVal:.2f}")
    else:
        print("❌ No optimal solution found.")

    results = {
        "objval": only.objVal if only.status == GRB.OPTIMAL else None,
        "x": np.array([[x[i, t].X for t in range(T)] for i in range(I)]),
        "yp": np.array([[[yp[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
        "ym": np.array([[[ym[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
        "z": np.array([[[z[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
        "zc": np.array([[[zc[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
        "zd": np.array([[[zd[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    }

    return results

def agg(I, T, S, R, P_DA, P_RT, P_PN, K, K0, M2):
    
    agg = gp.Model("agg")
    agg.setParam("MIPGap", 1e-7)

    a = agg.addVars(T, vtype=GRB.CONTINUOUS, name="alpha")
    bp = agg.addVars(T, S, vtype=GRB.CONTINUOUS, name="beta_plus")
    bm = agg.addVars(T, S, vtype=GRB.CONTINUOUS, name="beta_minus")
    g = agg.addVars(T, S, vtype=GRB.CONTINUOUS, name="gamma")
    gc = agg.addVars(T, S, vtype=GRB.CONTINUOUS, name="gamma_charge")
    gd = agg.addVars(T, S, vtype=GRB.CONTINUOUS, name="gamma_discharge")
    phi = agg.addVars(T, S, vtype=GRB.BINARY, name="phi")  # zeta 대체
    eta = agg.addVars(T, S, vtype=GRB.BINARY, name="eta")

    agg.update()

    obj = gp.quicksum(P_DA[t] * a[t] for t in range(T)) + \
          gp.quicksum(1 / S * (P_RT[t, s] * bp[t, s] - P_PN[t] * bm[t, s])
                      for t, s in product(range(T), range(S)))

    agg.setObjective(obj, GRB.MAXIMIZE)

    for t, s in product(range(T), range(S)):
        agg.addConstr(gp.quicksum(R[i, t, s] for i in range(I)) - a[t] ==
                      bp[t, s] - bm[t, s] + gc[t, s] - gd[t, s])
        agg.addConstr(gp.quicksum(R[i, t, s] for i in range(I)) + gd[t, s] >=
                      bp[t, s] + gc[t, s])

        agg.addConstr(bm[t, s] <= M2 * eta[t, s])
        agg.addConstr(gc[t, s] <= M2 * (1 - eta[t, s]))
        agg.addConstr(gc[t, s] <= M2 * phi[t, s])
        agg.addConstr(gd[t, s] <= M2 * (1 - phi[t, s]))

        agg.addConstr(g[t, s] <= sum(K))
        agg.addConstr(g[t, s] >= 0)

    for s in range(S):
        agg.addConstr(g[0, s] == sum(K0) + gc[0, s] - gd[0, s])
        agg.addConstr(gd[0, s] <= sum(K0))

    for t, s in product(range(1, T), range(S)):
        agg.addConstr(g[t, s] == g[t - 1, s] + gc[t, s] - gd[t, s])
        agg.addConstr(gd[t, s] <= g[t - 1, s])

    agg.optimize()

    if agg.status == GRB.OPTIMAL:
        print(f"✅ Optimal solution found! Objective value: {agg.objVal:.2f}")
    else:
        print("❌ No optimal solution found.")

    results = {
        "objval": agg.objVal if agg.status == GRB.OPTIMAL else None,
        "a": np.array([a[t].X for t in range(T)]),
        "bp": np.array([[bp[t, s].X for s in range(S)] for t in range(T)]),
        "bm": np.array([[bm[t, s].X for s in range(S)] for t in range(T)]),
        "g": np.array([[g[t, s].X for s in range(S)] for t in range(T)]),
        "gc": np.array([[gc[t, s].X for s in range(S)] for t in range(T)]),
        "gd": np.array([[gd[t, s].X for s in range(S)] for t in range(T)])
    }
    
    return results
        

def set(I, T, S, R, P_DA, P_RT, P_PN, K, K0, M1, M2):
    set= gp.Model("set")
    # set.setParam("MIPGap", 1e-7)

    x = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="x")  # x_{it}
    yp = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_plus")
    ym = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="y_minus")
    zc = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="z_charge")
    zd = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="z_discharge")
    z = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="z_soc")
    d = set.addVars(I, I, T, S, vtype=GRB.CONTINUOUS, lb=0, name="d")
    zeta = set.addVars(I, T, S, vtype=GRB.BINARY, name="zeta")
    delta = set.addVars(I, T, S, vtype=GRB.BINARY, name="delta")
    eta = set.addVars(T, S, vtype=GRB.BINARY, name="eta")
    a = set.addVars(T, vtype=GRB.CONTINUOUS, name="alpha")
    bp = set.addVars(T, S, vtype=GRB.CONTINUOUS, name="beta_plus")
    bm = set.addVars(T, S, vtype=GRB.CONTINUOUS, name="beta_minus")
    gc = set.addVars(T, S, vtype=GRB.CONTINUOUS, name="gamma_charge")
    gd = set.addVars(T, S, vtype=GRB.CONTINUOUS, name="gamma_discharge")
    g = set.addVars(T, S, vtype=GRB.CONTINUOUS, name="gamma")
    ep = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="e_plus")
    em = set.addVars(I, T, S, vtype=GRB.CONTINUOUS, name="e_minus")

    obj = gp.quicksum(P_DA[t] * a[t] for t in range(T)) + \
        gp.quicksum((1 / S) * (P_RT[t, s] * bp[t, s] - P_PN[t] * bm[t, s]) 
                    for t in range(T) for s in range(S))
        
    set.setObjective(obj, GRB.MAXIMIZE)

    for i, t, s in product(range(I), range(T), range(S)):
        set.addConstr(R[i, t, s] - x[i, t, s] == yp[i, t, s] - ym[i, t, s] + zc[i, t, s] - zd[i, t, s])
        set.addConstr(R[i, t, s] + zd[i, t, s] >= yp[i, t, s] + zc[i, t, s])
        set.addConstr(ym[i, t, s] <= M1 * delta[i, t, s])
        set.addConstr(zc[i, t, s] <= M1 * (1 - delta[i, t, s]))
        set.addConstr(zc[i, t, s] <= M1 * zeta[i, t, s])
        set.addConstr(zd[i, t, s] <= M1 * (1 - zeta[i, t, s]))
        set.addConstr(z[i, t, s] <= K[i])
        set.addConstr(z[i, t, s] >= 0)
        set.addConstr(z[i, t, s] >= zd[i, t, s])

    for i, s in product(range(I), range(S)):
        set.addConstr(z[i, 0, s] == K0[i] + zc[i, 0, s] - zd[i, 0, s])
        set.addConstr(zd[i, 0, s] <= K0[i])
    for i, t, s in product(range(I), range(1, T), range(S)):
        set.addConstr(z[i, t, s] == z[i, t - 1, s] + zc[i, t, s] - zd[i, t, s])
        set.addConstr(z[i, t-1, s] >= zd[i, t, s])

    for t, s in product(range(T), range(S)):
        set.addConstr(a[t] == gp.quicksum(x[i, t, s] for i in range(I)))
        set.addConstr(bp[t, s] == gp.quicksum(ep[i, t, s] for i in range(I)))
        set.addConstr(bm[t, s] == gp.quicksum(em[i, t, s] for i in range(I)))
        set.addConstr(g[t, s] == gp.quicksum(z[i, t, s] for i in range(I)))
        set.addConstr(gc[t, s] == gp.quicksum(zc[i, t, s] for i in range(I)))
        set.addConstr(gd[t, s] == gp.quicksum(zd[i, t, s] for i in range(I)))
        set.addConstr(gp.quicksum(R[i, t, s] for i in range(I)) - a[t] == bp[t, s] - bm[t, s] + gc[t, s] - gd[t, s])
        set.addConstr(gp.quicksum(R[i, t, s] for i in range(I)) + gd[t, s] >= bp[t, s] + gc[t, s])
        set.addConstr(bm[t, s] <= M2 * eta[t, s])
        set.addConstr(gc[t, s] <= M2 * (1 - eta[t, s]))
        set.addConstr(g[t, s] <= sum(K))
        set.addConstr(g[t, s] >= 0)
        set.addConstr(g[t, s] >= gd[t, s])

    for s in range(S):
        set.addConstr(g[0, s] == sum(K0) + gc[0, s] - gd[0, s])
        set.addConstr(gd[0, s] <= sum(K0))
    for t, s in product(range(1, T), range(S)):
        set.addConstr(g[t, s] == g[t - 1, s] + gc[t, s] - gd[t, s])
        set.addConstr(gd[t, s] <= g[t - 1, s])

    for i, t, s in product(range(I), range(T), range(S)):
        set.addConstr(ep[i, t, s] == yp[i, t, s] + zd[i, t, s] - zc[i, t, s] - gp.quicksum(d[i, j, t, s] for j in range(I) if j != i))
        set.addConstr(em[i, t, s] == ym[i, t, s] - zd[i, t, s] + zc[i, t, s] - gp.quicksum(d[j, i, t, s] for j in range(I) if j != i))
        set.addConstr(gp.quicksum(d[i, j, t, s] for j in range(I) if j != i) <= yp[i, t, s] + zd[i, t, s] - zc[i, t, s])
        set.addConstr(gp.quicksum(d[j, i, t, s] for j in range(I) if j != i) <= ym[i, t, s] - zd[i, t, s] + zc[i, t, s])
        set.addConstr(d[i, i, t, s] == 0)
        
    set.optimize()

    if set.status == GRB.OPTIMAL:
        print(f"Optimal solution found! Objective value: {set.objVal}")
    else:
        print("No optimal solution found.")

    results = {
    "objval": set.objVal if set.status == GRB.OPTIMAL else None,
    "x": np.array([[[x[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    "yp": np.array([[[yp[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    "ym": np.array([[[ym[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    "z":  np.array([[[z[i, t, s].X  for s in range(S)] for t in range(T)] for i in range(I)]),
    "zc": np.array([[[zc[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    "zd": np.array([[[zd[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    "d":  np.array([[[[d[i, j, t, s].X for s in range(S)] for t in range(T)] for j in range(I)] for i in range(I)]),
    "a":  np.array([a[t].X for t in range(T)]),
    "bp": np.array([[bp[t, s].X for s in range(S)] for t in range(T)]),
    "bm": np.array([[bm[t, s].X for s in range(S)] for t in range(T)]),
    "g":  np.array([[g[t, s].X  for s in range(S)] for t in range(T)]),
    "gc": np.array([[gc[t, s].X for s in range(S)] for t in range(T)]),
    "gd": np.array([[gd[t, s].X for s in range(S)] for t in range(T)]),
    "ep": np.array([[[ep[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)]),
    "em": np.array([[[em[i, t, s].X for s in range(S)] for t in range(T)] for i in range(I)])
    }
    
    return results