# ADMM No-ADMM-Regularization Distributed Stochastic Coordination Benchmark

This folder contains the ADMM no-regularization ablation baseline for the paper
revision.

Unlike the main `admm` folder, this copy removes the original objective
regularization term only from:

- the DER-local ADMM subproblems.

The proposed/current centralized stochastic benchmark keeps the regularization
term because it is treated as part of the model contribution. The ADMM
augmented-Lagrangian penalty is also still kept because it is the algorithmic
consensus mechanism, not the original economic/contribution regularization.

Files:

- `ADMM_Distributed_Stochastic_Coordination_Benchmark.ipynb`: step-by-step notebook with markdown explanation, centralized solve, ADMM iterations, convergence plots, price comparison, and scalability experiment cells.
- `admm_stochastic_benchmark.py`: reusable implementation of the centralized stochastic QP and DER-wise stochastic consensus ADMM.

The current `0527.ipynb` model has one explicit market-level coupling constraint:

```text
sum_i dp[i,t,s] == sum_i dm[i,t,s]
```

The ADMM baseline decouples this with `q_i(t,s)=dp_i(t,s)-dm_i(t,s)`, local consensus copies, coordinator projection onto `sum_i consensus_q_i(t,s)=0`, and dual updates. Scenario-dependent internal prices are recovered as:

```text
lambda_admm(t,s) = rho * mean_i u_i(t,s)
P_internal(t,s) = -S * lambda_admm(t,s)
```

Start with the notebook's small settings before running the full paper grid:

```text
N_VALUES = [10]
S_VALUES = [10]
```

For the paper-scale overnight experiment requested for the revision, use:

```text
OVERNIGHT_N_VALUES = [10, 30, 50, 75, 100]
OVERNIGHT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 25]
OVERNIGHT_SCENARIOS = 100
```

The notebook calls `run_overnight_scalability(...)`, which saves one row after
each completed `(N, S=100, seed)` case:

```text
admm_noreg_overnight_results.csv
admm_noreg_overnight_seed_average.csv
```

The run can be resumed with `skip_completed=True`. The full grid can be slow
because every ADMM iteration solves one local stochastic QP per DER and
exchanges full `(T,S)` vectors.
