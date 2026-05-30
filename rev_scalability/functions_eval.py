import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm


def calculate_total_profit_best_case(x_part, yp_part, ym_part, dp_part, dm_part, dp_extracted, dm_extracted, RP_CLEARED, RM_CLEARED, P_DA, P_RT, P_PN, T, S, I):
    BEST_CASE_PROFIT = np.zeros(I)
    
    for target_i in range(I):
        profit = 0
        for t in range(T):
            for s in range(S):
                basic_profit = (x_part[target_i][t] * P_DA[t] + 
                              yp_part[target_i][t, s] * P_RT[t, s] - 
                              ym_part[target_i][t, s] * P_PN[t, s])

                # dp_others = sum(dp_part[i][t, s] for i in range(I) if i != target_i)
                # dm_others = sum(dm_part[i][t, s] for i in range(I) if i != target_i)

                dp_others = sum(dp_extracted[target_i][j, t, s] for j in range(I-1))
                dm_others = sum(dm_extracted[target_i][j, t, s] for j in range(I-1))
                
                sold_internal = min(dp_part[target_i][t, s], dm_others)
                sold_rt = max(0, dp_part[target_i][t, s] - dm_others)
                
                bought_internal = min(dm_part[target_i][t, s], dp_others)
                bought_penalty = max(0, dm_part[target_i][t, s] - dp_others)
                
                internal_profit = (RP_CLEARED[target_i, t, s] * sold_internal + 
                                 P_RT[t, s] * sold_rt - 
                                 RM_CLEARED[target_i, t, s] * bought_internal - 
                                 P_PN[t, s] * bought_penalty)
                
                profit += (basic_profit + internal_profit) / S
        
        BEST_CASE_PROFIT[target_i] = profit
    
    return BEST_CASE_PROFIT


def calculate_total_profit_worst_case(x_part, yp_part, ym_part, dp_part, dm_part, dp_extracted, dm_extracted, RP_CLEARED, RM_CLEARED, P_DA, P_RT, P_PN, T, S, I):
    WORST_CASE_PROFIT = np.zeros(I)
    
    for target_i in range(I):
        profit = 0
        for t in range(T):
            for s in range(S):
                basic_profit = (x_part[target_i][t] * P_DA[t] + 
                              yp_part[target_i][t, s] * P_RT[t, s] - 
                              ym_part[target_i][t, s] * P_PN[t, s])

                # dp_others = sum(dp_part[i][t, s] for i in range(I) if i != target_i)
                # dm_others = sum(dm_part[i][t, s] for i in range(I) if i != target_i)

                dp_others = sum(dp_extracted[target_i][j, t, s] for j in range(I-1))
                dm_others = sum(dm_extracted[target_i][j, t, s] for j in range(I-1))
                
                net_others_demand = max(0, dm_others - dp_others)  # others' net demand
                net_others_supply = max(0, dp_others - dm_others)  # others' net supply
                
                sold_internal = min(dp_part[target_i][t, s], net_others_demand)
                sold_rt = max(0, dp_part[target_i][t, s] - net_others_demand)
                
                bought_internal = min(dm_part[target_i][t, s], net_others_supply)
                bought_penalty = max(0, dm_part[target_i][t, s] - net_others_supply)
                
                internal_profit = (RP_CLEARED[target_i, t, s] * sold_internal + 
                                 P_RT[t, s] * sold_rt - 
                                 RM_CLEARED[target_i, t, s] * bought_internal - 
                                 P_PN[t, s] * bought_penalty)
                
                profit += (basic_profit + internal_profit) / S
        
        WORST_CASE_PROFIT[target_i] = profit
    
    return WORST_CASE_PROFIT

def calculate_total_profit_no_virtual_price(x_part, yp_part, ym_part, dp_part, dm_part, RP_CLEARED, RM_CLEARED, P_DA, P_RT, P_PN, T, S, I):
    NO_VIRTUAL_PRICE_PROFIT = np.zeros(I)
    
    for target_i in range(I):
        profit = 0
        for t in range(T):
            for s in range(S):
                basic_profit = x_part[target_i][t] * P_DA[t] + (yp_part[target_i][t, s]+dp_part[target_i][t, s]) * P_RT[t, s] - (ym_part[target_i][t, s]+dm_part[target_i][t, s]) * P_PN[t, s]

                profit += basic_profit / S
        
        NO_VIRTUAL_PRICE_PROFIT[target_i] = profit
    
    return NO_VIRTUAL_PRICE_PROFIT


def evaluate_and_visualize_profits(x_ind, yp_ind, ym_ind, P_DA, P_RT, P_PN, NO_VIRTUAL_PRICE_PROFIT, WORST_CASE_PROFIT, BEST_CASE_PROFIT, T, S, I):

    individual_profit = []
    
    for i in range(I):
        profit = 0
        for t in range(T):
            for s in range(S):
                profit += x_ind[i][t] * P_DA[t] + yp_ind[i][t, s] * P_RT[t, s] - ym_ind[i][t, s] * P_PN[t, s]
        individual_profit.append(profit / S)

    df_profits = pd.DataFrame({
        'Individual_Profit': individual_profit,
        'No_Virtual_Price': NO_VIRTUAL_PRICE_PROFIT,
        'Worst_Case_Profit': WORST_CASE_PROFIT,
        'Best_Case_Profit': BEST_CASE_PROFIT
    }, index=pd.Index([f'i={i}' for i in range(I)], name='Participant'))
    
    # 시각화
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(I)
    width = 0.2
    
    plt.bar(x_pos - 1.5*width, df_profits['Individual_Profit'], width, label='Individual Profit', alpha=0.8)
    plt.bar(x_pos - 0.5*width, df_profits['No_Virtual_Price'], width, label='No Virtual Price', alpha=0.8)
    plt.bar(x_pos + 0.5*width, df_profits['Worst_Case_Profit'], width, label='Worst Case Profit', alpha=0.8)
    plt.bar(x_pos + 1.5*width, df_profits['Best_Case_Profit'], width, label='Best Case Profit', alpha=0.8)
    
    plt.xlabel('Participant (i)', fontsize=12)
    plt.ylabel('Profit', fontsize=12)
    plt.title('Profit Comparison by Participant', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, [f'i={i}' for i in range(I)])
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(df_profits)


def compare_step_vs_ind(x_part, yp_part, ym_part, dp_part, dm_part, x_ind, yp_ind, ym_ind, 
                               RP_CLEARED, RM_CLEARED, P_DA, P_RT, P_PN, T, S, I):

    for target_i in range(I):
        header = (
            f"{'t':>3} | "
            f"{'P_DA':>8} {'x_step':>8} {'x_ind':>8} || "
            f"{'RHO_P':>15} {'dp_step':>8} {'P_RT':>8} {'yp_step':>8} {'yp_ind':>8} || "
            f"{'RHO_M':>15} {'dm_step':>8} {'P_PN':>8} {'ym_step':>8} {'ym_ind':>8}"
        )
        print(f"\n타겟 참여자 {target_i}번의 시간대별 입찰량 비교 (시나리오 평균):")
        print("-" * 145)
        print(header)
        print("-" * 145)

        sum_x_step = sum_x_ind = sum_dp_step = sum_yp_step = sum_yp_ind = 0
        sum_dm_step = sum_ym_step = sum_ym_ind = 0

        for t in range(T):
            x_step = x_part[target_i][t] 
            x_indv = x_ind[target_i][t]
            dp_step = np.mean(dp_part[target_i][t, :]) 
            yp_step = np.mean(yp_part[target_i][t, :])
            yp_indv = np.mean(yp_ind[target_i][t, :])
            dm_step = np.mean(dm_part[target_i][t, :]) 
            ym_step = np.mean(ym_part[target_i][t, :]) 
            ym_indv = np.mean(ym_ind[target_i][t, :])
            rho_p = np.mean(RP_CLEARED[target_i, t, :]) 
            rho_m = np.mean(RM_CLEARED[target_i, t, :])
            p_rt = np.mean(P_RT[t, :])
            p_pn = np.mean(P_PN[t, :])

            sum_x_step += x_step 
            sum_x_ind += x_indv 
            sum_dp_step += dp_step 
            sum_yp_step += yp_step 
            sum_yp_ind += yp_indv
            sum_dm_step += dm_step 
            sum_ym_step += ym_step 
            sum_ym_ind += ym_indv

            print(
                f"{t:>3} | "
                f"{P_DA[t]:>8.2f} {x_step:>8.2f} {x_indv:>8.2f} || "
                f"{rho_p:>15.2f} {dp_step:>8.2f} {p_rt:>8.2f} {yp_step:>8.2f} {yp_indv:>8.2f} || "
                f"{rho_m:>15.2f} {dm_step:>8.2f} {p_pn:>8.2f} {ym_step:>8.2f} {ym_indv:>8.2f}"
            )

        print("-" * 145)
        print(
            f"{'합계':>3} | "
            f"{'':>8} {sum_x_step:>8.2f} {sum_x_ind:>8.2f} || "
            f"{'':>15} {sum_dp_step:>8.2f} {'':>8} {sum_yp_step:>8.2f} {sum_yp_ind:>8.2f} || "
            f"{'':>15} {sum_dm_step:>8.2f} {'':>8} {sum_ym_step:>8.2f} {sum_ym_ind:>8.2f}"
        )


def print_summary_only(x_part, yp_part, ym_part, dp_part, dm_part, x_ind, yp_ind, ym_ind, T, S, I):
    for target_i in range(I):

        sum_x_step = sum_x_ind = sum_dp_step = sum_yp_step = sum_yp_ind = 0
        sum_dm_step = sum_ym_step = sum_ym_ind = 0

        for t in range(T):
            x_step = x_part[target_i][t] 
            x_indv = x_ind[target_i][t]
            dp_step = np.mean(dp_part[target_i][t, :]) 
            yp_step = np.mean(yp_part[target_i][t, :])
            yp_indv = np.mean(yp_ind[target_i][t, :])
            dm_step = np.mean(dm_part[target_i][t, :]) 
            ym_step = np.mean(ym_part[target_i][t, :]) 
            ym_indv = np.mean(ym_ind[target_i][t, :])

            sum_x_step += x_step 
            sum_x_ind += x_indv 
            sum_dp_step += dp_step 
            sum_yp_step += yp_step 
            sum_yp_ind += yp_indv
            sum_dm_step += dm_step 
            sum_ym_step += ym_step 
            sum_ym_ind += ym_indv

        header = (
            f"{'t':>3} | "
            f"{'P_DA':>8} {'x_step':>8} {'x_ind':>8} || "
            f"{'RHO_P':>15} {'dp_step':>8} {'P_RT':>8} {'yp_step':>8} {'yp_ind':>8} || "
            f"{'RHO_M':>15} {'dm_step':>8} {'P_PN':>8} {'ym_step':>8} {'ym_ind':>8}"
        )
        print(header)
        print("-" * 145)
        print(
            f"{target_i:>3} | "
            f"{'':>8} {sum_x_step:>8.2f} {sum_x_ind:>8.2f} || "
            f"{'':>15} {sum_dp_step:>8.2f} {'':>8} {sum_yp_step:>8.2f} {sum_yp_ind:>8.2f} || "
            f"{'':>15} {sum_dm_step:>8.2f} {'':>8} {sum_ym_step:>8.2f} {sum_ym_ind:>8.2f}"
        )


def compare_step_vs_hol(x_hol, ep_hol, em_hol, dp_hol, dm_hol, x_part, yp_part, ym_part, dp_part, dm_part, T, S, I):
    print("=" * 110)
    print("STEPWISE vs HOLISTIC 모델 비교 (시나리오 평균)")
    print("=" * 110)
    
    for target_i in range(I):
        print(f"\n타겟 참여자 {target_i}번:")
        print("-" * 110)
        print(f"{'t':>2} | {'x_hol':>8} {'x_step':>8} | {'yp_hol':>8} {'yp_step':>8} | {'ym_hol':>8} {'ym_step':>8} | {'dp_hol':>8} {'dp_step':>8} | {'dm_hol':>8} {'dm_step':>8}")
        print("-" * 110)
        
        for t in range(T):
            x_step_avg = x_part[target_i][t]
            yp_step_avg = np.mean(yp_part[target_i][t, :])
            ym_step_avg = np.mean(ym_part[target_i][t, :])
            dp_step_avg = np.mean(dp_part[target_i][t, :])
            dm_step_avg = np.mean(dm_part[target_i][t, :])
            
            x_hol_avg = x_hol[target_i, t]
            ep_hol_avg = np.mean(ep_hol[target_i, t, :])
            em_hol_avg = np.mean(em_hol[target_i, t, :])
            dp_hol_avg = np.mean(dp_hol[target_i, t, :])
            dm_hol_avg = np.mean(dm_hol[target_i, t, :])
            
            print(f"{t:>2} | {x_hol_avg:>8.2f} {x_step_avg:>8.2f} | {ep_hol_avg:>8.2f} {yp_step_avg:>8.2f} | {em_hol_avg:>8.2f} {ym_step_avg:>8.2f} | {dp_hol_avg:>8.2f} {dp_step_avg:>8.2f} | {dm_hol_avg:>8.2f} {dm_step_avg:>8.2f}")

def calculate_aggregated_profit_evaluation(x_ind, yp_ind, ym_ind, NO_VIRTUAL_PRICE_PROFIT, WORST_CASE_PROFIT, BEST_CASE_PROFIT, a_hol, bp_hol, bm_hol, P_DA, P_RT, P_PN, T, S, I):
    individual_profit = 0
    for i in range(I):
            for t in range(T):
                individual_profit += (x_ind[i][t] * P_DA[t] + 
                                    np.mean([yp_ind[i][t, s] * P_RT[t, s] for s in range(S)]) - 
                                    np.mean([ym_ind[i][t, s] * P_PN[t, s] for s in range(S)]))

    stepwise_profit_no_virtual_price = sum(NO_VIRTUAL_PRICE_PROFIT)
    stepwise_profit_worst_case = sum(WORST_CASE_PROFIT)
    stepwise_profit_best_case = sum(BEST_CASE_PROFIT)

    aggregation_profit = 0
    for t in range(T):
        aggregation_profit += (a_hol[t] * P_DA[t] + 
                                np.mean([bp_hol[t, s] * P_RT[t, s] for s in range(S)]) - 
                                np.mean([bm_hol[t, s] * P_PN[t, s] for s in range(S)]))

    df_profit = pd.DataFrame({
            "Profit Type": ["Individual Profit", "Stepwise Profit (No Virtual Price)", 
                        "Stepwise Profit (Worst Case)", "Stepwise Profit (Best Case)", "Aggregation Profit"],
            "Profit": [individual_profit, stepwise_profit_no_virtual_price, 
                    stepwise_profit_worst_case, stepwise_profit_best_case, aggregation_profit]
        })
        
    print(df_profit)

def compare_holall_vs_holwithstep(x_hol, ep_hol, em_hol, dp_hol, dm_hol, x_part, yp_part, ym_part, dp_part, dm_part, T, S, I):
    print(f"전체 holistic 최적화 vs 본인만 stepwise 최적화 + 나머지는 (본인포함) holistic")
    for i in range(I):
        print(f"{'i':>2} | {'t':>2} | {'x_hol_sum':>10} {'x_part_sum':>12} | {'ep_hol_mean':>12} {'yp_part_sum':>12} | {'em_hol_mean':>12} {'ym_part_sum':>12} | {'dp_hol_mean':>12} {'dp_part_sum':>12} | {'dm_hol_mean':>12} {'dm_part_sum':>12}")
        print("-" * 145)
        
        total_x_hol_sum = 0
        total_x_part_sum = 0
        total_ep_hol_mean = 0
        total_yp_part_sum = 0
        total_em_hol_mean = 0
        total_ym_part_sum = 0
        total_dp_hol_mean = 0
        total_dp_part_sum = 0
        total_dm_hol_mean = 0
        total_dm_part_sum = 0
        
        for t in range(T):
            x_hol_sum = np.sum([x_hol[j, t] for j in range(I)])
            x_part_sum = x_part[i][t] + np.sum([x_hol[j, t] for j in range(I) if j != i])

            ep_hol_mean = np.sum([np.mean(ep_hol[j, t, :]) for j in range(I)])
            yp_part_mean = np.mean(yp_part[i][t, :])
            yp_part_sum = yp_part_mean + np.sum([np.mean(ep_hol[j, t, :]) for j in range(I) if j != i])

            em_hol_mean = np.sum([np.mean(em_hol[j, t, :]) for j in range(I)])
            ym_part_mean = np.mean(ym_part[i][t, :])
            ym_part_sum = ym_part_mean +  np.sum([np.mean(em_hol[j, t, :]) for j in range(I) if j != i])

            dp_hol_mean = np.sum([np.mean(dp_hol[j, t, :]) for j in range(I)])
            dp_part_mean = np.mean(dp_part[i][t, :])
            dp_part_sum = dp_part_mean + np.sum([np.mean(dp_hol[j, t, :]) for j in range(I) if j != i])

            dm_hol_mean = np.sum([np.mean(dm_hol[j, t, :]) for j in range(I)])
            dm_part_mean = np.mean(dm_part[i][t, :])
            dm_part_sum = dm_part_mean + np.sum([np.mean(dm_hol[j, t, :]) for j in range(I) if j != i])

            total_x_hol_sum += x_hol_sum
            total_x_part_sum += x_part_sum
            total_ep_hol_mean += ep_hol_mean
            total_yp_part_sum += yp_part_sum
            total_em_hol_mean += em_hol_mean
            total_ym_part_sum += ym_part_sum
            total_dp_hol_mean += dp_hol_mean
            total_dp_part_sum += dp_part_sum
            total_dm_hol_mean += dm_hol_mean
            total_dm_part_sum += dm_part_sum

            print(f"{i:>2} | {t:>2} | {x_hol_sum:>10.2f} {x_part_sum:>12.2f} | {ep_hol_mean:>12.2f} {yp_part_sum:>12.2f} | {em_hol_mean:>12.2f} {ym_part_sum:>12.2f} | {dp_hol_mean:>12.2f} {dp_part_sum:>12.2f} | {dm_hol_mean:>12.2f} {dm_part_sum:>12.2f}")

        print("-" * 145)
        print(f"Sum | -- | {total_x_hol_sum:>10.2f} {total_x_part_sum:>12.2f} | {total_ep_hol_mean:>12.2f} {total_yp_part_sum:>12.2f} | {total_em_hol_mean:>12.2f} {total_ym_part_sum:>12.2f} | {total_dp_hol_mean:>12.2f} {total_dp_part_sum:>12.2f} | {total_dm_hol_mean:>12.2f} {total_dm_part_sum:>12.2f}")
        print()


def compare_holall_vs_stepsum(a_hol, bp_hol, bm_hol, dp_hol, dm_hol, x_part, yp_part, ym_part, dp_part, dm_part, T, S, I):
    print(f"전체 holistic 최적화 vs 전체 stepwise 최적화")
    print("-" * 140)
    print(f"{'t':>2} | {'a_hol':>8} {'x_part_sum':>12} | {'bp_hol_avg':>12} {'yp_part_sum':>12} | {'bm_hol_avg':>12} {'ym_part_sum':>12} | {'dp_hol_sum':>12} {'dp_part_sum':>12} | {'dm_hol_sum':>12} {'dm_part_sum':>12}")
    print("-" * 140)

    # 합계를 위한 변수 초기화
    sum_a_hol = 0
    sum_x_part_sum = 0
    sum_bp_hol_avg = 0
    sum_yp_part_sum = 0
    sum_bm_hol_avg = 0
    sum_ym_part_sum = 0
    sum_dp_hol_sum = 0
    sum_dp_part_sum = 0
    sum_dm_hol_sum = 0
    sum_dm_part_sum = 0

    for t in range(T):
        bp_hol_avg = np.mean([bp_hol[t, s] for s in range(S)])
        bm_hol_avg = np.mean([bm_hol[t, s] for s in range(S)])
        
        x_part_sum = np.sum([x_part[target_i][t] for target_i in range(I)])
        yp_part_sum = np.mean([np.sum([yp_part[target_i][t, s] for target_i in range(I)]) for s in range(S)])
        ym_part_sum = np.mean([np.sum([ym_part[target_i][t, s] for target_i in range(I)]) for s in range(S)])
        dp_hol_sum = np.mean([np.sum([dp_hol[target_i][t, s] for target_i in range(I)]) for s in range(S)])
        dp_part_sum = np.mean([np.sum([dp_part[target_i][t, s] for target_i in range(I)]) for s in range(S)])
        dm_hol_sum = np.mean([np.sum([dm_hol[target_i][t, s] for target_i in range(I)]) for s in range(S)])
        dm_part_sum = np.mean([np.sum([dm_part[target_i][t, s] for target_i in range(I)]) for s in range(S)])
        
        # 합계 누적
        sum_a_hol += a_hol[t]
        sum_x_part_sum += x_part_sum
        sum_bp_hol_avg += bp_hol_avg
        sum_yp_part_sum += yp_part_sum
        sum_bm_hol_avg += bm_hol_avg
        sum_ym_part_sum += ym_part_sum
        sum_dp_hol_sum += dp_hol_sum
        sum_dp_part_sum += dp_part_sum
        sum_dm_hol_sum += dm_hol_sum
        sum_dm_part_sum += dm_part_sum
        
        print(
            f"{t:>2} | "
            f"{a_hol[t]:>8.2f} {x_part_sum:>12.2f} | "
            f"{bp_hol_avg:>12.2f} {yp_part_sum:>12.2f} | "
            f"{bm_hol_avg:>12.2f} {ym_part_sum:>12.2f} | "
            f"{dp_hol_sum:>12.2f} {dp_part_sum:>12.2f} | "
            f"{dm_hol_sum:>12.2f} {dm_part_sum:>12.2f}"
        )

    print("-" * 140)
    print(
        f"{'합계':>2} | "
        f"{sum_a_hol:>8.2f} {sum_x_part_sum:>12.2f} | "
        f"{sum_bp_hol_avg:>12.2f} {sum_yp_part_sum:>12.2f} | "
        f"{sum_bm_hol_avg:>12.2f} {sum_ym_part_sum:>12.2f} | "
        f"{sum_dp_hol_sum:>12.2f} {sum_dp_part_sum:>12.2f} | "
        f"{sum_dm_hol_sum:>12.2f} {sum_dm_part_sum:>12.2f}"
    )