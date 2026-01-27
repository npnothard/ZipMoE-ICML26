# Copyright (c) 2026 <ZipMoE / Anonymous Team>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.



import math
import torch
import random
import numpy as np
from collections import Counter
import itertools
import pandas as pd


def CalculateProb(
    prob_distribution
):
    cache_pool_size = len(prob_distribution)

    output_distribution = [0.0] * (cache_pool_size + 1)
    output_distribution[0] = 1.0
    
    for p in prob_distribution:
        for j in range(cache_pool_size, 0 , -1):
            output_distribution[j] = output_distribution[j] * (1-p) + output_distribution[j-1] * p
        output_distribution[0] = output_distribution[0] * (1 - p)

    return output_distribution


def DelaySimu(
    k,
    num_full_cached,
    num_sm_cached,
    decompression_delay,
    SM_IO_delay,
    compression_ratio,
    num_file_chunks,
    tensors_per_expert,
    compute_pool_size
):
    num_compute_ops = num_file_chunks * ( k - num_full_cached ) * tensors_per_expert
    EXP_IO_DELAY = SM_IO_delay * compression_ratio / num_file_chunks
    ComputeBottleneck = (
        ( num_compute_ops / compute_pool_size ) * ( decompression_delay + EXP_IO_DELAY )
    )     
    IOBottleneck = tensors_per_expert * SM_IO_delay * ( k - num_full_cached - num_sm_cached ) + num_sm_cached * tensors_per_expert* SM_IO_delay * compression_ratio
    return max(ComputeBottleneck, IOBottleneck)


def CachePoolPlanningSimu(
    k,
    num_experts,
    num_sparse_layers,
    tensors_per_expert,
    num_elements_per_tensor,
    compression_ratio,
    num_file_chunks,
    compute_pool_size,
    device_memory_ratio,
    decompression_delay,
    SM_IO_delay,
    all_probs, 
    step_size = 0.01
):
    ratio_list = []
    simu_delay_list = []
    total_mem = device_memory_ratio * torch.cuda.get_device_properties(0).total_memory / num_sparse_layers
    sm_slot_size = tensors_per_expert * num_elements_per_tensor
    full_slot_size = sm_slot_size * 2 
    print(f"Total memory available per layer: {total_mem}")
    print(f"Max potential full slots: {int(total_mem/full_slot_size)}")
    minAvgDelay = float('inf')
    best_full_ratio = 0
    s = sum(all_probs)
    num_steps = int(1 / step_size)
    ratio_full_cache_list = [i * step_size for i in range(1,num_steps,1)]
    feasible = False

    for ratio_full in ratio_full_cache_list:
        current_probs_list = []
        ratio_sm = 1 - ratio_full
        full_cache_size = int((ratio_full * total_mem) / full_slot_size)
        sm_cache_size = int((ratio_sm * total_mem) / sm_slot_size)
        if full_cache_size + sm_cache_size > num_experts:
            continue
        if full_cache_size*num_sparse_layers < num_experts * 1 or sm_cache_size*num_sparse_layers < num_experts * 1:
            continue
        feasible = True
        p_full = [all_probs[idx] for idx in range(0, full_cache_size)]
        p_sm = [all_probs[idx] for idx in range(full_cache_size, full_cache_size + sm_cache_size)]
        current_probs_list = [p_full, p_sm]
        in_pool_probs = p_full + p_sm
        rest_probs = list((Counter(all_probs) - Counter(in_pool_probs)).elements())
        dists = [CalculateProb(p) for p in current_probs_list]
        dist_R = CalculateProb(rest_probs)
        total_dist = dists[0]
        for i in range(1, len(dists)):
            total_dist = np.convolve(total_dist, dists[i])
        total_dist = np.convolve(total_dist, dist_R)       
        p_total_k = total_dist[k] if k < len(total_dist) else 0
        if p_total_k <= 0:
            continue
        avgDelay = 0
        for full_hit in range(len(dists[0])):
            for sm_hit in range(len(dists[1])):
                if full_hit + sm_hit > k:
                    continue
                r = k - (full_hit + sm_hit)
                if r >= len(dist_R):
                    continue
                prob = (dists[0][full_hit] * dists[1][sm_hit] * dist_R[r]) / p_total_k
                delay = DelaySimu(
                    k, full_hit, sm_hit, 
                    decompression_delay, SM_IO_delay, 
                    compression_ratio, num_file_chunks, 
                    tensors_per_expert, compute_pool_size
                )
                avgDelay += prob * delay
        ratio_list.append(ratio_full)
        simu_delay_list.append(avgDelay)
        if avgDelay < minAvgDelay and ratio_full>=0.3:
            minAvgDelay = avgDelay
            best_full_ratio = ratio_full
        print(f"[ZipMoE Invocation Modeling] Ratio Full: {ratio_full:.2f} | Avg Delay: {avgDelay:.4f}")
    if minAvgDelay == float('inf'): 
        return 1.0 - step_size, feasible
    
    series = pd.Series(simu_delay_list)
    smooth_simu_delay_list = series.rolling(window=5, center=True, min_periods=1).mean()
    for i in range(len(smooth_simu_delay_list)):
        print(f"[ZipMoE Cost Modeling]: Smoothed Ratio Full = {ratio_list[i]},  Cost = {smooth_simu_delay_list[i]}")
    best_full_ratio_index = smooth_simu_delay_list.idxmin()
    best_full_ratio = ratio_list[best_full_ratio_index]

    return best_full_ratio, feasible


def ObtainDistFromTrace(
    batch_size,
    trace_path
):
    print("[ZipMoE Cache Pool Planning] Loading trace...")
    trace = torch.load(trace_path, map_location="cpu")
    num_prompts = len(trace)
    num_layers = len(trace[0])
    num_experts = len(trace[0][0])
    num_batches = num_prompts // batch_size
    many_batches_data = [0]*num_experts
    for b in range(num_batches):
        batch_data = [0]*num_experts
        for layer_id in range(num_layers):
            in_batch_invocation = [0] * num_experts
            for i in range(batch_size):
                in_batch_invocation = [ a + b for a,b in zip( in_batch_invocation , trace[b*batch_size + i][layer_id] ) ]
            in_batch_invocation.sort(reverse=True)
            batch_data = [ a + b for a,b in zip(batch_data,in_batch_invocation)]
        many_batches_data = [ a+b for a,b in zip(many_batches_data, batch_data) ]
    return [p/sum(many_batches_data) for p in many_batches_data]
    



def CoverageToWeight(
    target_probs,
    k, 
    max_iter=50,
    tol=1e-8
):
    # This function uses the algorithm proposed by:
    # X.-H. Chen et al., "Weighted Finite Population Sampling to Maximize Entropy" in Biometrika (1994)
    # To convert frequency obtained from trace to Poission sampling probabilities that complies with the frequency constraints.
    pi = np.array(target_probs)
    N = len(pi)
    if not np.isclose(np.sum(pi), k):
        raise ValueError(f"[ZipMoE Trace Error] Sum of target probabilities ({np.sum(pi)}) must equal k: ({k})")
    # Initialize w
    w = pi / (1.0 - pi + 1e-12) 
    for iteration in range(max_iter):
        # dp[j][m] is the weighted sum of selecting m elements in the j elements
        dp = np.zeros((N + 1, k + 1))
        dp[:, 0] = 1.0
        for j in range(1, N + 1):
            for m in range(1, k + 1):
                # State transition: E(j, m) = E(j-1, m) + w[j-1] * E(j-1, m-1)
                dp[j][m] = dp[j-1][m] + w[j-1] * dp[j-1][m-1]
        
        # Compute current_pi: pi_i = w_i * [E_{-i}(k-1) / E(k)]
        b_dp = np.zeros((N + 2, k + 1))
        b_dp[:, 0] = 1.0
        for j in range(N, 0, -1):
            for m in range(1, k + 1):
                b_dp[j][m] = b_dp[j+1][m] + w[j-1] * b_dp[j+1][m-1]
        
        current_pi = np.zeros(N)
        for i in range(N):
            # choose m elements from top i-1, and k-1-m elements from i+1 to end
            e_minus_i_k_minus_1 = 0
            for m in range(k):
                e_minus_i_k_minus_1 += dp[i][m] * b_dp[i+2][k-1-m]
            
            current_pi[i] = w[i] * e_minus_i_k_minus_1 / dp[N][k]
        
        # Update the weight via (7) in the paper:
        #  w_new = w_old * (target_pi / current_pi)
        diff = np.linalg.norm(current_pi - pi)
        if diff < tol:
            print(f"[Coverage To Weight] Converged in {iteration} iterations.")
            return w.tolist()
            
        w = w * (pi / (current_pi + 1e-12))
        
        # Normalization
        w /= np.mean(w)

    print("[Coverage To Weight] Did not reach convergence tolerance.")
    return w.tolist()



def WeightToProb(
    weight
):
    Prob = [w / (1 + w) for w in weight]
    return Prob#[p/sum(Prob) for p in Prob]


def scale_to_k(probs, k, tol=1e-9):
    pi = np.array(probs, dtype=float)
    if np.sum(pi) == 0:
        return pi
    pi = pi * (k / np.sum(pi))

    n = len(pi)
    remaining_indices = np.arange(n)
    remaining_k = float(k)
    while len(remaining_indices) > 0:
        current_sum = np.sum(pi[remaining_indices])
        if current_sum < tol:
            break
        factor = remaining_k / current_sum
        new_values = pi[remaining_indices] * factor
        over_ones = remaining_indices[new_values >= 1.0 - tol]
        if len(over_ones) == 0:
            pi[remaining_indices] = new_values
            break
        pi[over_ones] = 1.0 - tol
        remaining_k -= len(over_ones)
        remaining_indices = np.array([i for i in remaining_indices if i not in over_ones])
        if remaining_k <= 0:
            pi[remaining_indices] = 0.0
            break
            
    return pi


def WeightToProb(weight):
    return [w / (1 + w) for w in weight]


def PlanCache(
    trace_path,
    batch_size,
    k,
    num_experts,
    num_sparse_layers,
    tensors_per_expert,
    num_elements_per_tensor,
    compression_ratio,
    num_file_chunks,
    compute_pool_size,
    device_memory_ratio,
    decompression_delay,
    SM_IO_delay,
    step_size = 0.05,
    synthetic = False
):
    sorted_prob_dist = ObtainDistFromTrace(batch_size, trace_path)
    target_probs = scale_to_k(sorted_prob_dist, k)
    sorted_prob_weight = CoverageToWeight(target_probs, k)
    cps_probs = WeightToProb(sorted_prob_weight)
    print(f"Bernoulli probs sum: {np.sum(cps_probs):.4f}")
    avgk = int(np.sum(cps_probs))
    best_cache_pool_ratio , feasible = CachePoolPlanningSimu(
        avgk, 
        num_experts,
        num_sparse_layers,
        tensors_per_expert,
        num_elements_per_tensor,
        compression_ratio,
        num_file_chunks,
        compute_pool_size,
        device_memory_ratio,
        decompression_delay,
        SM_IO_delay,
        cps_probs,
        step_size
    )
    print(f"[ZipMoE Cache Pool Planning] Best ratio: {best_cache_pool_ratio:.2f}")
    return best_cache_pool_ratio if feasible else 0.99, feasible
