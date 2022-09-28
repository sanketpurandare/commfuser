import logging
import os
import sys
from decimal import MAX_EMAX
from enum import Enum
from enum import auto
from typing import Dict
from typing import List
from typing import Set
from typing import Union

import torch
from commfuser.demo.partial_ddp_front_end import get_all_reduce_burst_time
from commfuser.graph_profiling.graph_profiler import GraphProfiler
from commfuser.graph_profiling.graph_profiler_front_end import BACKWARD, FORWARD
from commfuser.graph_profiling.graph_profiler_utils import NodeInfo
from commfuser.graph_profiling.graph_profiler_utils import GraphType
from commfuser.simulation.simulators import get_latency_from_simulator
from scheduling.scheduling_policies import Process, SchedulingPolicy, branch_and_bound_algorithm, fcfs_scheduling, greedy_scheduling_IPRTT, greedy_scheduling_NDPRTT
from torch import fx


class BucketingStrategy(Enum):
    FIXED = auto()
    VARIABLE = auto()


class BucketElement:
    def __init__(self, grad: fx.Node, param:fx.Node, gm_id: int, numel: int) -> None:
        self.grad: fx.Node = grad
        self.param: fx.Node = param
        self.gm_id = gm_id
        self.numel = numel


class Bucket:
    def __init__(
        self, bid:int, bucket_list: List[BucketElement], last_grad: fx.Node, first_param: fx.node, burst_time:float
    ):
        self.id:int = bid
        self.bucket_list: List[BucketElement] = bucket_list
        self.last_grad: fx.Node = last_grad
        self.first_param: fx.Node = first_param
        self.burst_time: float = burst_time



def get_scheduled_bucket_order(
    bucket_list: List[List[BucketElement]],
    profilers: Dict[int, Dict[GraphType, GraphProfiler]],
    prev_runtimes: Dict[int, Dict[GraphType, float]],
    scheduling_policy: SchedulingPolicy,
) -> List[Bucket]:

    def get_last_grad_generated(bucket:List[BucketElement], graph_profiler:GraphProfiler)->fx.Node:
        bucket_grads:List[fx.Node] = [be.grad for be in bucket]
        node_info:Dict[fx.Node,NodeInfo] = graph_profiler.node_info
        last_grad:fx.Node = max(bucket_grads, key= lambda grad_node: node_info[grad_node].rank)
        return last_grad

    def get_first_param_usage(bucket:List[BucketElement], graph_profiler:GraphProfiler)->fx.Node:
        node_info:Dict[fx.Node, NodeInfo] = graph_profiler.node_info
        bucket_params:List[fx.Node] = [be.param for be in bucket]
        first_param:fx.Node = min(bucket_params, key = lambda param_node: node_info[node_info[param_node].first_forward_access].rank)
        return first_param

    def process_creator(bucket:Bucket, profilers:Dict[GraphType, GraphProfiler], prev_runtimes:Dict[GraphType, float])->Process:
        pid:int = bucket.id
        arrival_time:float = prev_runtimes[BACKWARD] + profilers[BACKWARD].node_info[bucket.last_grad].cumulative_run_time
        first_forward_access:fx.Node = profilers[FORWARD].node_info[bucket.first_param].first_forward_access
        deadline:float = prev_runtimes[FORWARD] + profilers[FORWARD].node_info[first_forward_access].cumulative_run_time - profilers[FORWARD].node_info[first_forward_access].run_time
        burst_time:float = bucket.burst_time 
        return Process(pid, arrival_time, deadline, burst_time)


    unordered_buckets:List[Bucket] = []
    unscheduled_processes:List[Process] = []
    for bid, b_list in enumerate(bucket_list):
        gm_ids:Set[int] = set([be.gm_id for be in b_list])
        assert(len(gm_ids)==1)
        b_gm_id:int = gm_ids.pop()
        last_grad:fx.Node = get_last_grad_generated(b_list, profilers[b_gm_id][BACKWARD])
        first_param:fx.Node = get_first_param_usage(b_list, profilers[b_gm_id][FORWARD])
        total_bucket_numel:int = sum([be.numel for be in b_list])
        burst_time:float = get_all_reduce_burst_time(total_bucket_numel)
        bucket:Bucket = Bucket(bid, b_list, last_grad, first_param, burst_time)
        process:Process = process_creator(bucket, profilers[b_gm_id], prev_runtimes[b_gm_id])
        unscheduled_processes.append(process)
        unordered_buckets.append(bucket)
    scheduled_processes:List[Process] = None
    if(scheduling_policy == SchedulingPolicy.FCFS):
        scheduled_processes = fcfs_scheduling(unscheduled_processes)
    elif(scheduling_policy == SchedulingPolicy.GREEDY_IPRTT):
        scheduled_processes = greedy_scheduling_IPRTT(unscheduled_processes)
    elif(scheduling_policy == SchedulingPolicy.GREEDY_NDPRTT):
        scheduled_processes = greedy_scheduling_NDPRTT(unscheduled_processes)
    elif(scheduling_policy == SchedulingPolicy.BRANCH_AND_BOUND):
        scheduled_processes = branch_and_bound_algorithm(unscheduled_processes)

    ordered_buckets:List[Bucket] = [unordered_buckets[s_process.p_id] for s_process in scheduled_processes]

    return ordered_buckets



def fixed_bucketing(
    bucket_dict: Dict[int, List[List[BucketElement]]],
    profilers: Dict[int, Dict[GraphType, GraphProfiler]],
    prev_runtimes: Dict[int, Dict[GraphType, float]],
    l_bucket_size: int,
    r_bucket_size: int,
    scheduling_policy: SchedulingPolicy,
):
    def get_fixed_buckets(buckets: List[List[BucketElement]], bucket_size: int):
        def get_numel(bucket: List[BucketElement]):
            total_numel = 0
            for b in bucket:
                total_numel += b.numel
            return total_numel

        bucket_list = []
        bsize = 0
        current_head = None
        for b in buckets:
            bsize += get_numel(b)
            if current_head is None:
                current_head = b
            else:
                current_head.extend(b)

            if bsize >= bucket_size:
                bsize = 0
                bucket_list.append(current_head)
                current_head = None

        if current_head is not None:
            bucket_list.append(current_head)
        return bucket_list

    meta_l_bucket_list: List[List[BucketElement]] = []
    meta_r_bucket_list: List[List[BucketElement]] = []
    for gid in reversed(range(len(bucket_dict))):
        bucket_list: List[List[BucketElement]] = bucket_dict[gid]
        l_bucket_list = get_fixed_buckets(bucket_list, l_bucket_size)
        r_bucket_list = get_fixed_buckets(bucket_list, r_bucket_size)
        meta_l_bucket_list.extend(l_bucket_list)
        meta_r_bucket_list.extend(r_bucket_list)

    ordered_l_buckets: List[Bucket] = get_scheduled_bucket_order(
        meta_l_bucket_list, profilers, prev_runtimes, scheduling_policy
    )
    ordered_r_buckets: List[Bucket] = get_scheduled_bucket_order(
        meta_r_bucket_list, profilers, prev_runtimes, scheduling_policy
    )

    # Now get the end to end iteration latency for these bucket configurations
    l_latency = get_latency_from_simulator(ordered_l_buckets, profilers, prev_runtimes)
    r_latency = get_latency_from_simulator(ordered_r_buckets, profilers, prev_runtimes)

    def binary_search_bucket_size(
        bucket_dict: Dict[int, List[List[BucketElement]]],
        profilers: Dict[int, Dict[GraphType, GraphProfiler]],
        prev_runtimes: Dict[int, Dict[GraphType, float]],
        l_bucket_size: int,
        l_latency: float,
        ordered_l_buckets: List[Bucket],
        r_bucket_size: int,
        r_latency: float,
        ordered_r_buckets: List[Bucket],
        scheduling_policy: SchedulingPolicy,
    ):

        if abs(l_bucket_size - r_bucket_size) < l_bucket_size:
            if l_latency < r_latency:
                return ordered_l_buckets
            else:
                return ordered_r_buckets

        m_bucket_size = (l_bucket_size + r_bucket_size) / 2
        meta_m_bucket_list: List[List[BucketElement]] = []
        for gid in reversed(range(len(bucket_dict))):
            bucket_list: List[List[BucketElement]] = bucket_dict[gid]
            m_bucket_list = get_fixed_buckets(bucket_list, m_bucket_size)
            meta_m_bucket_list.extend(m_bucket_list)

        ordered_m_buckets: List[Bucket] = get_scheduled_bucket_order(
            meta_m_bucket_list, profilers, prev_runtimes, scheduling_policy
        )
        m_latency = get_latency_from_simulator(
            ordered_m_buckets, profilers, prev_runtimes
        )

        if l_latency < m_latency:
            # search for a bucket size between l and m
            return binary_search_bucket_size(
                bucket_dict,
                profilers,
                prev_runtimes,
                l_bucket_size,
                l_latency,
                ordered_l_buckets,
                m_bucket_size,
                m_latency,
                ordered_m_buckets,
                scheduling_policy,
            )
        else:
            # search for a bucket size between m and r
            return binary_search_bucket_size(
                bucket_dict,
                profilers,
                prev_runtimes,
                m_bucket_size,
                m_latency,
                ordered_m_buckets,
                r_bucket_size,
                r_latency,
                ordered_r_buckets,
                scheduling_policy,
            )

    return binary_search_bucket_size(
        bucket_dict,
        profilers,
        prev_runtimes,
        l_bucket_size,
        l_latency,
        ordered_l_buckets,
        r_bucket_size,
        r_latency,
        ordered_r_buckets,
        scheduling_policy,
    )


def variable_bucketing(
    bucket_dict: Dict[int, List[List[BucketElement]]],
    profilers: Dict[int, Dict[GraphType, GraphProfiler]],
    prev_runtimes: Dict[int, Dict[GraphType, float]],
    scheduling_policy: SchedulingPolicy,
):
    meta_bucket_list: List[List[BucketElement]] = []
    for gid in reversed(range(len(bucket_dict))):
        gm_bucket_list: List[List[BucketElement]] = bucket_dict[gid]
        meta_bucket_list.extend(gm_bucket_list)
    ordered_buckets: List[Bucket] = get_scheduled_bucket_order(
        meta_bucket_list, profilers, prev_runtimes, scheduling_policy
    )
    latency = get_latency_from_simulator(ordered_buckets, profilers, prev_runtimes)

    def check_valid_pair(bucket_list: List[List[BucketElement]], i, j) -> bool:
        last_elem_bucket_i: BucketElement = bucket_list[i][-1]
        first_elem_bucket_j: BucketElement = bucket_list[j][0]
        return last_elem_bucket_i.gm_id == first_elem_bucket_j.gm_id

    ret_ordered_buckets: List[Bucket] = ordered_buckets
    while True:
        min_pair_latency: float = sys.float_info.max
        min_pair_index: int = -1
        min_ordered_buckets: List[Bucket] = None
        for i in range(len(meta_bucket_list) - 1):
            if check_valid_pair(meta_bucket_list, i, i + 1):
                pair = [meta_bucket_list[i] + meta_bucket_list[i + 1]]
                i_bucket_list = meta_bucket_list[:i] + pair + meta_bucket_list[i + 2 :]
                i_ordered_buckets: List[Bucket] = get_scheduled_bucket_order(
                    i_bucket_list, profilers, prev_runtimes, scheduling_policy
                )
                i_latency = get_latency_from_simulator(
                    i_ordered_buckets, profilers, prev_runtimes
                )
                if i_latency < min_pair_latency:
                    min_pair_latency = i_latency
                    min_pair_index = i
                    min_ordered_buckets = i_ordered_buckets
            else:
                continue
        if min_pair_latency > latency:
            break
        else:
            meta_bucket_list[min_pair_index].extend(
                meta_bucket_list[min_pair_index + 1]
            )
            meta_bucket_list.remove(meta_bucket_list[min_pair_index + 1])
            ret_ordered_buckets = min_ordered_buckets

    return ret_ordered_buckets
