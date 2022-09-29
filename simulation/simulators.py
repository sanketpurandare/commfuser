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
from commfuser.bucketing.bucketing_strategies import Bucket
from commfuser.graph_profiling.graph_profiler import GraphProfiler
from commfuser.graph_profiling.graph_profiler_front_end import BACKWARD
from commfuser.graph_profiling.graph_profiler_front_end import FORWARD
from commfuser.graph_profiling.graph_profiler_utils import GraphType
from commfuser.graph_profiling.graph_profiler_utils import NodeInfo
from scheduling.scheduling_policies import SchedulingPolicy
from torch import fx


class BucketStatus(Enum):
    NOT_QUEUED = auto()
    READY = auto()
    RUNNING = auto()
    FINISHED = auto()


class BucketState:
    def __init__(self, bucket: Bucket):
        self.bucket: Bucket = bucket
        self.status: BucketStatus = BucketStatus.NOT_QUEUED
        self.time_remaining: float = bucket.burst_time


def get_latency_from_simulator(
    ordered_buckets: List[Bucket],
    profilers: Dict[int, Dict[GraphType, GraphProfiler]],
) -> float:

    num_graphs: int = len(profilers)
    bw_graphs_to_run: List[fx.GraphModule] = []
    fw_graphs_to_run: List[fx.GraphModule] = []

    for gid in reversed(range(num_graphs)):
        bw_graphs_to_run.append[profilers[gid][BACKWARD].module]
    for gid in range(num_graphs):
        fw_graphs_to_run.append[profilers[gid][FORWARD].module]

    grad_to_bucket_state: Dict[fx.Node, BucketState] = {}
    param_access_to_bucket_state: Dict[fx.Node, BucketState] = {}

    for bucket in ordered_buckets:
        last_grad: fx.Node = bucket.last_grad
        first_param_access: fx.Node = bucket.first_param_access
        bucket_state: BucketState = BucketState(bucket)
        grad_to_bucket_state[last_grad] = bucket_state
        param_access_to_bucket_state[first_param_access] = bucket_state

    total_latency: float = 0
    network_stream_exe_queue: List[BucketState] = []
    network_stream_ready_buckets: List[BucketState] = []
    next_bucket_idx: int = 0

    for gid, gm in enumerate(bw_graphs_to_run):
        bw_gid: int = num_graphs - 1 - gid
        node_info: Dict[fx.Node, NodeInfo] = profilers[bw_gid][BACKWARD].node_info
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            n_info: NodeInfo = node_info.get(node, None)
            if n_info is not None:
                node_exe_time: float = n_info.run_time
                total_latency += node_exe_time
                bucket_states_to_remove: List[BucketState] = []
                for bucket_state in network_stream_exe_queue:
                    node_exe_time -= bucket_state.time_remaining
                    if node_exe_time < 0:
                        bucket_state.time_remaining -= node_exe_time
                        break
                    else:
                        bucket_states_to_remove.append(bucket_state)

                for bucket_state in bucket_states_to_remove:
                    network_stream_exe_queue.remove(bucket_state)
                    bucket_state.status = BucketStatus.FINISHED
                    bucket_state.time_remaining = 0

                bucket_state: BucketState = grad_to_bucket_state.get(node, None)
                if bucket_state is not None:
                    if bucket_state.bucket is ordered_buckets[next_bucket_idx]:
                        bucket_state.status = BucketStatus.RUNNING
                        network_stream_exe_queue.append(bucket_state)
                        next_bucket_idx += 1
                    else:
                        bucket_state.status = BucketStatus.READY
                        network_stream_ready_buckets.append(bucket_state)

            ready_buckets: List[Bucket] = [
                bucket_state.bucket for bucket_state in network_stream_ready_buckets
            ]
            while len(ready_buckets) > 0:
                if ordered_buckets[next_bucket_idx] in ready_buckets:
                    bs_idx: int = ready_buckets.index(ordered_buckets[next_bucket_idx])
                    bucket_state: BucketState = network_stream_ready_buckets[bs_idx]
                    bucket_state.status = BucketStatus.RUNNING
                    network_stream_exe_queue.append(bucket_state)
                    ready_buckets.remove(ordered_buckets[next_bucket_idx])
                    network_stream_ready_buckets.remove(bucket_state)
                    next_bucket_idx += 1
                else:
                    break

    # Make sure that all of the buckets are in running/finished state

    for bucket_state in grad_to_bucket_state.values():
        assert bucket_state.status in [BucketStatus.RUNNING, BucketStatus.FINISHED]

    for gid, gm in enumerate(fw_graphs_to_run):
        node_info: Dict[fx.Node, NodeInfo] = profilers[bw_gid][FORWARD].node_info
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue

            bucket_state: BucketState = param_access_to_bucket_state.get(node, None)
            if bucket_state is not None:
                if bucket_state.status is not BucketStatus.FINISHED:
                    bucket_states_to_remove: List[BucketState] = []
                    for exe_bucket_state in network_stream_exe_queue:
                        total_latency += exe_bucket_state.time_remaining
                        bucket_states_to_remove.append(exe_bucket_state)
                        if exe_bucket_state is bucket_state:
                            break
                    for bs in bucket_states_to_remove:
                        bs.status = BucketStatus.FINISHED
                        bs.time_remaining = 0
                        network_stream_exe_queue.remove(bs)

            n_info: NodeInfo = node_info.get(node, None)
            if n_info is not None:
                node_exe_time: float = n_info.run_time
                total_latency += node_exe_time
                bucket_states_to_remove: List[BucketState] = []
                for bucket_state in network_stream_exe_queue:
                    node_exe_time -= bucket_state.time_remaining
                    if node_exe_time < 0:
                        bucket_state.time_remaining -= node_exe_time
                        break
                    else:
                        bucket_states_to_remove.append(bucket_state)

                for bucket_state in bucket_states_to_remove:
                    network_stream_exe_queue.remove(bucket_state)
                    bucket_state.status = BucketStatus.FINISHED
                    bucket_state.time_remaining = 0

    return total_latency
