import json
import os
from typing import Dict, List
import torch.fx as fx
from enum import Enum
from enum import auto

ALL_RED_DICT_DIR = "/data/home/sanketpurandare/benchmark_fork/benchmark/commfuser/demo"
with open(f"{ALL_RED_DICT_DIR}/all_red_model.dict", "r") as all_red_model_file:
    all_red_model: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = json.load(
        all_red_model_file
    )
with open(f"{ALL_RED_DICT_DIR}/data_sizes.list", "r") as data_sizes_file:
    data_sizes = json.load(data_sizes_file)
with open(f"{ALL_RED_DICT_DIR}/nodes.list", "r") as nodes_file:
    nodes = json.load(nodes_file)
with open(f"{ALL_RED_DICT_DIR}/network_types.list", "r") as network_types_file:
    network_types = json.load(network_types_file)
MB = 2**20


def get_all_reduce_burst_time(numel: int) -> float:
    num_nodes = int(os.environ["WORLD_SIZE"])
    net_type = os.environ["NET_TYPE"]
    data_size = numel * 4 / MB
    data_size_q = 5
    for d_size in data_sizes:
        if d_size >= data_size:
            data_size_q = d_size
            break
    comm_latency = all_red_model[net_type][f"{num_nodes}"][f"{data_size_q}"]["comm"]
    straggle_latency = all_red_model[net_type][f"{num_nodes}"][f"{data_size_q}"][
        "straggle"
    ]
    burst_time = comm_latency + straggle_latency
    return burst_time


# We support three types of bucketing strategies
# 1) Fixed: The bucket size will be the same for all the buckets. But we will do
#    a binary search, to find the best bucket size that minimizes the end-to-end
#    iteration latency.
# 2) Variable: The bucket size can be different for each bucket. We form these
#    buckets dynamically
# 3) Constant: This is mimicking of original DDP algorithm, that given a bucket
#    size, we just form the buckets with the specified bucket size.


class BucketingStrategy(Enum):
    FIXED = auto()
    VARIABLE = auto()
    CONSTANT = auto()


# This class represents a single bucketing element
class BucketElement:
    def __init__(self, grad: fx.Node, param: fx.Node, gm_id: int, numel: int) -> None:
        self.grad: fx.Node = grad
        self.param: fx.Node = param
        self.gm_id = gm_id
        self.numel = numel


class Bucket:
    def __init__(
        self,
        bid: int,
        bucket_list: List[BucketElement],
        last_grad: fx.Node,
        first_param: fx.node,
        first_param_access: fx.Node,
        burst_time: float,
    ):
        self.id: int = bid
        self.bucket_list: List[BucketElement] = bucket_list
        self.last_grad: fx.Node = last_grad
        self.first_param: fx.Node = first_param
        self.first_param_access: fx.Node = first_param_access
        self.burst_time: float = burst_time


class SchedulingPolicy(Enum):
    FCFS = auto()
    GREEDY_IPRTT = auto()
    GREEDY_NDPRTT = auto()
    BRANCH_AND_BOUND = auto()


class Process:
    def __init__(
        self, p_id: int, arrival_time: float, deadline: float, burst_time: float
    ) -> None:
        self.p_id: int = p_id
        self.arrival_time: float = arrival_time
        self.burst_time: float = burst_time
        self.deadline: float = deadline
