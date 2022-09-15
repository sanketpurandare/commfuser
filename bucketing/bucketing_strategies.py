import logging
import os
from decimal import MAX_EMAX
from enum import Enum
from enum import auto
from typing import List
from typing import Set
from typing import Union

import torch
from torch import fx

from scheduling.scheduling_policies import SchedulingPolicy


class BucketingStrategy(Enum):
    STATIC = auto()
    DYNAMIC = auto()


class BucketElement:
    def __init__(self, grad: fx.Node, gm_id: int, numel: int) -> None:
        self.grad: fx.Node = grad
        self.gm_id = gm_id
        self.numel = numel


def get_latency_from_simulator(
    structured_bwd_gms: List[Union[fx.GraphModule, Set[fx.GraphModule]]],
    buckets: List[List[BucketElement]],
):
    pass


def static_bucketing(
    buckets: List[List[BucketElement]],
    structured_bwd_gms: List[Union[fx.GraphModule, Set[fx.GraphModule]]],
    l_bucket_size: int,
    r_bucket_size: int,
):

    l_bucket_list = get_static_buckets(buckets, l_bucket_size)
    r_bucket_list = get_static_buckets(buckets, r_bucket_size)

    # Now get the end to end iteration latency for these bucket configurations
    l_latency = get_latency_from_simulator(structured_bwd_gms, l_bucket_list)

    r_latency = get_latency_from_simulator(structured_bwd_gms, r_bucket_list)

    def binary_search_bucket_size(
        buckets: List[List[BucketElement]],
        l_bucket_size: int,
        l_latency: float,
        r_bucket_size: int,
        r_latency: float,
    ):
        nonlocal structured_bwd_gms

        if abs(l_bucket_size - r_bucket_size) < 20:
            return l_bucket_size
        m_bucket_size = (l_bucket_size + r_bucket_size) / 2
        m_bucket_list = get_static_buckets(buckets, m_bucket_size)
        m_latency = get_latency_from_simulator(structured_bwd_gms, m_bucket_list)

        if l_latency < m_latency:
            # search for a bucket size between l and m
            return binary_search_bucket_size(
                buckets, l_bucket_size, l_latency, m_bucket_size, m_latency
            )
        else:
            # search for a bucket size between m and r
            return binary_search_bucket_size(
                buckets, m_bucket_size, m_latency, r_bucket_size, r_latency
            )


def get_static_buckets(buckets: List[List[BucketElement]], bucket_size: int):
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


def dynamic_bucketing(
    buckets: List[List[BucketElement]],
    structured_bwd_gms: List[Union[fx.GraphModule, Set[fx.GraphModule]]],
):
    latency = get_latency_from_simulator(buckets, structured_bwd_gms)
    while True:
        min_pair_latency = 1000000.0
        min_pair_index = -1
        for i in range(len(buckets) - 1):
            pair = [buckets[i] + buckets[i + 1]]
            bucket_list = buckets[:i] + pair + buckets[i + 2 :]
            i_latency = get_latency_from_simulator(bucket_list, structured_bwd_gms)
            if i_latency < min_pair_latency:
                min_pair_latency = i_latency
                min_pair_index = i
        if min_pair_latency > latency:
            break
        else:
            buckets[min_pair_index].extend(buckets[min_pair_index + 1])
            buckets.remove(buckets[min_pair_index + 1])


if __name__ == "__main__":
    bucket_list = []
    for _ in range(50):
        bucket = BucketElement(None, 0, 30)
        bucket_list.append([bucket])

    n_bucket_list = get_static_buckets(bucket_list, 70)
    for b_list in n_bucket_list:

        print(b_list)
