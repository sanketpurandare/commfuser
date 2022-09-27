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
from commfuser.graph_profiling.graph_profiler_utils import GraphProfiler
from commfuser.graph_profiling.graph_profiler_utils import GraphType
from scheduling.scheduling_policies import SchedulingPolicy
from torch import fx

def get_latency_from_simulator(
    ordered_buckets: List[Bucket],
    profilers: Dict[int, Dict[GraphType, GraphProfiler]],
    prev_runtimes: Dict[int, Dict[GraphType, float]],
) -> float:
    pass