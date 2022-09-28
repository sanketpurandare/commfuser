import logging
import os
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Set
from typing import Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils._pytree as pytree
import torchdynamo
from commfuser.bucketing.bucketing_strategies import BucketElement
from commfuser.bucketing.bucketing_strategies import BucketingStrategy
from commfuser.bucketing.bucketing_strategies import fixed_bucketing
from commfuser.bucketing.bucketing_strategies import variable_bucketing
from commfuser.graph_profiling.graph_profiler_utils import GraphProfiler
from commfuser.graph_profiling.graph_profiler_utils import GraphType
from commfuser.scheduling.scheduling_policies import SchedulingPolicy
from functorch.compile import aot_function
from functorch.compile import aot_module
from functorch.compile import draw_graph
from graph_profiling.graph_profiler_front_end import BACKWARD
from graph_profiling.graph_profiler_front_end import FORWARD
from graph_profiling.graph_profiler_front_end import ProfileEngine
from torch import fx
from torch import nn
from torch import optim
from torch.distributed import ProcessGroup

MIN_BUCKET_SIZE = 25 * (2**18)  # 25MB/4
MAX_BUCKET_SIZE = 2**28  # 1024MB/4 = 1GB/4

def get_all_reduce_burst_time()->float:
    pass

# Type of the distributed tensor
class DTensorType(Enum):
    REPLICATED = auto()
    SHARDED = auto()
    PARTIAL = auto()


# A tag attached to local parameter to indicating how users plan to convert it
# to a distributed tensor. Note that, one local param can have multiple
# DTensorTag, and the order of these tags dictates the communication order.


@dataclass
class DTensorTag:
    dttype: DTensorType = DTensorType.REPLICATED
    pg: ProcessGroup = None


# A thin layer implementation of DDP, that only add tags to model parameters.
class DDP(nn.Module):
    """
    Tag each param as replicated
    """

    def __init__(self, model: nn.Module, forward_loss: Callable, pg=None):
        super().__init__()
        self.model: nn.Module = model
        self.forward_loss: Callable = forward_loss

        for p in model.parameters():
            if not hasattr(p, "_dtags"):
                p._dtags = []

            p._dtags.append(DTensorTag(dttype=DTensorType.REPLICATED, pg=pg))


# HACK: dist.allreduce is not compatible with fx/AOTAutograd yet. It will be
# better if we convert comm ops into ATen operators. That will help address two
# problems:
# 1. We can get rid of this function, and directly do
#    graph.call_function(dist.all_reduce, ...)
# 2. It will also be prettier and more readable with ATen comm ops in fx graph;
#    Currently it shows as "call_function  all_reduce_5  <function all_reduce at
#    0x7ff2524e7b80>  (t_11, None)".
def allreduce(tensor, pg):
    logging.info(f"AllReduce Tensor of shape {tensor.shape}")
    dist.all_reduce(tensor, group=pg)


def grad_as_bucket_view(states, grad, bucket_id, offset):
    states.buckets[bucket_id][offset : (offset + grad.numel())] = grad.view(-1)
    grad.data = states.buckets[bucket_id][offset : (offset + grad.numel())].view(
        grad.shape
    )


def fused_allreduce(states, bucket_id, pg, blocked_by):
    logging.info(f"Fused AllReduce bucket of shape {states.buckets[bucket_id].shape}")
    dist.all_reduce(states.buckets[bucket_id], group=pg)


class Engine:
    r"""
    Compile the provided ``train_step`` function. Then, based on the tags on the
    local ``module``, insert communication ops and fuse them.

    Args:
        module (nn.Module): a local model instance with tags on parameters
                            indicating users' intent for distributed training.
                            It's preferred to create this module on Meta device.
        train_step (Callable): a user-defined function that contains forward,
                               backward, and optimizer.step() for one iteration.
                               ``Engine`` will joinly optimize the entire
                               ``train_step``.
    """

    def __init__(
        self,
        model: nn.Module,
        forward_loss: Callable,
        optimizer: optim.Optimizer,
        example_inputs: Any,
        profile_mode: str,
    ):
        # HACK: Meta device tracing is not ready. Have to create the module on
        # CPU for now.
        self.model = model
        self.forward_loss: Callable = forward_loss
        # HACK: train_step is ignored at this time, as AOTAutograd cannot trace
        # through the full fwd + bwd + opt.step yet. Based on the discussion
        # with compiler this, this is addressable.

        # HACK: ideally, it will be better if fx/AOTAutograd can provide a way
        # to access original param, instead of us keeping the following maps.
        self.primal_name_to_node: Dict[int, Dict[str, fx.Node]] = {}
        self.primal_to_param: Dict[int, Dict[fx.Node, nn.Parameter]] = {}
        self.grad_to_primal: Dict[int, Dict[fx.Node, fx.Node]] = {}
        self.optimizer: optim.Optimizer = optimizer
        self.example_inputs: Any = example_inputs
        self.profile_mode: str = profile_mode
        self.profile_engine: ProfileEngine = ProfileEngine(
            self.model,
            self.forward_loss,
            self.optimizer,
            self.example_inputs,
            self.profile_mode,
        )
        self.profilers: Dict[int, Dict[GraphType, GraphProfiler]] = None
        self.optimize_ctx = None

        # HACK: Today, to pass non-placeholder states to fx.Nodes, we have to
        # 1) add a submodule 2) use graph.get_attr() to retrieve this submodule.
        # N.B.: States are required to conduct cross-subgraph fusion.
        class StatesModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.buckets = []
                self.bucket_blocked_by = []

        self.states = StatesModule()

    def _profile(self):
        self.profile_engine.run(warm_up_iters=2, profile_iters=3)
        self.profilers = self.profile_engine.profilers
        self._process_node_info()

    def _process_node_info(self):
        # Process the run_times of individual sub-graphs.
        # Process the backward graphs in reverse order and
        # then process the forward graphs in the given order
        self.prev_runtimes: Dict[int, Dict[GraphType, float]] = {}
        num_graphs = len(self.profilers.keys())
        cumulative_run_time: float = 0
        for gid in reversed(range(num_graphs)):
            bwd_profiler: GraphProfiler = self.profilers[gid][BACKWARD]
            self.prev_runtimes[gid] = {}
            self.prev_runtimes[gid][BACKWARD] = cumulative_run_time
            cumulative_run_time += bwd_profiler.total_runtime

        for gid in range(num_graphs):
            fwd_profiler: GraphProfiler = self.profilers[gid][FORWARD]
            self.prev_runtimes[gid][FORWARD] = cumulative_run_time
            cumulative_run_time += fwd_profiler.total_runtime

    def run(self, x: torch.Tensor):
        if self.optimize_ctx is None:
            # _compile() does the following
            # 1. Use TorchDynamo to get individual subgraphs.
            # 2. For each sub-graph, use AOTAutograd to insert individual
            #    AllReduce
            # 3. After getting all subgraphs, organize them structured format
            #    [g1, {g2, g3}], where list means ordered execution, while set
            #    means parallel branches.
            # 4. Process all these subgraphs together to fuse AllReduce across
            #    subgraphs.
            self._profile()
            self.optimize_ctx = self._compile()

        # Dynamo's context caches compiled graph. Use it for real execution.
        with self.optimize_ctx:
            out = self.module(x)

        out.sum().backward()

    def _allreduce_bucketing_scheduling(
        self,
        structured_bwd_gms: List[Union[fx.GraphModule, Set[fx.GraphModule]]],
        bucketing_strategy: BucketingStrategy,
        scheduling_policy: SchedulingPolicy,
    ):
        # 1. Create a bucket with an individual gradient being a bucket
        # 2. Create a list of bucket elements for each backward graph module
        # 3. Create a dict for graph_id and bucket list
        # 4. Pass the raw individual buckets, profiling information to the chosen
        #   bucketing strategy

        bucket_dict: Dict[int, List[List[BucketElement]]] = {}
        reversed_bwd_gms: List[List[fx.GraphModule]] = [
            [x] if isinstance(x, fx.GraphModule) else list[x]
            for x in reversed(structured_bwd_gms)
        ]
        for phase in reversed_bwd_gms:
            for gm in phase:
                gm_bucket_list: List[List[BucketElement]] = []
                for node in gm.graph.nodes:
                    if node.name.startswith("allreduce"):
                        grad_node:fx.Node = node.args[0]
                        primal_node:fx.Node = self.grad_to_primal[gm._id][grad_node]
                        grad_numel:int = self.primal_to_param[gm._id][primal_node].numel()
                        bucket_element: BucketElement = BucketElement(
                            grad_node, primal_node, gm._id, grad_numel
                        )
                        gm_bucket_list.append([bucket_element])
                bucket_dict[gm._id] = gm_bucket_list

        if bucketing_strategy == BucketingStrategy.FIXED:
            ordered_buckets: List[List[BucketElement]] = fixed_bucketing(
                bucket_dict,
                self.profilers,
                self.prev_runtimes,
                MIN_BUCKET_SIZE,
                MAX_BUCKET_SIZE,
                scheduling_policy,
            )
        elif bucketing_strategy == BucketingStrategy.VARIABLE:
            ordered_buckets: List[List[BucketElement]] = variable_bucketing(
                bucket_dict, self.profilers, self.prev_runtimes, scheduling_policy
            )

        # perfrom graph rewrite here

    def _aot_compile_fwd(self, gid: int, dynamo_fwd_gm: fx.GraphModule):
        def compile_fwd(gm: fx.GraphModule, inps) -> fx.GraphModule:
            nonlocal gid, dynamo_fwd_gm

            def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
                idx = int(primal_name.split("_")[-1]) - 1
                # HACK: Dynamo primal order is the reverse of AOTAutograd???
                params = [
                    p
                    for _, p in reversed(
                        list(pytree.tree_flatten(model.named_parameters())[0][0])
                    )
                ]
                return params[idx] if idx < len(params) else None

            # get tags on each param
            for node in gm.graph.nodes:
                if node.op == "placeholder" and node.target.startswith("primal"):
                    p = to_param(dynamo_fwd_gm, node.name)
                    if p is not None:
                        assert (
                            node not in self.primal_to_param
                        ), f"inserting {node.target} twice"
                        # HACK: use sub-graph gid to distinguish primals with
                        # the same name
                        self.primal_to_param[gid][node] = p
                        self.primal_name_to_node[gid][node.target] = node

            logging.info(
                f"\nCompiled SubGraph-{gid} forward, identified following Distributed Tensors\n"
                + "\n".join(
                    [
                        f"{pl} : {pm._dtags}"
                        for pl, pm in self.primal_to_param[gid].items()
                    ]
                )
            )
            dynamo_fwd_gm._aot_fwd_graph = gm
            return gm

        return compile_fwd

    def _aot_compile_bwd(self, gid: int, dynamo_fwd_gm: fx.GraphModule):
        def compile_bwd(gm: fx.GraphModule, inps) -> fx.GraphModule:
            # HACK: AOTAutograd's GraphModule does not provide a .parameters()
            # member method. Therefore, have to use dynamo's GraphModule to
            # access original parameters.
            nonlocal gid, dynamo_fwd_gm

            # HACK: today, there is no good way to map AOTAutograd primals back
            # to parameters in the original model. The current implementation
            # relies on the implicit AOTAutograd behavior that primals match the
            # order of params in pytree(model.named_parameters()), and grad
            # output in the backward graph matches the same order. So we count
            # the number of params and use that to access primals and grads in
            # the fwd/bwd graphs.
            n_grads = sum([p.requires_grad for p in dynamo_fwd_gm.parameters()])

            logging.info("Compiling backward")
            logging.info("Original backward graph")
            gm.graph.print_tabular()
            # insert individual allreduce. This can be done within each subgraph
            gm.add_submodule("states", self.states)
            for node in gm.graph.nodes:
                if node.op == "output":
                    # HACK: again, relying on the implicit guarantee that
                    # primals and gradient outputs follow the same order.
                    for i, grad_node in enumerate(node.args[0][:n_grads]):
                        primal:str = f"primals_{i+1}"
                        primal_node:fx.Node = self.primal_name_to_node[gid][primal]
                        self.grad_to_primal[gid][grad_node] = primal_node
                        for dtag in self.primal_to_param[gid][primal_node]._dtags:
                            if dtag.dttype == DTensorType.REPLICATED:
                                with gm.graph.inserting_after(grad_node):
                                    gm.graph.call_function(
                                        allreduce, args=(grad_node, dtag.pg)
                                    )
                    break

            gm.graph.lint()
            gm.recompile()
            logging.info("Modified backward graph")
            gm.graph.print_tabular()

            # HACK: Remember individual sub-graphs. Later, will use these to do
            # global fusion across sub-graphs
            gm._id = gid
            dynamo_fwd_gm._aot_bwd_graph = gm
            return gm

        return compile_bwd

    def _compile(self):
        # HACK: check if the graph that produces the activations are the same.
        # We use this hack to check whether two subgraphs are sibling branches.
        # This is not a generic solution. It might be a lot easier to do this
        # within TorchDyanmo.
        def same_activation(x, y):
            if x.shape != y.shape or x.dtype != y.dtype or x.stride() != y.stride():
                return False

            if x.grad_fn is None and y.grad_fn is None:
                return True

            def same_autograd_graph(fn1, fn2):
                if fn1 is None or fn2 is None:
                    return fn1 is None and fn2 is None

                next_fns1, next_fns2 = fn1.next_functions, fn2.next_functions
                if fn1.name() != fn2.name() or len(next_fns1) != len(next_fns2):
                    return False

                for next_fn1, next_fn2 in zip(next_fns1, next_fns2):
                    if not same_autograd_graph(next_fn1[0], next_fn2[0]):
                        return False

                return True

            return same_autograd_graph(x.grad_fn, y.grad_fn)

        # HACK: compiler for Dyanmo. Record all graphs compiled and save them
        # for global fusion.
        graphs, graph_to_inputs, gid = [], {}, 0

        def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            nonlocal graphs, graph_to_inputs, gid

            logging.info(f"Compile Sub-Graph{gid}")
            gm.graph.print_tabular()
            gm._siblings, gm._id, gm._inputs = [gm], gid, example_inputs

            self.primal_to_param[gid] = {}
            self.grad_to_primal[gid] = {}
            self.primal_name_to_node[gid] = {}

            for prior_gm in graphs:
                prior_inputs = graph_to_inputs[prior_gm]
                if all(
                    [
                        same_activation(x, y)
                        for x, y in zip(example_inputs, prior_inputs)
                    ]
                ):
                    prior_gm._siblings.append(gm)
                    gm._siblings = prior_gm._siblings
                    logging.info(
                        f"Found siblings Sub-Graph-{gm._id} and Sub-Graph-{prior_gm._id}"
                    )

            if len(gm._siblings) <= 1:
                graphs.append(gm)
                graph_to_inputs[gm] = example_inputs
            else:
                raise AssertionError("Sibling Graphs are not expected.")

            # Calling AOTAutograd to insert individual allreduce
            compiled_m = aot_module(
                gm, self._aot_compile_fwd(gid, gm), self._aot_compile_bwd(gid, gm)
            )
            gid += 1
            return compiled_m

        optimize_ctx = torchdynamo.optimize(compiler)

        # HACK: use dummy inputs to extract all subgraphs.
        dummy_inputs = self.module.dummy_inputs()
        with optimize_ctx:
            for x in dummy_inputs:
                self.module(x)

        # HACK: retrieve backward graphs and organize them in a structured way
        structured_graphs = []
        for gm in graphs:
            if len(gm._siblings) > 1:
                structured_graphs.append(set([g._aot_bwd_graph for g in gm._siblings]))
            else:
                structured_graphs.append(gm._aot_bwd_graph)

        # HACK: fuse allreduces across sub-graphs
        self._allreduce_bucketing_scheduling(structured_graphs)
        self._fuse_allreduce(self.bucket_mb, structured_graphs)

        logging.info(f"Structured Sub-Graphs: {structured_graphs}")
        return optimize_ctx


################################################################################
#                          Below is User-Facing API                            #
# Note that these are the user-facing APIs for ML practitioners. There will be #
# another user-facing API at the DistributedTensor level for ML system         #
# develoeprs.                                                                  #
# N.B.: APIs below is NOT the final proposal
# #
################################################################################


# this function is what will be compiled and optimized.
def train_step(model: nn.Module, x: torch.Tensor):
    # AOTAutograd cannot trace train_step yet today.
    model(x).sum().backward()


def run_worker(rank, world_size):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    n_features = 1000
    # create local model on CPU
    model = MyModel(n_features)
    # tag all parameters as replicated tensor
    model = DDP(model)
    # we should be able to support the following as well DDP(FSDP(model,
    # pg=intra_node), pg=inter_node)

    # compile train_step, insert comm ops based on tags in model, and fuse them
    engine = Engine(model, train_step, bucket_mb=1)
    for batch in [
        torch.zeros(2, n_features),
        torch.ones(2, n_features),
        -torch.ones(2, n_features),
    ]:
        engine.run(batch)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
