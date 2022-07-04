from torch import fx
from torch import nn
from torch.distributed import ProcessGroup
from functorch.compile import aot_function, aot_module, draw_graph

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils._pytree as pytree
import torchdynamo

from dataclasses import dataclass
from enum import Enum, auto
from typing import  (
    Callable,
    List,
    Set,
    Union,
)
import logging
import os


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.l1(x)
        if x.sum() > 0:
            return self.l2(y)
        else:
            return self.l3(y)

    def dummy_inputs(self) -> List[torch.Tensor]:
        return [torch.ones(2, 10), -torch.ones(2, 10)]


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
    def __init__(self, module, pg=None):
        super().__init__()
        self.module = module

        for p in module.parameters():
            if not hasattr(p, "_dtags"):
                p._dtags = []

            p._dtags.append(DTensorTag(dttype=DTensorType.REPLICATED, pg=pg))

        if hasattr(module, "dummy_inputs"):
            self.dummy_inputs = module.dummy_inputs

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# HACK: dist.allreduce is not compatible with fx/AOTAutograd yet. It will be
# better if we convert comm ops into ATen operators. That will help address two
# problems:
# 1. We can get rid of this function, and directly do graph.call_function(dist.all_reduce, ...)
# 2. It will also be prettier and more readable with ATen comm ops in fx graph;
#    Currently it shows as "call_function  all_reduce_5  <function all_reduce at 0x7ff2524e7b80>  (t_11, None)".
def allreduce(tensor, pg):
    logging.info(f"AllReduce Tensor of shape {tensor.shape}")
    dist.all_reduce(tensor, group=pg)


def grad_as_bucket_view(states, grad, bucket_id, offset):
    states.buckets[bucket_id][offset:(offset + grad.numel())] = grad.view(-1)
    grad.data = states.buckets[bucket_id][offset:(offset + grad.numel())].view(grad.shape)


def fused_allreduce(states, bucket_id, pg):
    logging.info(f"Fused AllReduce bucket of shape {states.buckets[bucket_id].shape}")
    dist.all_reduce(states.buckets[bucket_id], group=pg)


class Engine:
    r"""
    Compile the provided ``train_step`` function. Then, based on the tags on
    the local ``module``, insert communication ops and fuse them.

    Args:
        module (nn.Module): a local model instance with tags on parameters
                            indicating users' intent for distributed training.
                            It's preferred to create this module on Meta device.
        train_step (Callable): a user-defined function that contains forward,
                               backward, and optimizer.step() for one iteration.
                               ``Engine`` will joinly optimize the entire
                               ``train_step``.
    """
    def __init__(self, module: nn.Module, train_step: Callable, bucket_mb: int=25):
        # HACK: Meta device tracing is not ready. Have to create the module on
        # CPU for now.
        self.module = module
        # HACK: train_step is ignored at this time, as AOTAutograd cannot trace
        # through the full fwd + bwd + opt.step yet. Based on the discussion with
        # compiler this, this is addressable.
        self.train_step = train_step
        self.bucket_mb = bucket_mb
        # HACK: today, there is no good way to map AOTAutograd primals back to
        # parameters in the original model. The current implementation relies on
        # the implicit AOTAutograd behavior that primals match the order of
        # params in pytree(model.named_parameters()), and grad output in the
        # backward graph matches the same order. So we count the number of params
        # and use that to access primals and grads in the fwd/bwd graphs.
        #self.n_grads = {}
        # HACK: ideally, it will be better if fx/AOTAutograd can provide a way
        # to access original param, instead of us keeping the following maps.
        self.primal_to_param = {}
        self.grad_to_primal = {}

        self.compiled_m = None
        self.optimize_ctx = None

        # HACK:
        class StatesModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.buckets = []
                self.grads = []

        self.states = StatesModule()

    def run(self, x: torch.Tensor):
        if self.optimize_ctx is None:
            self.optimize_ctx = self._compile_and_extract_partial_graphs()

        print("==== running forward!")
        with self.optimize_ctx:
            out = self.module(x)

        out.sum().backward()


    def _fuse_allreduce(
        self,
        bucket_mb: int,
        structured_bwd_gms: List[Union[fx.GraphModule, Set[fx.GraphModule]]]
    ):
        print("111111111")
        ordered_bwd_gms = [[x] if isinstance(x, fx.GraphModule) else list(x) for x in structured_bwd_gms]

        # normalized allreduce, similar to DDP
        bucket_id, bucket_size, pg = 0, 0, None
        for phase in ordered_bwd_gms:
            print("22222222")
            starting_bucket_id = bucket_id
            for gm in phase:
                # fuse allreduce ops based on bucket_mb
                for node in gm.graph.nodes:
                    # HACK: allreduce is a custom Python function for now. It will be
                    # more readable if we convert it into an ATen operator
                    if node.name.startswith("allreduce"):
                        # 1. build bucket
                        primal = self.grad_to_primal[gm._id][node.args[0].name]
                        offset = bucket_size
                        bucket_size += self.primal_to_param[gm._id][primal].numel()

                        # 2. insert copy_to_bucket node
                        # NB: bucket is not ready, but we can insert node
                        with gm.graph.inserting_after(node):
                            states_node = gm.graph.get_attr("states")

                        with gm.graph.inserting_after(states_node):
                            gm.graph.call_function(
                                grad_as_bucket_view,
                                args=(states_node, node.args[0], bucket_id, offset)
                            )



                        # 3. remember the blocking allreduce for the bucket
                        if bucket_size >= self.bucket_mb * 1e6:
                            self.states.buckets.append(torch.zeros(bucket_size))
                            self.states.bucket_blocked_by.append((gm, node))
                            bucket_size = 0
                            bucket_id += 1

            # 4. insert same sequence of buckets for all branches
            for gm in phase:
                last_node = next(iter(gm.graph.nodes))
                for i, bucket in enumerate(self.states.buckets[starting_bucket_id:]):
                    curr_bucket_id = starting_bucket_id + i
                    blocked_by_gm, block_by_node = self.states.bucket_blocked_by[curr_bucket_id]
                    if gm != blocked_by_gm:
                        with gm.graph.inserting_after(last_node):
                            states_node = gm.graph.get_attr("states")

                        with gm.graph.inserting_after(states_node):
                            last_node = gm.graph.call_function(fused_allreduce, args=(states_node, curr_bucket_id, None))
                    else:
                        with gm.graph.inserting_after(blocked_by_node):
                            states_node = gm.graph.get_attr("states")

                        with gm.graph.inserting_after(states_node):
                            last_node = gm.graph.call_function(fused_allreduce, args=(states_node, curr_bucket_id, None))

        # 5. process remaining grads
        if bucket_size > 0:
            self.states.buckets.append(torch.zeros(bucket_size))
            for gm in ordered_bwd_gms[-1]:
                last_node = next(iter(reversed(gm.graph.nodes)))

                with gm.graph.inserting_before(last_node):
                    states_node = gm.graph.get_attr("states")

                with gm.graph.inserting_after(states_node):
                    gm.graph.call_function(fused_allreduce, args=(states_node, bucket_id, None))

        # t. erase individual allreduce node
        def erase_allreduce_node(gm):
            nodes_to_erase = []
            for node in gm.graph.nodes:
                if node.name.startswith("allreduce"):
                    nodes_to_erase.append(node)

            for node in nodes_to_erase:
                gm.graph.erase_node(node)

        for phase in ordered_bwd_gms:
            for gm in phase:
                print("3333333")
                erase_allreduce_node(gm)
                gm.graph.lint()
                gm.recompile()
                gm.graph.print_tabular()



    def _aot_compile_fwd(self, gid: int, dynamo_fwd_gm: fx.GraphModule):
        def compile_fwd(gm: fx.GraphModule, inps) -> fx.GraphModule:
            nonlocal gid, dynamo_fwd_gm

            def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
                idx = int(primal_name.split("_")[-1]) - 1
                params = [p for _, p in list(pytree.tree_flatten(model.named_parameters())[0][0])]
                #print(f"????? {idx}, {len(params)}")
                return params[idx] if idx < len(params) else None

            # get tags on each param
            for node in gm.graph.nodes:
                if node.op == "placeholder" and node.target.startswith("primal"):
                    p = to_param(dynamo_fwd_gm, node.name)
                    if p is not None:
                        assert node.target not in self.primal_to_param, (
                            f"inserting {node.target} twice"
                        )
                        # HACK: use sub-graph gid to distinguish primals with the same name
                        self.primal_to_param[gid][node.target] = p
                        #print(f"++++ inserted {gid}, {node.target}")

            logging.info(
                f"\nCompiled SubGraph-{gid} forward, identified following Distributed Tensors\n" +
                "\n".join([f"{pl} : {pm._dtags}" for pl, pm in self.primal_to_param[gid].items()])
            )

            gm._id = gid
            dynamo_fwd_gm._aot_fwd_graph = gm
            return gm
        return compile_fwd

    def _aot_compile_bwd(self, gid: int, dynamo_fwd_gm: fx.GraphModule):
        def compile_bwd(gm: fx.GraphModule, inps) -> fx.GraphModule:
            nonlocal gid, dynamo_fwd_gm

            n_grads = sum([p.requires_grad for p in dynamo_fwd_gm.parameters()])

            logging.info("Compiling backward")
            logging.info("Original backward graph")
            gm.graph.print_tabular()
            # insert individual allreduce
            pgs = {}

            gm.add_submodule("states", self.states)
            for node in gm.graph.nodes:
                if node.op == "output":
                    # HACK: again, relying on the implicit guarantee that primals
                    # and gradient outputs follow the same order.
                    for i, grad_node in enumerate(node.args[0][:n_grads]):
                        primal = f"primals_{i+1}"
                        self.grad_to_primal[gid][grad_node.name] = primal
                        for dtag in self.primal_to_param[gid][primal]._dtags:
                            if dtag.dttype == DTensorType.REPLICATED:
                                with gm.graph.inserting_after(grad_node):
                                    gm.graph.call_function(allreduce, args=(grad_node, dtag.pg))
                                    pgs[grad_node] = dtag.pg
                    break

            gm.graph.lint()
            gm.recompile()
            logging.info("Modified backward graph")
            gm.graph.print_tabular()

            logging.info("finished compiling backward")

            gm._id = gid
            dynamo_fwd_gm._aot_bwd_graph = gm
            return gm
        return compile_bwd

    def _compile_and_extract_partial_graphs(self):
        # HACK: get these graphs from compiler
        # HACK: this is not a generic solution to get structured graphs, for testing
        # purpose only

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

        graphs, graph_to_inputs, gid = [], {}, 0
        def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            nonlocal graphs, graph_to_inputs, gid

            logging.info(f"Compile Sub-Graph{gid}")
            gm.graph.print_tabular()
            gm._siblings, gm._id, gm._inputs = [gm], gid, example_inputs

            self.primal_to_param[gid] = {}
            self.grad_to_primal[gid] = {}

            for prior_gm in graphs:
                prior_inputs = graph_to_inputs[prior_gm]
                if all([same_activation(x, y) for x, y in zip(example_inputs, prior_inputs)]):
                    prior_gm._siblings.append(gm)
                    gm._siblings = prior_gm._siblings
                    logging.info(f"Found siblings Sub-Graph-{gm._id} and Sub-Graph-{prior_gm._id}")

            if len(gm._siblings) <= 1:
                graphs.append(gm)
                graph_to_inputs[gm] = example_inputs

            #print(f"=== dynamo submodule parameters {list(gm.parameters())}")
            compiled_m = aot_module(gm, self._aot_compile_fwd(gid, gm), self._aot_compile_bwd(gid, gm))
            gid += 1
            return compiled_m
            #return gm.forward

        optimize_ctx = torchdynamo.optimize(compiler)

        dummy_inputs = self.module.dummy_inputs()
        with optimize_ctx:
            for x in dummy_inputs:
                self.module(x)

        structured_graphs = []
        for gm in graphs:
            if len(gm._siblings) > 1:
                structured_graphs.append(set([g._aot_bwd_graph for g in gm._siblings]))
            else:
                structured_graphs.append(gm._aot_bwd_graph)

        self._fuse_allreduce(self.bucket_mb, structured_graphs)

        logging.info(f"Structured Sub-Graphs: {structured_graphs}")
        return optimize_ctx


# 1. how do we deal with local recompilation?
# 2. can we fuse across partial graphs?
#
# advantages:
# 1. we deterministically know which are model params as we come from DDP/FSDP
# 2. isolate comm executor?



################################################################################
#                          Below is User-Facing API                            #
# Note that these are the user-facing APIs for ML practitioners. There will be #
# another user-facing API at the DistributedTensor level for ML system         #
# develoeprs.                                                                  #
################################################################################


# this function is what will be compiled and optimized.
def train_step(model: nn.Module, x: torch.Tensor):
    # AOTAutograd cannot trace train_step yet today.
    model(x).sum().backward()


def run_worker(rank, world_size):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    n_features = 10000
    # create local model on CPU
    model = MyModel()
    # tag all parameters as replicated tensor
    model = DDP(model)
    # we should be able to support the following as well
    # DDP(FSDP(model, pg=intra_node), pg=inter_node)

    # compile train_step, insert comm ops based on tags in model, and fuse them
    engine = Engine(model, train_step)
    engine.run(torch.zeros(2, 10))
    engine.run(torch.ones(2, 10))
    engine.run(-torch.ones(2, 10))

    #get_partial_graphs(model)



if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 1
    """
    mp.spawn(run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    """
    run_worker(0, 1)