from functorch.compile import aot_function, aot_module, draw_graph
from torch import nn
from torch.distributed import ProcessGroup

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.multiprocessing as mp
import torch.utils._pytree as pytree


from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import (
    Any,
    Callable,
    Optional,
)
import logging
import os



class MyModel(nn.Module):
    def __init__(self, n_features, n_layers):
        super().__init__()
        self.seq = nn.Sequential(*[nn.Linear(n_features, n_features) for _ in range(n_layers)])

    def forward(self, x):
        return self.seq(x)

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


# HACK:
# 1. ditto
# 2. The current version is a synchronized call. We need to make it asynchronized,
#    queue the copy-back bottom-half as callbacks on the returned future, and then
#    insert a wait() fx node at the end of backward.
def fused_allreduce(tensors, pg):
    logging.info(f"Fused AllReduce Tensors of shape {[t.shape for t in tensors]}")
    # TODO: we are creating a new buffer on every fused allreduce, which is slow.
    # We should instead creating the buffers upfront and just doing copy here.
    buffer = torch.empty(sum([t.numel() for t in tensors]))
    offset = 0
    for t in tensors:
        numel = t.numel()
        buffer[offset:offset + numel] = t.view(-1)
        offset += numel

    dist.all_reduce(buffer, group=pg)

    offset = 0
    for t in tensors:
        numel = t.numel()
        t = buffer[offset:offset + numel].view(t.shape)
        offset += numel


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
        self.n_grads = sum([p.requires_grad for p in module.parameters()])
        # HACK: ideally, it will be better if fx/AOTAutograd can provide a way
        # to access original param, instead of us keeping the following maps.
        self.primal_to_param = {}
        self.grad_to_primal = {}

        self.compiled_m = None

    def run(self, x: torch.Tensor):
        if self.compiled_m is None:
            self.compiled_m = aot_module(self.module, self.compile_fwd, self.compile_bwd)

        # HACK: AOTAutograd cannot trace the train_step yet, so compile the
        # module for now.
        self.compiled_m(x).sum().backward()

    def compile_fwd(self, gm: fx.GraphModule, inps):
        # HACK: use pytree order of params to map to primals, and save the info
        # for compile_bwd.
        def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
            idx = int(primal_name.split("_")[-1]) - 1
            params = [p for _, p in list(pytree.tree_flatten(model.named_parameters())[0][0])]
            return params[idx] if idx < len(params) else None

        logging.info("Compiling forward")
        gm.graph.print_tabular()
        # get tags on each param
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.target.startswith("primal"):
                p = to_param(self.module, node.name)
                if p is not None:
                    assert node.target not in self.primal_to_param, (
                        f"inserting {node.target} twice"
                    )
                    self.primal_to_param[node.target] = p

        logging.info(
            "\nFinished compiling forward, identified following Distributed Tensors\n" +
            "\n".join([f"{pl} : {pm._dtags}" for pl, pm in self.primal_to_param.items()])
        )
        return gm

    def compile_bwd(self, gm: fx.GraphModule, inps):
        logging.info("Compiling backward")
        logging.info("Original backward graph")
        gm.graph.print_tabular()
        # insert individual allreduce
        pgs = {}
        for node in gm.graph.nodes:
            if node.op == "output":
                # HACK: again, relying on the implicit guarantee that primals
                # and gradient outputs follow the same order.
                for i, grad_node in enumerate(node.args[0][:self.n_grads]):
                    primal = f"primals_{i + 1}"
                    self.grad_to_primal[grad_node.name] = primal
                    for dtag in self.primal_to_param[primal]._dtags:
                        if dtag.dttype == DTensorType.REPLICATED:
                            with gm.graph.inserting_after(grad_node):
                                gm.graph.call_function(allreduce, args=(grad_node, dtag.pg))
                                pgs[grad_node] = dtag.pg
                break

        # fuse allreduce ops based on bucket_mb
        comm_args, comm_nodes, comm_size, pg = [], [], 0, None
        for node in gm.graph.nodes:
            # HACK: allreduce is a custom Python function for now. It will be
            # more readable if we convert it into an ATen operator
            if node.name.startswith("allreduce"):
                comm_args.append(node.args[0])
                comm_size += self.primal_to_param[self.grad_to_primal[node.args[0].name]].numel()
                comm_nodes.append(node)
                assert pg is None or pg == pgs[node.args[0]], (
                    "expecting the same ProcessGroup instance for now"
                )
                pg = pgs[node.args[0]]
                last_node = node

                if comm_size >= self.bucket_mb * 1e6:
                    # accumulated comm size larger than the bucket size, fuse them.
                    with gm.graph.inserting_after(last_node):
                        gm.graph.call_function(fused_allreduce, args=(comm_args, pg))

                    comm_args, comm_size = [], 0

        if len(comm_args) > 0:
            with gm.graph.inserting_after(last_node):
                gm.graph.call_function(fused_allreduce, args=(comm_args, pg))

        for node in comm_nodes:
            gm.graph.erase_node(node)

        gm.graph.lint()
        gm.recompile()
        logging.info("Modified backward graph")
        gm.graph.print_tabular()

        logging.info("finished compiling backward")
        return gm


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
    model = MyModel(n_features, 4)
    # tag all parameters as replicated tensor
    model = DDP(model)
    # we should be able to support the following as well
    # DDP(FSDP(model, pg=intra_node), pg=inter_node)

    # compile train_step, insert comm ops based on tags in model, and fuse them
    engine = Engine(model, train_step)
    for i in range(3):
        logging.info(f"================== ITERATION {i} ====================")
        # dummy input
        x = torch.randn(2, n_features)
        # run the compiled train_step
        engine.run(x)

    # Discussion:
    # Explicitly passing train_step to Engine rather than using the following API
    # because we might prefer to optimize the entire training step or even
    # multiple training steps together, instead of just optimizing fwd/bwd
    #   model = DistributedModel(model)
    #   train_step(model, x)


if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True)