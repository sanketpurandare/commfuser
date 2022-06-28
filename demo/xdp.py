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
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
)

import logging
import math
import os

# HACK:
engine = None


class MyModel(nn.Module):
    def __init__(self, n_features, n_layers):
        super().__init__()
        self.seq = nn.Sequential(*[nn.Linear(n_features, n_features) for _ in range(n_layers)])

    def forward(self, x):
        print("=== before fwd")
        out = self.seq(x)
        print("=== after fwd")
        return out

# Type of the distributed tensor
class DTensorType(Enum):
    REPLICATED = auto()
    SHARDED = auto()
    PARTIAL = auto()
    ONDEMAND = auto()


# A tag attached to local parameter to indicating how users plan to convert it
# to a distributed tensor. Note that, one local param can have multiple
# DTensorTag, and the order of these tags dictates the communication order.
@dataclass
class DTensorTag:
    dttype: DTensorType = DTensorType.REPLICATED
    pg: ProcessGroup = None


def _tag_module(module: nn.Module, tag: DTensorTag):
    for p in module.parameters():
        if not hasattr(p, "_dtags"):
            p._dtags = []

        p._dtags.append(tag)


# A thin layer implementation of DDP, that only add tags to model parameters.
class DDP(nn.Module):
    """
    Tag each param as replicated
    """
    def __init__(self, module: nn.Module, pg: ProcessGroup=None):
        super().__init__()
        self.module = module

        _tag_module(module, DTensorTag(dttype=DTensorType.REPLICATED, pg=pg))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FSDP(nn.Module):
    """
    Tag each param as ondemand
    """
    def __init__(self, module: nn.Module, pg: ProcessGroup=None):
        super().__init__()
        self.module = module

        _tag_module(module, DTensorTag(dttype=DTensorType.ONDEMAND, pg=pg))

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


def ondemand_allgather(param, pg):
    # HACK
    global engine

    print(f"==== before calling on-demand allgather {param.data.shape}, {param._local_shard.shape}")
    #param = engine.get_param(primal)
    local_shard = param._local_shard
    orig_size = param._orig_size
    with torch.no_grad():
        world_size = dist.get_world_size(group=pg)
        buffer = torch.empty([world_size] + list(local_shard.shape))
        tensors = [buffer[i] for i in range(world_size)]
        dist.all_gather(tensors, local_shard, group=pg)
        size = list(orig_size)
        numel = reduce(lambda x, y: x * y, size, 1)
        param.data = buffer[:numel].view(size)

    print(f"==== after calling on-demand allgather {param.data.shape}")
    return param

def ondemand_discard(param, _):
    # HACK
    global engine

    print(f"==== before calling on-demand discard {param.data.shape}")
    #param = engine.get_param(primal)
    with torch.no_grad():
        param.data = param._local_shard
        #pass

    print(f"==== after calling on-demand discard {param.data.shape}")


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
        self.view_to_parent = {}
        self.view_name_to_primal = {}
        self.primal_to_param = {}
        self.grad_to_primal = {}
        self.param_views = {}
        self.pytree_params = [p for _, p in list(pytree.tree_flatten(module.named_parameters())[0][0])]
        self.pytree_params.reverse()

        self.compiled_m = None
        self.fwd_gm = None

        for p in self.module.parameters():
            if hasattr(p, "_dtags"):
                for tag in p._dtags:
                    if tag.dttype == DTensorType.ONDEMAND:
                        self._shard_param_storage(p, tag.pg)
                        break

    def run(self, x: torch.Tensor):
        if self.compiled_m is None:
            self.compiled_m = aot_module(self.module, self._compile_fwd, self._compile_bwd)


        if self.fwd_gm is None:
            # HACK: AOTAutograd cannot trace the train_step yet, so compile the
            # module for now.
            # self.compiled_m(x).sum().backward()
            self.compiled_m(x).sum()
        else:
            # HACK: need to disable guards
            print("----calling fwd_gm")
            self.fwd_gm.graph.print_tabular()
            outs = self.fwd_gm(*self.pytree_params, x)
            out, activations = outs[0], outs[1:]
            out_grad = torch.ones_like(out)
            # needs to change backward input as after compilation, it takes original
            # param instead of its views
            #self.bwd_gm(*activations, out_grad)
            #print(f"forward outout is {outs}")

    def get_param(self, primal):
        return self.primal_to_param[primal.target]

    def _shard_param_storage(self, param, pg):
        with torch.no_grad():
            world_size = dist.get_world_size(group=pg)
            rank = dist.get_rank(group=pg)

            padded_size = int(math.ceil(param.numel() / world_size))
            buffer = torch.empty(padded_size)
            offset = rank * padded_size
            to = min(offset + padded_size, param.numel())
            buffer[:(to - offset)] = param.view(-1)[offset : to]
            param._local_shard = buffer
            param._orig_size = param.size()
            #param.data = buffer

    def _find_primal_views(self, gm: fx.GraphModule, primal: fx.Node):
        view_to_parent = {primal: primal}
        for node in gm.graph.nodes:
            if all([
                node.op == "call_function" and
                str(node.target) == "aten.t" and
                len(node.args) == 1 and
                node.args[0] in view_to_parent
            ]):
                view_to_parent[node] = node.args[0]

        return view_to_parent

    def _find_param_usages(self, gm: fx.GraphModule, views: Set[fx.Node]):
        usages = []
        for node in gm.graph.nodes:
            for view in views:
                if view in node.args:
                    usages.append(node)

        return usages

    def _recover_param_primals(self, gm: fx.GraphModule):

        view_name_to_node = {v.name: v for v, p in self.view_to_parent.items()}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.name in view_name_to_node:
                view_node = view_name_to_node[node.name]
                node_to_insert = []
                while view_node != self.view_to_parent[view_node]:
                    node_to_insert.append(view_node)
                    view_node = self.view_to_parent[view_node]

                node_to_insert.append(view_node)
                node_to_insert.reverse()
                print("++++ inserting ", node_to_insert)
                new_nodes = {}

                def arg_transform(arg):
                    if arg.name in new_nodes:
                        return new_nodes[arg.name]
                    else:
                        raise RuntimeError(f"Unrecognized arg {arg}")

                with gm.graph.inserting_before(node):
                    for to_insert in node_to_insert:
                        for arg in to_insert.args:
                            new_node = gm.graph.node_copy(arg, arg_transform=arg_transform)
                            new_nodes[arg.name] = new_node
                        new_node = gm.graph.node_copy(to_insert, arg_transform=arg_transform)
                        new_nodes[to_insert.name] = new_node

                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

        gm.graph.print_tabular()
        gm.graph.lint()
        #gm.recompile()

    def _handle_ondemand_fwd(self, gm: fx.GraphModule, primal: fx.Node, pg: ProcessGroup):
        views = self._find_primal_views(gm, primal)
        self.view_to_parent.update(views)
        usages = self._find_param_usages(gm, set(views.keys()))

        # insert allgather before first usage
        with gm.graph.inserting_before(usages[0]):
            # HACK: call_function target cannot be member methods?
            new_node = gm.graph.call_function(
                ondemand_allgather,
                args=(primal, pg)
            )
            usages[0].replace_input_with(primal, new_node)

        # insert reshard after last usage
        with gm.graph.inserting_after(usages[-1]):
            gm.graph.call_function(
                ondemand_discard,
                args=(primal, usages[-1]))

    def _handle_ondemand_bwd(self, gm: fx.GraphModule, primal: fx.Node):
        param_primal = self.view_name_to_primal[node.name]
        param = self.primal_to_param[param_primal]
        with gm.graph.inserting_before(primal):
            param_node = gm.graph.placeholder(param_primal.name, default_value=param)
        usages = self._find_param_usages(gm, self._find_primal_views(gm, primal))

        with gm.graph.inserting_before(usage[0]):
            new_node

    def _compile_fwd(self, gm: fx.GraphModule, inps):
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
                if p is not None and hasattr(p, "_dtags"):
                    assert node.target not in self.primal_to_param, (
                        f"inserting {node.target} twice"
                    )
                    self.primal_to_param[node.target] = p

                    for tag in p._dtags:
                        if tag.dttype == DTensorType.ONDEMAND:
                            self._handle_ondemand_fwd(gm, node, tag.pg)

        logging.info(
            "\nFinished compiling forward, identified following Distributed Tensors\n" +
            "\n".join([f"{pl} : {pm._dtags}" for pl, pm in self.primal_to_param.items()])
        )

        gm.graph.lint()
        gm.recompile()
        logging.info("Modified forward")
        gm.graph.print_tabular()
        self.fwd_gm = gm
        return gm

    def _compile_bwd(self, gm: fx.GraphModule, inps):
        logging.info("Compiling backward")
        logging.info("Original backward graph")
        gm.graph.print_tabular()

        # insert individual allgather
        self._recover_param_primals(gm)
        logging.info("==== After recover param primals")
        gm.graph.print_tabular()
        """
        self.view_name_to_primal = {v.name : p for v, p in self.view_to_primal}
        primal_to_param = {}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.name in view_name_to_primal:
                param_primal = self.view_name_to_primal[node.name]
                param = self.primal_to_param[param_primal]
                for tag in param._dtags:
                    if tag.dttype == DTensorType.ONDEMAND:
                        self._handle_ondemand_bwd(gm, node, tag.pg)
        """

        # insert individual allreduce
        pgs = {}
        for node in gm.graph.nodes:
            if node.op == "output":
                # HACK: again, relying on the implicit guarantee that primals
                # and gradient outputs follow the same order.
                i = 0
                for grad_node in node.args[0][:self.n_grads]:
                    i += 1
                    primal = f"primals_{i}"
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
        self.bwd_gm = gm
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
    global engine
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    n_features = 20
    # create local model on CPU
    model = MyModel(n_features, 2)
    # tag all parameters as replicated tensor
    # model = DDP(model)
    model = FSDP(model)
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
    """
    world_size = 1
    mp.spawn(run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    """
    run_worker(0, 1)