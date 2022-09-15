import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from enum import auto
from functools import partial
from functools import reduce
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.multiprocessing as mp
import torch.utils._pytree as pytree
from functorch.compile import aot_function
from functorch.compile import aot_module
from functorch.compile import draw_graph
from torch import nn
from torch.distributed import ProcessGroup


class MyModel(nn.Module):
    def __init__(self, n_features, n_layers):
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(n_features, n_features) for _ in range(n_layers)]
        )

    def forward(self, x):
        return self.seq(x)


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


class FSDP(nn.Module):
    """
    Tag each param as ondemand
    """

    def __init__(self, module: nn.Module, pg: ProcessGroup = None):
        super().__init__()
        self.module = module

        _tag_module(module, DTensorTag(dttype=DTensorType.ONDEMAND, pg=pg))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def ondemand_allgather(param, pg):
    logging.info(f"AllGather param {param.shape}")
    # HACK: using attributes on the parameter to keep track of local shard and
    # original size. These should all be handled by DistributedTensor when ready.
    local_shard = param._local_shard
    orig_size = param._orig_size
    with torch.no_grad():
        world_size = dist.get_world_size(group=pg)
        buffer = torch.empty(
            [world_size] + list(local_shard.shape), device=param.device
        )
        tensors = [buffer[i] for i in range(world_size)]
        # HACK: using synchronous allgather to demonstrate feasibility. This
        # should be asynchronous in the final stack.
        dist.all_gather(tensors, local_shard, group=pg)
        size = list(orig_size)
        numel = reduce(lambda x, y: x * y, size, 1)
        param.data = buffer[:numel].view(size)

    return param


# HACK: the second argument to this function is the output from the last usage
# of the parameter. Passing that in and ignore it to make sure parameter is not
# discarded before usage.
def ondemand_discard(param, _):
    logging.info(f"Discard param {param.shape}")
    with torch.no_grad():
        param.data = param._local_shard


def ondemand_reducescatter(grad, pg):
    with torch.no_grad():
        world_size = dist.get_world_size(group=pg)
        rank = dist.get_rank(group=pg)

        padded_size = int(math.ceil(grad.numel() / world_size))
        output = torch.empty([padded_size], device=grad.device)
        inputs_tensor = torch.empty([padded_size * world_size], device=grad.device)
        inputs_tensor[: grad.numel()].copy_(grad.view(-1))
        inputs = list(inputs_tensor.chunk(world_size))
        dist.reduce_scatter(output, inputs, group=pg)
        return output


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

    def __init__(self, module: nn.Module, train_step: Callable):
        # HACK: Meta device tracing is not ready. Have to create the module on
        # CPU for now.
        self.module = module
        # HACK: train_step is ignored at this time, as AOTAutograd cannot trace
        # through the full fwd + bwd + opt.step yet. Based on the discussion with
        # compiler this, this is addressable.
        self.train_step = train_step
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
        self.primal_to_param = {}
        self.grad_to_primal = {}
        self.pytree_params = [
            p for _, p in list(pytree.tree_flatten(module.named_parameters())[0][0])
        ]
        self.pytree_params.reverse()

        self.compiled_m = None
        # HACK: FSDP triggers recompilation after sharding param storage. To
        # avoid that recompilation, explicitly calling on fwd and bwd
        # GraphModules.
        self.fwd_gm = None
        self.bwd_gm = None

        for p in self.module.parameters():
            if hasattr(p, "_dtags"):
                for tag in p._dtags:
                    if tag.dttype == DTensorType.ONDEMAND:
                        self._prepare_param_shard(p, tag.pg)

    def run(self, x: torch.Tensor):
        if self.compiled_m is None:
            self.compiled_m = aot_module(
                self.module, self._compile_fwd, self._compile_bwd
            )

        if self.fwd_gm is None or self.bwd_gm is None:
            # HACK: AOTAutograd cannot trace the train_step yet, so compile the
            # module for now.
            self.compiled_m(x)
            assert (
                self.fwd_gm is not None and self.bwd_gm is not None
            ), "Forward and backward GraphModules are not generated."

        # HACK: Have to directly call fwd and bwd GraphModule to avoid
        # recompilation. Ideally, it will be helpful to control which guards
        # can be skipped.
        outs = self.fwd_gm(*self.pytree_params, x)
        out, activations = outs[0], outs[1:]
        # HACK: using a fack grad for output to trigger backward
        out_grad = torch.ones_like(out)
        self.bwd_gm(*activations, out_grad)

    def _prepare_param_shard(self, param: torch.nn.Parameter, pg: ProcessGroup):
        with torch.no_grad():
            world_size = dist.get_world_size(group=pg)
            rank = dist.get_rank(group=pg)

            padded_size = int(math.ceil(param.numel() / world_size))
            buffer = torch.empty([padded_size], device=param.device)
            offset = rank * padded_size
            to = min(offset + padded_size, param.numel())
            buffer[: (to - offset)] = param.view(-1)[offset:to]
            param._local_shard = buffer
            param._orig_size = param.size()
            # HACK: cannot set param.data to the shard yet, because AOTAutograd
            # requires to run the module once to get the graph. Eventually, we
            # need to make compilers work with meta device.

    # Find all views of a parameter, and return a dict that maps child view to
    # parent view.
    def _find_primal_views(
        self, gm: fx.GraphModule, primal: fx.Node
    ) -> Dict[fx.Node, fx.Node]:
        view_to_parent = {primal: primal}
        for node in gm.graph.nodes:
            if all(
                [
                    node.op == "call_function"
                    and str(node.target) == "aten.t"
                    and len(node.args) == 1
                    and node.args[0] in view_to_parent
                ]
            ):
                view_to_parent[node] = node.args[0]

        return view_to_parent

    # Find all usages on parameter and its views in the graph. This later helps
    # to insert allgather before first usage and discard after the last usage
    def _find_param_usages(
        self, gm: fx.GraphModule, views: Set[fx.Node]
    ) -> List[fx.Node]:
        usages = []
        for node in gm.graph.nodes:
            for view in views:
                if view in node.args:
                    usages.append(node)

        return usages

    # For one parameter primal, insert allgather before first usage, and discard
    # after the last usage.
    def _handle_one_param_primal(
        self, gm: fx.GraphModule, primal: fx.Node, pg: ProcessGroup
    ):
        views = self._find_primal_views(gm, primal)
        self.view_to_parent.update(views)
        usages = self._find_param_usages(gm, set(views.keys()))

        # insert allgather before first usage
        with gm.graph.inserting_before(usages[0]):
            new_node = gm.graph.call_function(ondemand_allgather, args=(primal, pg))
            usages[0].replace_input_with(primal, new_node)

        # insert reshard after last usage
        with gm.graph.inserting_after(usages[-1]):
            gm.graph.call_function(ondemand_discard, args=(primal, usages[-1]))

    def _compile_fwd(self, gm: fx.GraphModule, inps):
        # HACK: use pytree order of params to map to primals, and save the info
        # for compile_bwd.
        def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
            idx = int(primal_name.split("_")[-1]) - 1
            params = [
                p for _, p in list(pytree.tree_flatten(model.named_parameters())[0][0])
            ]
            return params[idx] if idx < len(params) else None

        logging.info("Compiling forward")
        gm.graph.print_tabular()
        # get tags on each param
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.target.startswith("primal"):
                p = to_param(self.module, node.name)
                if p is not None and hasattr(p, "_dtags"):
                    assert (
                        node.target not in self.primal_to_param
                    ), f"inserting {node.target} twice"
                    self.primal_to_param[node.target] = p

                    for tag in p._dtags:
                        if tag.dttype == DTensorType.ONDEMAND:
                            self._handle_one_param_primal(gm, node, tag.pg)

            # HACK: AOTAutograd records parameter views instead of parameters
            # and use them in the backward graph, which makes the FSDP fwd logic
            # differ from backward. Change this behavior to use parameter primal
            # as the fwd output to make allgather logic consistent in both
            # graphs.
            if node.op == "output":
                new_args = ([],)
                for i, arg in enumerate(node.args[0]):
                    if arg in self.view_to_parent:
                        view = arg
                        while view != self.view_to_parent[view]:
                            view = self.view_to_parent[view]
                        new_args[0].append(view)
                    else:
                        new_args[0].append(arg)
                node.args = new_args

        logging.info(
            "\nFinished compiling forward, identified following Distributed Tensors\n"
            + "\n".join(
                [f"{pl} : {pm._dtags}" for pl, pm in self.primal_to_param.items()]
            )
        )

        gm.graph.lint()
        gm.recompile()
        logging.info("Modified forward")
        gm.graph.print_tabular()
        # HACK: record the graph and directly call it.
        self.fwd_gm = gm
        return gm

    def _compile_bwd(self, gm: fx.GraphModule, inps):
        logging.info("Compiling backward")
        logging.info("Original backward graph")
        gm.graph.print_tabular()

        # insert individual allgather
        view_name_to_node = {v.name: v for v, p in self.view_to_parent.items()}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.name in view_name_to_node:
                # Found a view of parameter primal, retrieve all precedding ops
                # that generate this view
                view_node = view_name_to_node[node.name]
                node_to_insert = []
                while view_node != self.view_to_parent[view_node]:
                    node_to_insert.append(view_node)
                    view_node = self.view_to_parent[view_node]

                node_to_insert.append(view_node)
                node_to_insert.reverse()
                new_nodes = {}

                # bwd and fwd are different graphs, so bwd cannot directly
                # access fwd node as argument. Apply transform.
                def arg_transform(arg):
                    if arg.name in new_nodes:
                        return new_nodes[arg.name]
                    else:
                        raise RuntimeError(f"Unrecognized arg {arg}")

                # inserting the ops that generated the view all the way up to
                # the parameter primal placeholder
                param_primal = None
                with gm.graph.inserting_before(node):
                    for to_insert in node_to_insert:
                        for arg in to_insert.args:
                            new_node = gm.graph.node_copy(
                                arg, arg_transform=arg_transform
                            )
                            new_nodes[arg.name] = new_node
                        new_node = gm.graph.node_copy(
                            to_insert, arg_transform=arg_transform
                        )
                        new_nodes[to_insert.name] = new_node
                        param_primal = (
                            new_node if new_node.op == "placeholder" else param_primal
                        )

                node.replace_all_uses_with(new_node)

                # After usages of the view and insert reshard after last usage
                views = set(self._find_primal_views(gm, new_node).keys())
                usages = self._find_param_usages(gm, views)

                with gm.graph.inserting_after(usages[-1]):
                    gm.graph.call_function(
                        ondemand_discard, args=(param_primal, usages[-1])
                    )

                # erase original view node
                gm.graph.erase_node(node)

        logging.info("After recover param primals")
        gm.graph.print_tabular()

        logging.info("Insert Grad ReduceScatter")
        for node in gm.graph.nodes:
            if node.op == "output":
                new_output_args = []
                # HACK: again, relying on the implicit guarantee that primals
                # and gradient outputs follow the same order.
                i = 0
                for grad_node in node.args[0][: self.n_grads]:
                    i += 1
                    primal = f"primals_{i}"
                    self.grad_to_primal[grad_node.name] = primal
                    for dtag in self.primal_to_param[primal]._dtags:
                        if dtag.dttype == DTensorType.ONDEMAND:
                            with gm.graph.inserting_after(grad_node):
                                new_grad_node = gm.graph.call_function(
                                    ondemand_reducescatter, args=(grad_node, dtag.pg)
                                )

                                new_output_args.append(new_grad_node)

                new_output_args.extend(node.args[0][self.n_grads :])
                node.args = new_output_args
                break

        gm.graph.lint()
        gm.recompile()
        logging.info("Modified backward graph")
        gm.graph.print_tabular()

        logging.info("finished compiling backward")
        # HACK: record the graph and directly call it.
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
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    n_features = 20
    # create local model on CPU
    model = MyModel(n_features, 2)
    # tag all parameters as replicated tensor
    # model = DDP(model)
    model.to(rank)
    model = FSDP(model)
    # we should be able to support the following as well
    # DDP(FSDP(model, pg=intra_node), pg=inter_node)

    # compile train_step, insert comm ops based on tags in model, and fuse them
    engine = Engine(model, train_step)
    for i in range(3):
        logging.info(f"================== ITERATION {i} ====================")
        # dummy input
        x = torch.randn(2, n_features).to(rank)
        # run the compiled train_step
        engine.run(x)

    # Discussion:
    # Explicitly passing train_step to Engine rather than using the following API
    # because we might prefer to optimize the entire training step or even
    # multiple training steps together, instead of just optimizing fwd/bwd
    #   model = DistributedModel(model)
    #   train_step(model, x)


if __name__ == "__main__":

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
