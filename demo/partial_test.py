from torch import fx
from torch import nn
from functorch.compile import aot_function, aot_module, draw_graph
import torch
import torch.distributed as dist
import torchdynamo


from typing import  (
    Callable,
    List,
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


def get_partial_graphs(model):
    # HACK: get these graphs from compiler
    # HACK: this is not a generic solution to get structured graphs, for testing
    # purpose only

    graphs, graph_to_inputs, gid = [], {}, 0
    def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        nonlocal graphs, graph_to_inputs, gid

        gm._siblings, gm._id, gm._inputs = [gm], gid, example_inputs
        gid += 1
        for prior_gm in graphs:
            prior_inputs = graph_to_inputs[prior_gm]
            if all([same_activation(x, y) for x, y in zip(example_inputs, prior_inputs)]):
                prior_gm._siblings.append(gm)
                gm._siblings = prior_gm._siblings
                logging.info(f"Found siblings Sub-Graph-{gm._id} and Sub-Graph-{prior_gm._id}")

        if len(gm._siblings) <= 1:
            graphs.append(gm)
            graph_to_inputs[gm] = example_inputs
        return gm.forward

    dummy_inputs = model.dummy_inputs()
    with torchdynamo.optimize(compiler):
        for x in dummy_inputs:
            model(x)

    structured_graphs = []
    for gm in graphs:
        if len(gm._siblings) > 1:
            structured_graphs.append(set(gm._siblings))
        else:
            structured_graphs.append(gm)

    logging.info(f"Structured Sub-Graphs: {structured_graphs}")
    return structured_graphs


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
        self.compiled = False

        def dummy_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            print("====== dummy compile!")
            return gm.forward

        self.optimize_ctx = torchdynamo.optimize(dummy_compiler)

    def run(self, x: torch.Tensor):

        if not self.compiled:
            dummy_inputs = self.module.dummy_inputs()
            with self.optimize_ctx:
                for dummy_x in dummy_inputs:
                    self.module(dummy_x)

        print("==== running forward!")
        with self.optimize_ctx:
            self.module(x)



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
    #model = DDP(model)
    # we should be able to support the following as well
    # DDP(FSDP(model, pg=intra_node), pg=inter_node)

    # compile train_step, insert comm ops based on tags in model, and fuse them
    engine = Engine(model, train_step)
    engine.run(torch.zeros(2, 10))

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