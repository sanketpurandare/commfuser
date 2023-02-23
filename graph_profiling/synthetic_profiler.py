import torch
import torch.fx as fx
from typing import Dict, Any, List
from torchbenchmark.util.benchmark_utils import get_benchmark_model
from functorch.compile import aot_module, config
from torch._subclasses import FakeTensor, FakeTensorMode
import torch._dynamo as torchdynamo
config.use_fake_tensor = True
from torchbenchmark import load_model_by_name

def fake_compiler(fx_g: fx.GraphModule, inps):
    for node in fx_g.graph.nodes:
        print(node.meta.get('val', None))

    output_node = [node for node in fx_g.graph.nodes if node.op == 'output'][0]
    output_data = [node.meta['val'] if node is not None else node for node in output_node.args[0]]
    def new_f(args):
        return output_data
    new_f._boxed_call = True
    return new_f

class SyntheticProfiler(fx.Interpreter):
    def __init__(self, graphmod:fx.GraphModule):
        super().__init__(graphmod, garbage_collect_values=True)
        self.attr_map:Dict[fx.Node, Any] = {}


    def run_node(self, n:fx.Node):
        if n.op == 'placeholder':
            self.env[n] = None
            
        
        for inp_node in n.all_input_nodes:
            #collect meta information about the input node
            ft = inp_node.meta.get('val', None)
            size = ft.size()
            dtype= ft.dtype
            self.env[inp_node] = torch.randn(size, dtype=dtype, device= torch.cuda.current_device())

        return_val = super().run_node(n)

        if n.op == "get_attr":
            self.attr_map[n] = return_val

        return return_val

def dynamo_compiler(gm: fx.GraphModule, example_inputs:List[torch.Tensor]):
    compiled_m = aot_module(gm, fake_compiler)
    return compiled_m

def compute_loss(model, example_inputs):
    out = model(**example_inputs)
    return out.loss

if __name__ == "__main__":
    model_name = "hf_Bert"
    batch_size = 4
    device = torch.cuda.current_device()
    Model_Class = load_model_by_name(model_name)
    
    mod_id = f"{model_name}_{batch_size}"

    fakemode = FakeTensorMode()
    # with fakemode:
    model = Model_Class(device="cuda", test="train", batch_size=batch_size)
    print(torch.cuda.memory_allocated())
    optimize_ctx = torchdynamo.optimize(dynamo_compiler)
    compute_loss(optimize_ctx(model.model), model.example_inputs).backward()
