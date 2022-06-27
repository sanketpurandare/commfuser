from typing import List
import torch
import torchdynamo
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 10)

    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable


model = MyModel()
x = torch.ones(2, 10)
opt = torch.optim.SGD(model.parameters(), lr=0.01)


def func_step(x, model, params, opt):
    model(x).sum().backward()
    x += 1
    opt.step()

import dis
print(dis.dis(func_step))

with torchdynamo.optimize(my_compiler):
    func_step(x, model, list(model.parameters()), opt)
