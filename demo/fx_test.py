import torch
from functorch.compile import aot_function, nop
from functorch import make_fx
from torch.nn.utils import _stateless


class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(x)

mod = Foo().cuda()
# mod = resnet18().cuda()

def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]

inp = (torch.randn(3, 3, device='cuda'),)
# inp = (torch.randn(1, 3, 228, 228, device='cuda'),)

mod(*inp).sum().backward()

optim = torch.optim.Adam(get_sorted_params(dict(mod.named_parameters())), lr=0.01)
optim.step()

def f(params, buffers, optim_state, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(mod, params_and_buffers, args, {}).sum().backward()
    optim = torch.optim.Adam(get_sorted_params(params), lr=0.01)
    optim.load_state_dict(optim_state)
    optim.step()
    return params, buffers, optim_state

print(make_fx(f)(dict(mod.named_parameters()), dict(mod.named_buffers()), optim.state_dict(), inp).code)