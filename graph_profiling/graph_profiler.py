import logging
import statistics
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import tabulate
import torch

from commfuser.graph_profiling.graph_profiler_utils import BiDict
from commfuser.graph_profiling.graph_profiler_utils import GraphProfiler
from commfuser.graph_profiling.graph_profiler_utils import GraphType
from commfuser.graph_profiling.graph_profiler_utils import IntNodeInfo
from commfuser.graph_profiling.graph_profiler_utils import NodeInfo
from commfuser.graph_profiling.graph_profiler_utils import ProfileMode
from commfuser.graph_profiling.graph_profiler_utils import TensorStatus
from commfuser.graph_profiling.graph_profiler_utils import get_tensor_stat
from commfuser.graph_profiling.graph_profiler_utils import profile_mode_dict
from torch.autograd.profiler_util import EventList
from torch.fx import GraphModule
from torch.fx import Interpreter
from torch.fx import Node
from torch.fx.node import map_arg
from torch.profiler import record_function


class GraphProfiler(Interpreter):
    r"""The main GraphProfiler class that extends the fx.Interpreter and runs
    the input graph module node by node, collecting profiling information for
    each one of them.
    Args:
    graphmod (fx.GraphModule): The fx graphmodule to initialize the
                                GraphProfiler.
    gtype (GraphType): The type of fx graph module (forward/backward)
    fwd_profiler (GraphProfiler): To intialize the backward
                                profiler, an instance of forward profiler is
                                required. It is used to create a mapping from
                                nodes of forward graph to the backward graph,
                                merge the profiling information from forward
                                and backward graphs into a single dictionary
                                and use the attributes of the fwd_profiler.
    fw_num_outs (int): The number of outputs in the original dynamo graph used
                        to generate the aot_graphs.
    sync (bool): Flag to indicate whether the cuda stream should be
                synchronized for each node operation.
    profile_mode (str): The Graph Profiler provides three profiling
                        modes,``default``, ``memory`` and ``swap``.

    Dictionaries:
    node_info (Dict[fx.Node, NodeInfo]): Dictionary to store the
                                        mapping of the node
                                        to its NodeInfo object (Profiling
                                        Information)
    fwd_bwd_map (BiDict[fx.Node, fx.Node]): Bidirectional Dictionary to store
                                            the mapping of the forward and
                                            backward pass intermediate nodes.
    intermediate_nodes (List[fx.Node]): Activations retained from the forward
                                        pass for gradient calculation in the
                                        backward pass that are not a part of the
                                        forward graph output in the original
                                        dynamo graph.
    """

    def __init__(
        self,
        graphmod: GraphModule,
        gtype: GraphType,
        fwd_profiler: Optional[GraphProfiler] = None,
        fw_num_outs: Optional[int] = 1,
        sync: Optional[bool] = False,
        profile_mode: Optional[str] = "default",
    ):
        super().__init__(graphmod, True)
        print("Current Device: ",torch.cuda.current_device())
        self.gtype: GraphType = gtype
        self.id: int = graphmod._id
        torch.cuda.reset_peak_memory_stats()
        if self.gtype == GraphType.backward:
            logging.info("Initializing Backward Profiler")
            assert fwd_profiler is not None
            self.prefix_str: str = f"g{self.id}_bw"
            self.sync: bool = fwd_profiler.sync
            self.node_info: Dict[Node, NodeInfo] = fwd_profiler.node_info
            self.fwd_intermediate_nodes: List[Node] = fwd_profiler.intermediate_nodes
            self.fwd_intermediate_nodes_flags: List[
                bool
            ] = fwd_profiler.fwd_intermediate_nodes_flags
            self.fw_num_outs: int = fwd_profiler.fw_num_outs
            self.profile_mode: ProfileMode = fwd_profiler.profile_mode
        else:
            logging.info("Initializing Forward Profiler")
            self.prefix_str: str = f"g{self.id}_fw"
            self.sync: bool = sync
            self.node_info: Dict[Node, NodeInfo] = {}
            self.fwd_intermediate_nodes: List[Node] = []
            self.fwd_intermediate_nodes_flags: List[bool] = []
            self.fw_num_outs: int = fw_num_outs
            self.profile_mode: ProfileMode = profile_mode_dict[profile_mode]

        self.total_runtime_sec: List[float] = []
        self.attr_map: Dict[Node, Any] = {}
        self.node_active_mem: Dict[Node, List[int]] = {}
        self.node_peak_mem: Dict[Node, List[int]] = {}
        self.runtimes_sec: Dict[Node, float] = {}
        self.swaptimes_sec: Dict[Node, float] = {}
        self.node_cuda_time: Dict[Node, float] = {}
        self.node_cpu_time: Dict[Node, float] = {}
        self.node_cuda_swaptime: Dict[Node, float] = {}
        self.node_cpu_swaptime: Dict[Node, float] = {}
        self.intermediate_nodes: List[Node] = []
        self.torch_profiler: torch.profiler.profile = None
        self.prev_runtime = 0
        self.env = {}

        # Can define any variables that you need to measure the runtime events
        # at the Node level

        # If graph type is forward then find the last use of the intermediate
        # tensors during the forward pass The output node contains all the
        # tensors live at the end of forward pass Each node generates an
        # intermediate tensor, check if that tensor is active at end If yes, add
        # it to intermediate tensor list, find its last use node

        if gtype == GraphType.forward:
            # For the intermediate nodes obtain their last use in the forward
            # pass excluding the output node
            node_to_last_forward_use: Dict[Node, Node] = {}
            # stores the last forward uses
            self.user_to_last_forward_uses: Dict[Node, List[Node]] = {}

            for node in self.module.graph.nodes:
                if node.op == "output":
                    # get all the arguments form the output node these are all
                    # the nodes that are live at the end of forward pass
                    op_nodes = node.all_input_nodes[self.fw_num_outs :]
                    for n in op_nodes:
                        # We want to exclude the placeholder nodes since they
                        # represent the model parameters
                        if n.op != "placeholder":
                            ip_nodes: List[Node] = n.all_input_nodes
                            is_placeholder = [
                                True if (inode.op == "placeholder") else False
                                for inode in ip_nodes
                            ]
                            if all(is_placeholder):
                                self.fwd_intermediate_nodes_flags.append(False)
                            else:
                                self.intermediate_nodes.append(n)
                                self.fwd_intermediate_nodes_flags.append(True)
                        else:
                            self.fwd_intermediate_nodes_flags.append(False)
            rank = 0
            for node in self.module.graph.nodes:
                if node.op != "placeholder":
                    if node in self.intermediate_nodes:
                        n_info = IntNodeInfo()
                        # NOTE:This condition is especially for the node that is
                        # directly used in the output after generation but not a
                        # part of the output
                        users_count = len(node.users)
                        if users_count == 1:
                            users = [u for u in node.users.keys()]
                            user = users[0]
                            assert type(user) == Node
                            if user.op == "output":
                                self.user_to_last_forward_uses.setdefault(
                                    user, []
                                ).append(node)
                                n_info.last_forward_access = user

                    else:
                        n_info = NodeInfo()
                    n_info.rank = rank
                    rank += 1
                    n_info.gtype = GraphType.forward
                    self.node_info[node] = n_info

            def register_last_forward_uses(n: Node, user: Node):
                if n not in node_to_last_forward_use and n in self.intermediate_nodes:
                    node_to_last_forward_use[n] = user
                    self.node_info[n].last_forward_access = user
                    self.user_to_last_forward_uses.setdefault(user, []).append(n)

            # We traverse the nodes in a reverse order and find first use of the
            # intermediate tensor in its forward pass
            for node in reversed(self.module.graph.nodes):
                # Exclude the output node
                if node.op != "output":
                    map_arg(node.args, lambda n: register_last_forward_uses(n, node))
                    map_arg(node.kwargs, lambda n: register_last_forward_uses(n, node))

            # For the parameter nodes obtain their first use in the forward pass
            # excluding the output node
            node_to_first_forward_use: Dict[Node, Node] = {}
            # stores the first forward uses
            self.user_to_first_forward_uses: Dict[Node, List[Node]] = {}
            # registering first forward uses for parameters
            def register_first_forward_uses(n: Node, user: Node):
                if n not in node_to_first_forward_use and n.op == "placeholder":
                    node_to_first_forward_use[n] = user
                    self.node_info.setdefault(n, NodeInfo()).first_forward_access = user
                    self.user_to_first_forward_uses.setdefault(user, []).append(n)

            for node in self.module.graph.nodes:
                # Exclude the output node
                if node.op != "output":
                    map_arg(node.args, lambda n: register_first_forward_uses(n, node))
                    map_arg(node.kwargs, lambda n: register_first_forward_uses(n, node))

        if gtype == GraphType.backward:

            # populate the intermediate nodes for the backward pass as well

            fwd_intermediate_nodes_iterator: Iterator[Any] = iter(
                self.fwd_intermediate_nodes
            )
            self.fwd_bwd_intermediate: BiDict[Node, Node] = BiDict()
            placeholders = [
                node for node in self.module.graph.nodes if node.op == "placeholder"
            ]
            len_placeholders = len(placeholders)
            placeholders = placeholders[: (len_placeholders - self.fw_num_outs)]

            assert len(placeholders) == len(self.fwd_intermediate_nodes_flags)

            for node, is_intermediate in zip(
                placeholders, self.fwd_intermediate_nodes_flags
            ):
                if is_intermediate:
                    self.intermediate_nodes.append(node)
                    fwd_node = next(fwd_intermediate_nodes_iterator)
                    self.fwd_bwd_intermediate[fwd_node] = node

            rank = 0
            for node in self.module.graph.nodes:
                if node.op != "placeholder":
                    n_info = NodeInfo()
                    n_info.rank = rank
                    n_info.gtype = GraphType.backward
                    rank += 1
                    self.node_info[node] = n_info

            for node in self.module.graph.nodes:
                last_uses: List[Node] = self.user_to_last_uses.get(node, None)
                if last_uses is not None:
                    for lunode in last_uses:
                        if lunode in self.intermediate_nodes:
                            f_node = self.fwd_bwd_intermediate.inverse.get(lunode)[0]
                            n_info: IntNodeInfo = self.node_info[f_node]
                            n_info.last_back_access = node

            # must have input list of intermediate nodes
            node_to_first_backward_use: Dict[Node, Node] = {}
            self.user_to_first_backward_uses: Dict[Node, List[Node]] = {}

            def register_first_backward_use(n: Node, user: Node):
                if n not in node_to_first_backward_use and n in self.intermediate_nodes:
                    node_to_first_backward_use[n] = user
                    f_node = self.fwd_bwd_intermediate.inverse.get(n)[0]
                    assert isinstance(f_node, Node)
                    n_info: IntNodeInfo = self.node_info[f_node]
                    n_info.first_back_access = user
                    self.user_to_first_backward_uses.setdefault(user, []).append(n)

            for node in self.module.graph.nodes:
                if node.op != "placeholder":
                    map_arg(node.args, lambda n: register_first_backward_use(n, node))
                    map_arg(node.kwargs, lambda n: register_first_backward_use(n, node))

    def meta_run(self, args_list: List[Any]) -> Any:
        args_iter = iter(args_list)
        for n in self.module.graph.nodes:
            if n.op == "placeholder":
                self.env[n] = next(args_iter)
        args_list.clear()
        return self.run([])

    def run(self, *args) -> Any:
        if(self.gtype == GraphType.forward):       
            self.param_memory:int = torch.cuda.memory_allocated()
        return_val = super().run(*args, initial_env=self.env)
        args = None
        if self.gtype == GraphType.backward:
            torch.cuda.synchronize()
            self.param_grad_memory:int = torch.cuda.memory_allocated()
        self.env = {}
        return return_val

    def run_node(self, n: Node) -> Any:
        if n.op == "placeholder":
            return super().run_node(n)

        # preftech the tensors that have been offloaded and have their first
        # uses
        # 1) Get the nodes to be prefetched
        # 2) Retrieve their CPU reference (assert if it resides in pinned
        #    memory)
        # 3) Copy the tensor to GPU memory and add it to the Interpreter
        #    environment
        # 4) Update the state of intermediate tensor in NodeInfo

        if self.profile_mode == ProfileMode.swap:
            if self.gtype == GraphType.backward:
                nodes_to_fetch = self.user_to_first_backward_uses.get(n, None)
                if nodes_to_fetch is not None:
                    for p_node in nodes_to_fetch:
                        f_node = self.fwd_bwd_intermediate.inverse.get(p_node)[0]
                        assert isinstance(f_node, Node)
                        self.node_info[f_node].status = TensorStatus.gpu
                        cpu_ref: torch.Tensor = self.node_info[f_node].cpu_ref
                        assert isinstance(cpu_ref, torch.Tensor) and cpu_ref.is_pinned
                        with record_function(f"g{self.id}_{f_node.name}_swap"):
                            t = cpu_ref.to(
                                device=torch.cuda.current_device(),
                                memory_format=torch.preserve_format,
                            )
                        self.env[p_node] = t.contiguous()
                        if self.sync:
                            torch.cuda.synchronize()

        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        with record_function(f"{self.prefix_str}_{n.name}"):
            return_val = super().run_node(n)

        if self.sync:
            torch.cuda.synchronize()

        if n.op == "get_attr":
            self.attr_map[n] = return_val

        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            mem_stats = torch.cuda.memory_stats()
            self.node_peak_mem.setdefault(n, [])
            self.node_peak_mem[n].append(mem_stats["active_bytes.all.peak"])
            self.node_active_mem.setdefault(n, [])
            self.node_active_mem[n].append(mem_stats["active_bytes.all.current"])
            if self.gtype == GraphType.forward and n in self.intermediate_nodes:
                assert isinstance(return_val, torch.Tensor)
                (
                    self.node_info[n].size,
                    self.node_info[n].numel,
                    self.node_info[n].memory_size,
                ) = get_tensor_stat(return_val)

        # offload the tensors that have last uses at this node during forward
        # pass
        # 1) Get the nodes to be offloaded
        # 2) Retrieve their CPU reference (if none allocate a CPU tensor in
        #    pinned memory)
        # 3) Copy the tensor to the CPU, add the CPU tensor to the Interpreter
        #    environment
        # 4) Delete the GPU tensor
        if self.profile_mode == ProfileMode.swap:
            if self.gtype == GraphType.forward:
                nodes_to_offload: List[Node] = self.user_to_last_forward_uses.get(
                    n, None
                )
                if nodes_to_offload is not None:
                    for o_node in nodes_to_offload:
                        cpu_ref: torch.Tensor = self.node_info[o_node].cpu_ref
                        t = self.env[o_node]
                        assert isinstance(t, torch.Tensor)
                        if cpu_ref == None:
                            cpu_ref = torch.zeros(
                                t.size(), dtype=t.dtype, layout=t.layout
                            ).pin_memory()
                        assert cpu_ref.is_pinned
                        with record_function(f"g{self.id}_{n.name}_swap"):
                            cpu_ref = cpu_ref.copy_(t, False)
                        if self.sync:
                            torch.cuda.synchronize()
                        self.node_info[o_node].status = TensorStatus.cpu
                        self.node_info[o_node].cpu_ref = cpu_ref
                        self.env[o_node] = cpu_ref
                        del t
                        t = None
                        cpu_ref = None

        return return_val

    def reset_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        self.total_runtime_sec: List[float] = []
        self.node_active_mem: Dict[Node, List[Any]] = {}
        self.node_peak_mem: Dict[Node, List[Any]] = {}
        self.runtimes_sec: Dict[Node, List[float]] = {}
        self.swaptimes_sec: Dict[Node, List[float]] = {}

    def get_idle_times(self) -> None:
        for i_node in self.intermediate_nodes:
            if self.gtype == GraphType.forward:
                fn_info: IntNodeInfo = self.node_info[i_node]
                last_use = fn_info.last_forward_access
                fn_info.idle_time = self.total_runtime - (
                    self.node_info[last_use].cumulative_run_time + fn_info.swap_time
                )
            else:
                f_node = self.fwd_bwd_intermediate.inverse.get(i_node)[0]
                fn_info: IntNodeInfo = self.node_info[f_node]
                first_use = fn_info.first_back_access
                fn_info.idle_time += self.node_info[first_use].cumulative_run_time - (
                    self.node_info[first_use].run_time + fn_info.swap_time
                )

    def get_peakmem_usage(self) -> None:
        MEM_LIMIT = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        if self.profile_mode == ProfileMode.swap:
            intermediate_mem = 0
            if self.gtype == GraphType.backward:
                for i_node in self.intermediate_nodes:
                    f_node = self.fwd_bwd_intermediate.inverse.get(i_node)[0]
                    fn_info: IntNodeInfo = self.node_info[f_node]
                    intermediate_mem += fn_info.memory_size

            self.peak_start = None
            self.peak_end = None
            peak_interval: bool = False
            peak_end_reset: bool = False
            self.max_peak_mem = 0
            self.min_peak_mem = 0
            for node in self.module.graph.nodes:
                if node.op == "placeholder":
                    continue
                if self.gtype == GraphType.backward:
                    nodes_to_prefetch = self.user_to_first_backward_uses.get(node, None)
                    if nodes_to_prefetch is not None:
                        for p_node in nodes_to_prefetch:
                            f_node = self.fwd_bwd_intermediate.inverse.get(p_node)[0]
                            intermediate_mem -= self.node_info[f_node].memory_size
                min_peak_mem = self.node_info[node].peak_mem
                peak_mem = min_peak_mem + intermediate_mem
                if peak_mem > MEM_LIMIT:
                    peak_interval = True
                    peak_end_reset = True
                    if self.peak_start is None:
                        self.peak_start = node
                else:
                    peak_interval = False
                    if peak_end_reset:
                        self.peak_end = node
                        peak_end_reset = False

                self.node_info[node].in_peak_interval = peak_interval
                self.node_info[node].total_peak_mem = peak_mem
                self.max_peak_mem = max(self.max_peak_mem, peak_mem)
                self.min_peak_mem = max(self.min_peak_mem, min_peak_mem)

                if self.gtype == GraphType.forward:
                    nodes_to_offload = self.user_to_last_forward_uses.get(node, None)
                    if nodes_to_offload is not None:
                        for o_node in nodes_to_offload:
                            intermediate_mem += self.node_info[o_node].memory_size
        else:
            peak_mem_usages = [
                self.node_info[n].peak_mem
                for n in self.module.graph.nodes
                if n.op != "placeholder"
            ]
            self.max_peak_mem = max(peak_mem_usages)
            self.min_peak_mem = min(peak_mem_usages)
            self.peak_start = None
            self.peak_end = None

    def get_node_runtimes(self):
        event_list_avg: EventList = self.torch_profiler.key_averages()
        event_dict: Dict[str, Tuple[float, float]] = {}
        prefix = f"g{self.id}"
        for e in event_list_avg:
            if prefix in e.key:
                event_dict[e.key] = (e.cuda_time, e.cpu_time)
        for n in self.module.graph.nodes:
            if n.op != "placeholder":
                cuda_time, cpu_time = event_dict[f"{self.prefix_str}_{n.name}"]
                self.node_cuda_time[n] = cuda_time / 1000.0
                self.node_cpu_time[n] = cpu_time / 1000.0
                self.runtimes_sec[n] = max(cpu_time, cuda_time) / 1000.0
        if self.profile_mode == ProfileMode.swap:
            if self.gtype == GraphType.forward:
                for int_n in self.intermediate_nodes:
                    cuda_time, cpu_time = event_dict[f"g{self.id}_{int_n.name}_swap"]
                    self.node_cuda_swaptime[int_n] = cuda_time / 1000.0
                    self.node_cpu_swaptime[int_n] = cpu_time / 1000.0
                    self.swaptimes_sec[int_n] = max(cpu_time, cuda_time) / 1000.0

    def summarize(self) -> Optional[Tuple[Any]]:

        self.get_node_runtimes()
        self.total_runtime = 0

        for node in self.module.graph.nodes:
            if node.op != "placeholder":
                n_info: NodeInfo = self.node_info.setdefault(node, NodeInfo())
                n_info.run_time = self.runtimes_sec.get(node, 1.0)
                n_info.exe_time = n_info.run_time
                self.total_runtime += n_info.run_time
                n_info.cumulative_run_time = self.total_runtime

                if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
                    n_info.peak_mem = max(self.node_peak_mem.setdefault(node, [0]))
                    n_info.active_mem = max(self.node_active_mem.setdefault(node, [0]))

                    if (
                        node in self.intermediate_nodes
                        and self.profile_mode == ProfileMode.swap
                    ):
                        n_info: IntNodeInfo = self.node_info[node]
                        n_info.swap_time = self.swaptimes_sec[node]

        self.get_idle_times()
        self.get_peakmem_usage()

    def print_summary(self) -> str:

        node_summaries: List[List[Any]] = []
        mean_total_runtime = self.total_runtime
        logging.info(f"Execution Time (ms): {self.total_runtime}")
        logging.info(f"Max Peak Mem Usage (B): {self.max_peak_mem}")

        headers: List[str] = [
            "Target",
            "Op",
            "Average runtime (ms)",
            "Pct total runtime",
            "CUDA time(ms)",
            "CPU time(ms)",
        ]
        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            headers.extend(
                [
                    "Mem Active (B)",
                    "Mem Peak Active(B)",
                    "Tensor Size(B)",
                ]
            )
        if self.profile_mode == ProfileMode.swap:
            print("Peak Interval : ", str(self.peak_start), " - ", str(self.peak_end))
            headers.extend(
                [
                    "Swap Time (ms)",
                    "Idle_time(ms)",
                    "Simulated Peak Active(B)",
                ]
            )
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                continue
            n_info: NodeInfo = self.node_info[node]
            pct_total = n_info.run_time / mean_total_runtime * 100
            val_list = [
                node.target,
                str(node),
                n_info.run_time,
                pct_total,
                self.node_cuda_time[node],
                self.node_cpu_time[node],
            ]
            if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
                val_list.extend([n_info.active_mem, n_info.peak_mem])
            if node in self.intermediate_nodes:
                n_info: IntNodeInfo = n_info
                if self.profile_mode == ProfileMode.memory:
                    val_list.append(n_info.memory_size)
                if self.profile_mode == ProfileMode.swap:
                    val_list.extend(
                        [n_info.memory_size, n_info.swap_time, n_info.idle_time]
                    )
            else:
                if self.profile_mode == ProfileMode.memory:
                    val_list.append("")
                if self.profile_mode == ProfileMode.swap:
                    val_list.extend(["", "", ""])
            if self.profile_mode == ProfileMode.swap:
                val_list.append(n_info.total_peak_mem)
            node_summaries.append(val_list)
        return tabulate.tabulate(node_summaries, headers=headers)
