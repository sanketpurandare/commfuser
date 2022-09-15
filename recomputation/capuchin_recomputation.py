import math
import statistics
import time
from enum import Enum
from enum import auto
from turtle import backward
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import OrderedDict
from typing import Set
from typing import Tuple

import tabulate
import torch
import torch.fx as fx
import torch.nn.functional as F
import torch.utils._pytree as pytree
import torchvision
import torchvision.models as models
from functorch._src.named_members_polyfill import _named_buffers
from functorch._src.named_members_polyfill import _named_parameters
from functorch._src.partitioners import _extract_graph_with_inputs_outputs
from functorch.compile import aot_function
from functorch.compile import aot_module
from functorch.compile import draw_graph
from numpy import Inf
from torch.fx import GraphModule
from torch.fx import Interpreter
from torch.fx import Node
from torch.fx.node import map_arg
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function

from graph_profiling.graph_profiler import GraphProfiler
from graph_profiling.graph_profiler_utils import MEM_LIMIT
from graph_profiling.graph_profiler_utils import BiDict
from graph_profiling.graph_profiler_utils import GraphType
from graph_profiling.graph_profiler_utils import IntNodeInfo
from graph_profiling.graph_profiler_utils import M2Model
from graph_profiling.graph_profiler_utils import ModelType
from graph_profiling.graph_profiler_utils import NodeInfo
from graph_profiling.graph_profiler_utils import TensorStatus
from graph_profiling.graph_profiler_utils import get_model_graphs
from graph_profiling.graph_profiler_utils import get_tensor_stat


class CapuchinDistExecutor(Interpreter):
    def __init__(
        self,
        graphmod: GraphModule,
        gtype: GraphType,
        node_info: Dict[Node, NodeInfo],
        fw_bw_map: BiDict[Node, Node],
        recomps: Set[Node],
        gradients: List[Node],
        grad_info: Dict[Node, NodeInfo],
        bucket_list: List[int],
    ):
        super().__init__(graphmod, garbage_collect_values=True)
        self.gtype: GraphType = gtype
        self.node_info: Dict[Node, NodeInfo] = node_info
        self.fw_bw_map: BiDict[Node, Node] = fw_bw_map
        self.env: Dict[Node, Any] = {}
        self.recomps: Set[Node] = recomps
        self.gradients: List[Node] = gradients
        self.grad_info: Dict[Node, NodeInfo] = grad_info
        self.bucket_list: List[int] = bucket_list
        self.bucket_intervals: Dict[Node, int] = {}
        self.exe_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.data_stream: torch.cuda.Stream = torch.cuda.Stream()

    def init_env(self, *input_env):
        # populate the environment for the forward/backward pass
        # and any recomputation interpreters
        input_env_interator: Iterator[Any] = iter(input_env)
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                self.env[node] = next(input_env_interator)

        if self.gtype == GraphType.forward:
            for rp in self.recomps:
                rp_info: IntNodeInfo = self.node_info[rp]
                primal_count = len(rp_info.rcomp_primals)
                primals_itr: Iterator = iter(rp_info.rcomp_primals)
                for node in rp_info.rcomp_graph_mod.graph.nodes:
                    if node.op == "placeholder":
                        if primal_count == 0:
                            break
                        fw_node: Node = next(primals_itr)
                        rp_info.rcomp_executor.env[node] = self.env[fw_node]
                        primal_count -= 1

        if self.gtype == GraphType.backward:
            for rp in self.recomps:
                rp_info: IntNodeInfo = self.node_info[rp]
                primal_count = len(rp_info.rcomp_primals)
                back_rcomp_srcs: List[Node] = [
                    self.fw_bw_map[n] for n in rp_info.rcomp_sources
                ]
                back_rcomp_iter: Iterator = iter(back_rcomp_srcs)
                for node in rp_info.rcomp_graph_mod.graph.nodes:
                    if node.op == "placeholder":
                        if primal_count > 0:
                            primal_count -= 1
                            continue
                        else:
                            bw_node: Node = next(back_rcomp_iter)
                            rp_info.rcomp_executor.env[node] = self.env[bw_node]

    def run(self, *args) -> Any:
        if self.gtype == GraphType.backward:
            self.bucket_accumulated = 0
            self.bucket_iter: Iterator = iter(self.bucket_list)
            self.current_bucket = next(self.bucket_iter)
        return_val = super().run(*args, initial_env=self.env)
        self.env = {}
        return return_val

    def run_node(self, n: Node) -> Any:
        if n.op == "placeholder":
            return super().run_node(n)

        if self.gtype == GraphType.backward:

            # recompute input nnodes that were deleted during forward pass
            nodes_to_recompute: List[Node] = self.node_info[n].to_recompute
            if nodes_to_recompute is not None:
                for rnode in nodes_to_recompute:
                    # print("Recomputing: ",str(rnode))
                    r_info: IntNodeInfo = self.node_info[rnode]
                    r_back: Node = self.fw_bw_map[rnode]
                    with torch.cuda.stream(self.exe_stream):
                        rval = r_info.rcomp_executor.run(
                            None, initial_env=r_info.rcomp_executor.env
                        )
                        r_info.rcomp_executor.env = {}
                    assert isinstance(rval, list)
                    self.env[r_back] = rval[0]
                    rval = None
                    r_info.status = TensorStatus.recomputed
                    # print("Recomputation Complete.")

        with torch.cuda.stream(self.exe_stream):
            return_val = super().run_node(n)

        if self.gtype == GraphType.backward:
            if n in self.gradients:
                g_info: IntNodeInfo = self.grad_info[n]
                self.bucket_accumulated += g_info.memory_size

                if (
                    self.current_bucket is not None
                    and self.bucket_accumulated >= self.current_bucket
                ):
                    # TODO: Need to initiate the NCCL call here
                    print("Bucket full at : ", str(n))
                    self.bucket_accumulated = 0
                    self.current_bucket = next(self.bucket_iter, None)

            elif n.op == "output":
                if self.bucket_accumulated > 0:
                    # TODO: Need to initiate the NCCL call here
                    print("Bucket full at: ", str(n))
                    self.bucket_accumulated = 0

        if self.gtype == GraphType.forward:
            nodes_to_recompute: List[Node] = self.node_info[n].to_delete
            if nodes_to_recompute is not None:
                for rnode in nodes_to_recompute:
                    # print("Deleting: ", str(rnode))
                    r_info: IntNodeInfo = self.node_info[rnode]
                    r_info.status = TensorStatus.deleted
                    t = self.env[rnode]
                    self.env[rnode] = None
                    assert isinstance(t, torch.Tensor)
                    del t
                    t = None

        return return_val


################################################################################################################################
# Capuchin Class
# Accepts a module, loss function, input sample and target sample
# Produces the forward and backward graph using aot_autograd
# Profiles the forward and backward graph
# Gives three choices for scheduling policies
# 1) Swap Only
# 2) Recompute only
# 3) Hybrid (Swap + Recompute)


class CapuchinDistributed:
    def __init__(
        self,
        m_type: ModelType,
        num_models: int,
        bucket_size: int,
        loss_function: F,
        input_sample: torch.Tensor,
        target_sample: torch.Tensor,
    ) -> None:
        self.input: torch.Tensor = input_sample.cuda()
        self.target: torch.Tensor = target_sample.cuda()
        graph_modules: Dict[str, Any] = get_model_graphs(
            m_type, num_models, loss_function, self.input, self.target
        )
        self.fw_module: GraphModule = graph_modules.get("fw_module")
        self.bw_module: GraphModule = graph_modules.get("bw_module")
        self.num_outs: int = graph_modules.get("n_outs")
        self.params = graph_modules.get("params")
        self.buffers = graph_modules.get("buffers")
        self.grad_info: Dict[Node, IntNodeInfo] = {}
        self.gradients: List[Node] = self.get_gradient_nodes()
        self.profile(3)
        self.set_bucket_size(bucket_size)
        # print(self.fw_module.graph)
        # print(self.bw_module.graph)
        self.recomps: Set[Node] = set()
        self.init_executors()

    def get_gradient_nodes(self) -> List[Node]:
        gradients: List[Node] = []
        for n in self.bw_module.graph.nodes:
            if n.op == "output":
                op_nodes = pytree.tree_flatten((n.args, n.kwargs))[0]
                for onode in op_nodes:
                    gradients.append(onode)
                    self.grad_info[onode] = IntNodeInfo()
        return gradients

    def set_bucket_size(self, B):
        self.bucket_list: List[int] = []
        total_grad_size = self.total_grad_size

        while total_grad_size > 0:
            if total_grad_size - B > 0:
                self.bucket_list.append(B)
            else:
                self.bucket_list.append(B - total_grad_size)
            total_grad_size -= B

    def profile(self, profile_runs: int):
        # gets the profiling information for the forward and backward graphs
        print("Profiling Warm-up..")
        fw_profiler = M2Profiler(self.fw_module, True, GraphType.forward)
        fw_profiler.intermediate_nodes = (
            fw_profiler.intermediate_nodes[self.num_outs :]
            + fw_profiler.intermediate_nodes[0 : self.num_outs]
        )
        fw_profiler.fwd_intermediate_nodes_flags = (
            fw_profiler.fwd_intermediate_nodes_flags[self.num_outs :]
            + fw_profiler.fwd_intermediate_nodes_flags[0 : self.num_outs]
        )
        bw_profiler = M2Profiler(
            self.bw_module,
            True,
            GraphType.backward,
            fw_profiler.fwd_intermediate_nodes_flags,
            fw_profiler.intermediate_nodes,
            fw_profiler.node_info,
        )
        with torch.no_grad():
            fw_outputs = fw_profiler.run(
                *self.params, *self.buffers, self.input, self.target
            )
            fw_outputs = fw_outputs[self.num_outs :] + fw_outputs[0 : self.num_outs]
            bw_profiler.init_backward_env(fw_outputs)
            fw_outputs = None
            bw_outputs = bw_profiler.run()
            bw_op_iterator = iter(bw_outputs)
            self.total_grad_size = 0
            for gnode in self.gradients:
                g_tensor: torch.Tensor = next(bw_op_iterator)
                if g_tensor is not None:
                    assert isinstance(g_tensor, torch.Tensor)
                    size, numel, memory_size = get_tensor_stat(g_tensor)
                    g_info: IntNodeInfo = self.grad_info[gnode]
                    g_info.memory_size = memory_size
                    g_info.numel = numel
                    g_info.size = size
                    self.total_grad_size += memory_size
            bw_outputs = None
        print("Total Gradient Size: ", self.total_grad_size)
        print("Memory after warm-up: ", torch.cuda.memory_allocated())
        bw_profiler.reset_stats()
        fw_profiler.reset_stats()
        for i in range(profile_runs):
            print("Profiling Iteration: ", i)
            with torch.no_grad():
                fw_outputs = fw_profiler.run(
                    *self.params, *self.buffers, self.input, self.target
                )
                fw_outputs = fw_outputs[self.num_outs :] + fw_outputs[0 : self.num_outs]
                bw_profiler.init_backward_env(fw_outputs)
                fw_outputs = None
                bw_outputs = bw_profiler.run()
                bw_op_iterator = iter(bw_outputs)
                for m_inp in self.params:
                    m_inp.grad = next(bw_op_iterator)
                bw_outputs = None
            torch.cuda.empty_cache()

        fw_profiler.summary()
        self.node_info: Dict[Node, NodeInfo] = None
        self.fw_bw_map: BiDict[Node, Node] = None
        self.intermediate_nodes: List[Node] = fw_profiler.intermediate_nodes
        self.node_info, self.fw_bw_map = bw_profiler.summary()
        self.peak_end = bw_profiler.peak_end
        print(str(self.peak_end))
        # print(fw_profiler.print_summary())
        # print(bw_profiler.print_summary())
        self.max_peak_mem = max(fw_profiler.max_peak_mem, bw_profiler.max_peak_mem)
        self.min_peak_mem = max(fw_profiler.min_peak_mem, bw_profiler.min_peak_mem)
        print("Maximum Peak Memory Requirements: ", self.max_peak_mem)
        print("Minimum Peak Memory Requirements: ", self.min_peak_mem)
        print("GPU Memory Limit: ", MEM_LIMIT)

    def init_executors(self):
        print("Finding Tensors to Swap/Recompute..")
        self.recompute_only()
        print("Selected Tensors for Recompute: ", [str(node) for node in self.recomps])
        print("Initializing execution engine..")

        self.fw_executor = CapuchinDistExecutor(
            self.fw_module,
            GraphType.forward,
            self.node_info,
            self.fw_bw_map,
            self.recomps,
            self.gradients,
            self.grad_info,
            self.bucket_list,
        )
        self.bw_executor = CapuchinDistExecutor(
            self.bw_module,
            GraphType.backward,
            self.node_info,
            self.fw_bw_map,
            self.recomps,
            self.gradients,
            self.grad_info,
            self.bucket_list,
        )

        print("Ready to execute..")

    def execute(self, input: torch.Tensor, target: torch.tensor) -> torch.Tensor:
        self.input.copy_(input)
        self.target.copy_(target)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        with torch.no_grad():
            t_start = time.time() * 1000
            self.fw_executor.init_env(
                *self.params, *self.buffers, self.input, self.target
            )

            fw_outputs = self.fw_executor.run()
            torch.cuda.nvtx.range_pop()
            loss = fw_outputs[0 : self.num_outs]
            fw_outputs = fw_outputs[self.num_outs :] + fw_outputs[0 : self.num_outs]
            self.bw_executor.init_env(*fw_outputs)
            fw_outputs = None

            bw_outputs = self.bw_executor.run()

            # bw_op_iterator = iter(bw_outputs)
            # for m_inp in self.params:
            #     m_inp.grad = next(bw_op_iterator)
            bw_outputs = None
            torch.cuda.synchronize()
            t_end = time.time() * 1000
        print(torch.cuda.max_memory_allocated())
        print("Total Model Execution time: ", (t_end - t_start))
        return loss

    #####################################################################################################
    # Functions for Recompute

    def get_fw_placeholders(self) -> List[Node]:
        placeholders: List[Node] = []
        for node in self.fw_module.graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
        return placeholders

    def recompute_only(self):
        candidates: List[Node] = list(self.intermediate_nodes)
        placeholders: List[Node] = self.get_fw_placeholders()
        mem_saving = self.max_peak_mem - MEM_LIMIT
        print("Required mem_savings: ", mem_saving)
        self.initMSPS(candidates, placeholders)

        while mem_saving > 0:
            t = self.get_max_MSPS_candidate(candidates)
            print("Candidate: ", str(t), " selected for recompute")
            t_info: IntNodeInfo = self.node_info[t]
            exe_count = self.update_existing_recomps(t)
            self.recomps.add(t)
            candidates.remove(t)
            mem_saving -= t_info.memory_size
            self.update_rem_candidates(t, exe_count, candidates)
        self.prep_recomps()

    def update_existing_recomps(self, t: Node) -> int:
        exe_count = 1
        for rp in self.recomps:
            rp_info: IntNodeInfo = self.node_info[rp]
            t_info: IntNodeInfo = self.node_info[t]
            if t in rp_info.rcomp_sources:
                # NOTE: Think about how you want to use rcomp_extra
                rp_info.rcomp_extra.append(t)
                rp_info.rcomp_sources = [
                    src for src in rp_info.rcomp_sources if src != t
                ]
                rp_info.rcomp_sources.extend(t_info.rcomp_sources)
                rp_info.rcomp_primals.extend(t_info.rcomp_primals)
                exe_count += 1
        return exe_count

    def update_rem_candidates(
        self, t: Node, exe_count: int, candidates: List[Node]
    ) -> None:
        t_info: IntNodeInfo = self.node_info[t]
        for cand in candidates:
            cand_info: IntNodeInfo = self.node_info[cand]
            # Case 1:
            if t in cand_info.rcomp_sources:
                cand_info.rcomp_extra.append(t)
                cand_info.rcomp_sources = [
                    src for src in cand_info.rcomp_sources if src != t
                ]
                cand_info.rcomp_sources.extend(t_info.rcomp_sources)
                cand_info.rcomp_primals.extend(t_info.rcomp_primals)
                cand_info.rcomp_time += t_info.rcomp_time
                cand_info.exe_time = cand_info.rcomp_time

                for rp in self.recomps:
                    rp_info: IntNodeInfo = self.node_info[rp]
                    if cand in rp_info.rcomp_sources:
                        cand_info.exe_time += cand_info.rcomp_time
                cand_info.updateMSPS()
            # Case 2:
            if cand in t_info.rcomp_sources:
                cand_info.exe_time = exe_count * cand_info.rcomp_time
                cand_info.updateMSPS()

    def prep_recomps(self):
        # for each recomp_node in recomps
        # 1) extract subgraph from the forward pass
        # 2) initialize an interepreter for the recomp
        # 3) add the recomp_node to be deleted during it's last forward access
        # 4) add the recomp node to be recomputed during it's first backward access

        for rp in self.recomps:
            rp_info: IntNodeInfo = self.node_info[rp]
            # rp_info.rcomp_primals.reverse()
            # rp_info.rcomp_sources.reverse()
            # rp_info.rcomp_extra.reverse()
            rp_info.rcomp_primals = list(OrderedDict.fromkeys(rp_info.rcomp_primals))
            rp_info.rcomp_sources = list(OrderedDict.fromkeys(rp_info.rcomp_sources))
            rp_info.rcomp_extra = list(OrderedDict.fromkeys(rp_info.rcomp_extra))
            rcomp_graph = _extract_graph_with_inputs_outputs(
                self.fw_module.graph,
                rp_info.rcomp_primals + rp_info.rcomp_sources,
                [rp],
            )
            # print("Recomputation Graph for: ", str(rp))
            # print(rcomp_graph)
            rp_info.rcomp_graph_mod = GraphModule(self.fw_module, rcomp_graph)
            rp_info.rcomp_executor = Interpreter(
                rp_info.rcomp_graph_mod, garbage_collect_values=True
            )
            last_fw: Node = rp_info.last_forward_access
            last_fw_info: NodeInfo = self.node_info[last_fw]
            last_fw_info.to_delete.append(rp)
            last_bw: Node = rp_info.first_back_access
            last_bw_info: NodeInfo = self.node_info[last_bw]
            last_bw_info.to_recompute.append(rp)

    def initMSPS(self, candidates: List[Node], placeholders: List[Node]):
        def populate_sources_from_candidates(
            node: Node, candidates: List[Node], placeholders: List[Node]
        ) -> Tuple[List[Node], List[Node], List[Node]]:
            inp_nodes: List[Node] = node.all_input_nodes
            srcs: List[Node] = []
            primals: List[Node] = []
            extra_rcomp: List[Node] = []
            for i_node in inp_nodes:
                if i_node in candidates:
                    srcs.append(i_node)
                elif i_node in placeholders:
                    primals.append(i_node)
                else:
                    # TODO: CHeck what you really want to store here
                    # Storing only the intermediate tensors is useful for caching them during future
                    extra_rcomp.append(i_node)
                    s, p, r = populate_sources_from_candidates(
                        i_node, candidates, placeholders
                    )
                    srcs.extend(s)
                    primals.extend(p)
                    extra_rcomp.extend(r)
            return (srcs, primals, extra_rcomp)

        candidate_summaries: List[List[Any]] = []
        for cand in candidates:
            n_info: IntNodeInfo = self.node_info[cand]
            (
                n_info.rcomp_sources,
                n_info.rcomp_primals,
                n_info.rcomp_extra,
            ) = populate_sources_from_candidates(cand, candidates, placeholders)
            r_time: float = 0
            for n in n_info.rcomp_extra:
                r_time += self.node_info[n].run_time

            n_info.exe_time = n_info.rcomp_time = r_time + n_info.run_time
            n_info.MSPS = n_info.memory_size / n_info.exe_time
            candidate_summaries.append([str(cand), n_info.exe_time, n_info.memory_size])
        headers: List[str] = ["Candidate", "Cand Exe Time(ms)", "Cand Mem Size(B)"]
        # print(tabulate.tabulate(candidate_summaries, headers=headers))

    def get_max_MSPS_candidate(self, candidates: List[Node]) -> Node:
        max_cand: Node = None
        max_MSPS: float = 0
        for cand in candidates:
            cand_info: IntNodeInfo = self.node_info[cand]
            if cand_info.MSPS > max_MSPS:
                max_MSPS = cand_info.MSPS
                max_cand = cand
        return max_cand

    def recompute_overhead(self, t: Node) -> float:
        return self.node_info[t].exe_time
