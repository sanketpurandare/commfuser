import logging
import os
import sys
from enum import Enum
from enum import auto
from typing import List
from typing import Set
from typing import Union

import torch
from torch import fx

# Algorithms from paper: Chu, C. (1992). A branch-and-bound algorithm to
# minimize total tardiness with different release dates. Naval Research
# Logistics, 39(2), 265–283
# https://sci-hub.se/10.1002/1520-6750(199203)39:2%3C265::aid-nav3220390209%3E3.0.co;2-l

# Additional references
# Chu, C., and Portmann, M.C., “Minimisation de la Somme des Retards pour les
# Problkmes d’ordonnancement h Une Machine,” Rapport de Recherche No.
# 1023,INRIA, France (1989)
# https://hal.inria.fr/inria-00075535/document

# Chu, C., and Portmann, M.C., “Some New Efficient Methods to Solve the
# n/1/r_i/Sum(T_i), Scheduling Problem,” European Journal of Operational Research, 56
# (1991).
# https://sci-hub.se/https://www.sciencedirect.com/science/article/abs/pii/037722179290071G


class SchedulingPolicy(Enum):
    FCFS = auto()
    GREEDY_IPRTT = auto()
    GREEDY_NDPRTT = auto()
    BRANCH_AND_BOUND = auto()


class Process:
    def __init__(
        self, p_id: int, arrival_time: float, deadline: float, burst_time: float
    ) -> None:
        self.p_id: int = p_id
        self.arrival_time: float = arrival_time
        self.burst_time: float = burst_time
        self.deadline: float = deadline


def fcfs_scheduling(processes: List[Process]) -> List[Process]:
    unscheduled_jobs: List[Process] = list(processes)
    scheduled_processes = sorted(unscheduled_jobs, key=lambda x: x.arrival_time)
    return scheduled_processes


def earliest_beginning(process: Process, current_time: float) -> float:
    # Earliest beginning time of the process defined as max (current_time, arrival_time)
    return max(process.arrival_time, current_time)


def PRTT(process: Process, current_time: float) -> float:
    # Priority rule for total tardiness
    e_b: float = earliest_beginning(process, current_time)
    return e_b + max(e_b + process.burst_time, process.deadline)


def get_min_PRTT(processes: List[Process], current_time: float) -> float:
    min_prtt: float = min([PRTT(p, current_time) for p in processes])
    return min_prtt


def greedy_scheduling_IPRTT(processes: List[Process]):
    # Algorithm Using Insertion Technique and PRTT
    # Step 1: Initialize all the unscheduled jobs, scheduled jobs and the current time
    unscheduled_jobs: List[Process] = list(processes)
    scheduled_jobs: List[Process] = []
    current_time: float = 0

    while len(unscheduled_jobs) > 0:
        # Step 2:
        min_prtt: float = get_min_PRTT(unscheduled_jobs, current_time)
        processes_with_min_prtt: List[Process] = [
            p for p in unscheduled_jobs if PRTT(p, current_time) <= min_prtt
        ]
        min_release_time_process: Process = min(
            processes_with_min_prtt, key=lambda p: p.arrival_time
        )
        earliest_beginning_time: float = earliest_beginning(
            min_release_time_process, current_time
        )
        while True:
            # Step 3:
            unscheduled_pool: List[Process] = [
                p
                for p in unscheduled_jobs
                if p.p_id != min_release_time_process.p_id
                and (earliest_beginning(p, current_time) + p.burst_time)
                <= earliest_beginning_time
            ]

            if len(unscheduled_pool) == 0:
                # Step 5:
                scheduled_jobs.append(min_release_time_process)
                unscheduled_jobs.remove(min_release_time_process)
                current_time = (
                    earliest_beginning_time + min_release_time_process.burst_time
                )
                break
            else:
                # Step 4:
                min_earliest_beginning: float = min(
                    unscheduled_pool, key=lambda p: earliest_beginning(p, current_time)
                )
                processes_with_earliest_beginning: List[Process] = [
                    p
                    for p in unscheduled_pool
                    if earliest_beginning(p, current_time) <= min_earliest_beginning
                ]
                min_prtt_process: Process = min(
                    processes_with_earliest_beginning,
                    key=lambda p: PRTT(p, current_time),
                )
                scheduled_jobs.append(min_prtt_process)
                unscheduled_jobs.remove(min_prtt_process)
                current_time = min_earliest_beginning + min_prtt_process.burst_time
    return scheduled_jobs


def greedy_scheduling_NDPRTT(processes: List[Process]):
    # Algorithm Using Non-delay Schedule and PRTT Rule
    # Step 1: Initialize all the unscheduled jobs, scheduled jobs and the current time
    unscheduled_jobs: List[Process] = list(processes)
    scheduled_jobs: List[Process] = []
    current_time: float = 0
    while len(unscheduled_jobs) > 0:
        # Step 2:
        min_earliest_beginning: float = min(
            unscheduled_jobs, key=lambda p: earliest_beginning(p, current_time)
        )
        unscheduled_pool: List[Process] = [
            p for p in unscheduled_jobs if p.arrival_time <= min_earliest_beginning
        ]
        min_prtt: float = min(unscheduled_pool, key=lambda p: PRTT(p, current_time))
        reduced_unscheduled_pool: List[Process] = [
            p for p in unscheduled_pool if PRTT(p, current_time) <= min_prtt
        ]
        min_burst_process: Process = min(
            reduced_unscheduled_pool, key=lambda p: p.burst_time
        )

        # Step 3:
        scheduled_jobs.append(min_burst_process)
        unscheduled_jobs.remove(min_burst_process)
        current_time = (
            earliest_beginning(min_burst_process, current_time)
            + min_burst_process.burst_time
        )

    return scheduled_jobs


def branch_and_bound_algorithm(processes: List[Process]):
    # Step 1: Get the initial sequence from the best of the Greedy algorithms
    # defined above.
    # P <- List of processes scheduled from the beginning.
    # Q <- List of the processes scheduled from the end.
    # TODO: get min latency from the above two algorithms
    init_sequence: List[Process] = []
    P: List[Process] = []
    Q: List[Process] = []

    # Q is empty if there is at-least an unscheduled job whose arrival time is
    # strictly greater than the completion time of the sequence P.

    # For each node, fictitious arrival times of unscheduled jobs are used to
    # replace the original arrival times in the dominance theorems
    # The fictitious arrival time of a process i is the max of it's original
    # arrival time and the completion time of the last job of P.

    # When Q is empty:
    # Each descendant node is obtained by adding behind the
    # partial sequence P, a new process i, chosen among the unscheduled jobs
    # The choice of a job i is such that
    # (i) Not preceeded by another unscheduled jobs according to Theorems 2 and 4
    # (ii) It is not dominated by another unscheduled job according to Theorems
    #      3 and 5
    # (iii) job i and the last job of P verify the conditions of a T-active schedule

    # If the arrival time of all unscheduled jobs are smaller than the
    # completion time of P, each descendent node is obtained by adding a new job
    # i, from the unscheduled jobs, before the partial sequence Q

    # NOTE: When Q is empty, the completion time of job i is computed by adding
    # the sum of processing times of the unscheduled jobs to the completion time
    # of partial sequence P.

    # Choice of job i is such that there is no unscheduled job j
    # (i) j.deadline >= i.deadline and j.burst_time >= i.burst_time (at least
    # one of these inequalities is strict) or,
    # (ii) j.burst_time < i.burst_time and j.deadline > i.deadline and
    # j.deadline + j.burst_time >= delta, where delta = start_time of the first
    # job of Q.
    # (iii) The job i and first job of Q should also verify the conditions for a
    # schedule to be T-active

    # The lower bound is computed using Theorem 6. An upper bound is computed
    # for each descendant node by completing the partial sequences with the
    # heuristic NDPRTT.
    # If the solution found is better than the previous best, it is retained as
    # the new best.
    # A descendant node is eliminated if its lower bound is greater than or
    # equal to the total tardiness of the best solution found so far.
