from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import floor
from typing import Dict, List, Optional, Tuple

TARGET_BALANCE = 1_000_000
START_BALANCE = 100
START_CPS = 1


@dataclass(frozen=True)
class Building:
    base_cost: int
    income: int
    growth: float


BUILDINGS: Tuple[Building, ...] = (
    Building(base_cost=10, income=1, growth=0.20),
    Building(base_cost=100, income=5, growth=0.15),
    Building(base_cost=250, income=10, growth=0.10),
)


@dataclass
class Node:
    node_id: int
    time: int
    balance: int
    counts: Tuple[int, int, int]
    cps: int
    parent_id: Optional[int]
    action: Optional[int]


def round_half_up(value: float) -> int:
    return int(floor(value + 0.5))


class CostCache:
    def __init__(self, buildings: Tuple[Building, ...]):
        self._tables: List[List[int]] = [[b.base_cost] for b in buildings]
        self._buildings = buildings

    def get(self, building_idx: int, owned_count: int) -> int:
        table = self._tables[building_idx]
        growth = self._buildings[building_idx].growth
        while len(table) <= owned_count:
            table.append(round_half_up(table[-1] * (1.0 + growth)))
        return table[owned_count]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def is_dominated(candidate_time: int, candidate_balance: int, frontier: List[Tuple[int, int, int]]) -> bool:
    for t, b, _ in frontier:
        if t <= candidate_time and b >= candidate_balance:
            return True
    return False


def prune_dominated(candidate_time: int, candidate_balance: int, frontier: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return [(t, b, node_id) for t, b, node_id in frontier if not (candidate_time <= t and candidate_balance >= b)]


def solve_optimal(
    target_balance: int = TARGET_BALANCE,
    start_balance: int = START_BALANCE,
    start_cps: int = START_CPS,
    buildings: Tuple[Building, ...] = BUILDINGS,
) -> Tuple[List[Node], int, int]:
    costs = CostCache(buildings)
    nodes: Dict[int, Node] = {}
    next_id = 0

    start = Node(
        node_id=next_id,
        time=0,
        balance=start_balance,
        counts=(0, 0, 0),
        cps=start_cps,
        parent_id=None,
        action=None,
    )
    nodes[next_id] = start

    frontier: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {start.counts: [(0, start.balance, start.node_id)]}
    heap: List[Tuple[int, int, int]] = [(start.time, -start.balance, start.node_id)]

    best_finish_time = ceil_div(max(0, target_balance - start.balance), start.cps)
    best_finish_node = start.node_id
    best_finish_balance = start.balance + best_finish_time * start.cps

    while heap:
        _, _, node_id = heappop(heap)
        node = nodes[node_id]
        key = node.counts
        if is_dominated(node.time, node.balance, [entry for entry in frontier.get(key, []) if entry[2] != node_id]):
            continue

        if node.balance >= target_balance:
            if node.time < best_finish_time:
                best_finish_time = node.time
                best_finish_node = node.node_id
                best_finish_balance = node.balance
            continue

        wait_to_goal = ceil_div(target_balance - node.balance, node.cps)
        finish_time = node.time + wait_to_goal
        finish_balance = node.balance + wait_to_goal * node.cps
        if finish_time < best_finish_time:
            best_finish_time = finish_time
            best_finish_node = node.node_id
            best_finish_balance = finish_balance

        if node.time >= best_finish_time:
            continue

        for idx, building in enumerate(buildings):
            cost = costs.get(idx, node.counts[idx])
            if node.balance >= cost:
                dt = 0
            else:
                dt = ceil_div(cost - node.balance, node.cps)

            new_time = node.time + dt
            if new_time > best_finish_time:
                continue

            new_balance = node.balance + dt * node.cps - cost
            new_counts = list(node.counts)
            new_counts[idx] += 1
            new_counts_t = tuple(new_counts)
            new_cps = node.cps + building.income

            state_frontier = frontier.setdefault(new_counts_t, [])
            if is_dominated(new_time, new_balance, state_frontier):
                continue

            next_id += 1
            candidate = Node(
                node_id=next_id,
                time=new_time,
                balance=new_balance,
                counts=new_counts_t,
                cps=new_cps,
                parent_id=node.node_id,
                action=idx + 1,
            )
            nodes[candidate.node_id] = candidate
            frontier[new_counts_t] = prune_dominated(new_time, new_balance, state_frontier)
            frontier[new_counts_t].append((new_time, new_balance, candidate.node_id))
            heappush(heap, (candidate.time, -candidate.balance, candidate.node_id))

    path: List[Node] = []
    cur: Optional[int] = best_finish_node
    while cur is not None:
        n = nodes[cur]
        path.append(n)
        cur = n.parent_id
    path.reverse()

    return path, best_finish_time, best_finish_balance


def print_solution(path: List[Node], finish_time: int, finish_balance: int, target_balance: int = TARGET_BALANCE) -> None:
    print("Шаг | Время (сек) | Покупка | Доход/сек | Баланс | Постройки [1,2,3]")
    step = 0
    for node in path[1:]:
        step += 1
        print(
            f"{step} | {node.time} | {node.action} | {node.cps} | {node.balance} | "
            f"[{node.counts[0]}, {node.counts[1]}, {node.counts[2]}]"
        )
        if step == 100:
            try:
                input("Пауза: достигнут 100-й шаг. Нажмите Enter, чтобы продолжить...")
            except EOFError:
                print("STDIN недоступен, продолжаем без ожидания.")

    print(
        f"Финал | {finish_time} | GOAL | {path[-1].cps} | {finish_balance} | "
        f"[{path[-1].counts[0]}, {path[-1].counts[1]}, {path[-1].counts[2]}]"
    )
    print(f"Цель {target_balance} достигнута за {finish_time} секунд.")


def main() -> None:
    path, finish_time, finish_balance = solve_optimal()
    print_solution(path, finish_time, finish_balance)


if __name__ == "__main__":
    main()
