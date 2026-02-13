from clicker_optimizer import (
    BUILDINGS,
    Building,
    ceil_div,
    round_half_up,
    solve_optimal,
)


def test_round_half_up():
    assert round_half_up(10.4) == 10
    assert round_half_up(10.5) == 11
    assert round_half_up(11.5) == 12


def test_solution_reaches_target():
    path, finish_time, finish_balance = solve_optimal(target_balance=1_000_000)
    assert finish_time > 0
    assert finish_balance >= 1_000_000
    assert path[-1].time <= finish_time


def test_small_scenario_prefers_upgrade():
    buildings = (
        Building(base_cost=10, income=2, growth=0.5),
        Building(base_cost=200, income=20, growth=0.1),
        Building(base_cost=500, income=30, growth=0.1),
    )
    path, finish_time, finish_balance = solve_optimal(
        target_balance=200,
        start_balance=0,
        start_cps=1,
        buildings=buildings,
    )
    assert finish_balance >= 200
    assert finish_time < 200
    assert any(node.action == 1 for node in path[1:])


def test_ceil_div():
    assert ceil_div(0, 5) == 0
    assert ceil_div(1, 5) == 1
    assert ceil_div(10, 5) == 2
    assert ceil_div(11, 5) == 3


def test_monotonic_step_times():
    path, _, _ = solve_optimal(target_balance=10_000)
    times = [node.time for node in path]
    assert times == sorted(times)
