import lkh
import numpy as np

from tensordict import TensorDict
from torch import Tensor
import subprocess
from .constants import LKH_SCALING_FACTOR, ROUTEFINDER2LKH
from .utils import scale
import tempfile

import os
from typing import List, Optional


def compute_action_cost(action, edge_weight_matrix, scaling_factor=1):
    total_cost = 0
    for i in range(len(action) - 1):
        total_cost += edge_weight_matrix[action[i]][action[i + 1]]
    return total_cost / scaling_factor


def solve_with_lkh(par_file="problem.par", lkh_exec="./LKH", log_file='log.txt'):
    with open(log_file, 'w') as log:
        result = subprocess.run(
            [lkh_exec, par_file],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3600,  # 1小时超时
            check=False
        )

    # 读取日志文件
    with open(log_file, 'r') as f:
        log_content = f.read()

    success = result.returncode == 0
    return 1


def create_lkh_par_file(problem_file: str, output_tour_file: str,
                        max_trials: int = 10000, runs: int = 1,
                        trace_level: int = 1, optimum: Optional[int] = None) -> str:
    """
    创建LKH参数文件

    参数:
        problem_file: 问题文件路径（.vrptw文件）
        output_tour_file: 输出路线文件路径（.tour文件）
        max_trials: 最大试验次数
        runs: 运行次数
        trace_level: 跟踪级别（0=无输出，1=正常，2=详细）
        optimum: 最优值（可选，用于验证）

    返回:
        参数文件路径
    """
    # 创建临时参数文件
    par_file = tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False)

    # 写入参数
    par_file.write("SPECIAL\n")
    par_file.write(f"PROBLEM_FILE = {os.path.abspath(problem_file)}\n")
    par_file.write(f"OUTPUT_TOUR_FILE = {os.path.abspath(output_tour_file)}\n")
    par_file.write(f"MAX_TRIALS = {max_trials}\n")
    par_file.write(f"RUNS = {runs}\n")
    par_file.write(f"TRACE_LEVEL = {trace_level}\n")

    if optimum is not None:
        par_file.write(f"OPTIMUM = {optimum}\n")

    par_file.close()

    return par_file.name


def solve(
        instance: TensorDict,
        max_runtime: float,
        problem_type: str,
        num_runs: int,
        solver_loc: str,
) -> tuple[Tensor, Tensor]:
    """
    Solves an AnyVRP instance with OR-Tools.

    Parameters
    ----------
    instance
        The AnyVRP instance to solve.
    max_runtime
        The maximum runtime for the solver.
    problem_type
        The problem type for LKH3.
    num_runs
        The number of runs to perform and returns the best result.
    solver_loc
        The location of the LKH3 solver executable.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    problem = instance2problem(instance, problem_type, LKH_SCALING_FACTOR)
    if 'TW' in problem_type:
        with open(
                "/common/home/users/s/scgui/Code/2025-LLMCO/cvrptw_benchmark/CVRPTW/INSTANCES/Solomon_50/C101.50.5.vrptw",
                "w") as f:
            f.write(problem)
        problem_dir = '/common/home/users/s/scgui/Code/2025-LLMCO/cvrptw_benchmark/CVRPTW/INSTANCES/Solomon_50/C101.50.5.vrptw'
        par_dir = "/common/home/users/s/scgui/Code/2025-LLMCO/0424-routefinder-LO-CaDA-ReLD-dotNoise/problem.par"
        output_tour_file = problem_dir.replace('.vrptw', '.tour')
        par_file = create_lkh_par_file(
            problem_dir, output_tour_file)

        # 运行LKH solver
        log_file = 'log.txt'
        verbose = True
        try:
            with open(log_file, 'w') as log:
                result = subprocess.run(
                    [solver_loc, par_file],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,  # 1小时超时
                    check=False
                )

            # 读取日志文件
            with open(log_file, 'r') as f:
                log_content = f.read()

            success = result.returncode == 0 and os.path.exists(output_tour_file)

            # 解析结果
            result_info = {
                'problem_file': problem_dir,
                'success': success,
                'returncode': result.returncode,
                'tour_file': output_tour_file if success else None,
                'log_file': log_file,
                'log_content': log_content
            }

            if success:
                # 尝试从日志中提取成本信息
                lines = log_content.split('\n')
                for line in lines:
                    if 'Cost' in line or 'cost' in line:
                        result_info['cost_line'] = line
                    if 'Time' in line or 'time' in line:
                        result_info['time_line'] = line

            if verbose:
                if success:
                    print(f"  ✓ 求解成功")
                else:
                    print(f"  ✗ 求解失败 (返回码: {result.returncode})")

        except subprocess.TimeoutExpired:
            if verbose:
                print(f"  ✗ 求解超时")
        except Exception as e:
            if verbose:
                print(f"  ✗ 求解出错: {e}")
        finally:
            # 清理临时参数文件
            if os.path.exists(par_file):
                os.remove(par_file)

    else:
        action, cost = _solve(problem, max_runtime, num_runs, solver_loc)
        cost = compute_action_cost(action, problem.edge_weights, scaling_factor=LKH_SCALING_FACTOR)

    return action, cost


def write_vrplib(filename, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None,
                 tw_end=None,
                 grid_size=1, scale=100000, name="Instance", problem="CVRP"):
    # scale = 100000  # EAS uses 1000, while AM uses 100000
    to_int = lambda x: int(x / grid_size * scale + 0.5)
    # size_vehicle_dict = {50: 50, 100: 100}  # hardcoded, only for DCVRP
    size_vehicle_dict = {50: 8, 100: 12}  # hardcoded, only for DCVRP

    with open(filename, 'w') as f:
        # 1. file head
        # Note: 'VEHICLES' cannot >= 'DIMENSION'
        #   a. for CVRP, no need to specifiy VEHICLES
        #   b. for other problems, need to specifiy VEHICLES, otherwise, VEHICLES=1 -> cannot find feasible solutions
        #      for DCVRP, the performance heavily depend on the number of VEHICLES
        if problem in ["CVRP"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("CAPACITY", int(capacity)),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        elif problem in ["OVRP", "VRPB"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("VEHICLES", len(loc)),
                    ("CAPACITY", int(capacity)),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        elif problem in ["CVRPTW", "VRPBTW"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("VEHICLES", len(loc)),
                    ("CAPACITY", int(capacity * 40)),
                    ("SERVICE_TIME", to_int(service_time[0])),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        elif problem in ["DCVRP"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("CAPACITY", int(capacity)),
                    ("DISTANCE", to_int(route_limit)),
                    # ("SERVICE_TIME", 0),
                    ("VEHICLES", size_vehicle_dict[len(loc)]),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        else:
            raise NotImplementedError

        # 2. coordinates
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, to_int(x), to_int(y))  # VRPlib does not take floats
            for i, (x, y) in enumerate([depot] + loc)
        ]))

        # 3. demand
        f.write("\n")
        backhauls = [i + 1 for i, d in enumerate(demand) if d < 0]
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, abs(int(d * 40)))
            # convert to int for lkh3, otherwise "DEMAND_SECTION: Node number out of range: 0"
            for i, d in enumerate([0] + demand)
        ]))

        # 4. optional: time window
        if problem in ["CVRPTW", "VRPBTW"]:
            f.write("\n")
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, to_int(e), to_int(l))
                for i, (e, l) in enumerate(zip([0] + tw_start, [4.6] + tw_end))  # hardcoded: tw for depot: [0., 3.]
            ]))

            # 3. demand
            f.write("\n")
            f.write("SERVICE_TIME_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, abs(to_int(d)))
                # convert to int for lkh3, otherwise "DEMAND_SECTION: Node number out of range: 0"
                for i, d in enumerate([0] + service_time)
            ]))

        # 5. optional: backhauls
        if len(backhauls) > 0:
            f.write("\n")
            f.write("BACKHAUL_SECTION\n")
            f.write("\t".join(["{}".format(b) for b in backhauls]))
            f.write("\t-1")

        # 6. file tail
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def _solve(
        problem: lkh.LKHProblem,
        max_runtime: float,
        num_runs: int,
        solver_loc: str,
) -> tuple[Tensor, Tensor]:
    """
    Solves an instance with LKH3.

    Parameters
    ----------
    problem
        The LKHProblem instance.
    max_runtime
        The maximum runtime for each solver run.
    num_runs
        The number of runs to perform and returns the best result.
        Note: Each run uses a different initial solution. LKH has difficulty
        finding feasible solutions, so performing more runs can help to find
        solutions that are feasible.
    solver_loc
        The location of the LKH3 solver executable.
    """
    routes = lkh.solve(
        solver_loc,
        problem=problem,
        time_limit=max_runtime,
        runs=num_runs,
    )

    action = routes2action(routes)
    cost = 1
    return action, cost


def instance2problem(
        instance: TensorDict,
        problem_type: str,
        scaling_factor,
) -> lkh.LKHProblem:
    """
    Converts an AnyVRP instance to an LKHProblem instance.

    Parameters
    ----------
    instance
        The AnyVRP instance to convert.
    problem_type
        The problem type for LKH3. See ``constants.ROUTEFINDER2LKH`` for
        supported problem types.
    scaling_factor
        The scaling factor to apply to the instance data.
    """
    num_locations = instance["demand_linehaul"].size()[0]
    lkh_problem_type = ROUTEFINDER2LKH[problem_type]

    # Data specifications
    specs = {}
    specs["NAME"] = num_locations
    if lkh_problem_type is None:
        raise ValueError(f"Problem type {problem_type} is not supported by LKH.")

    specs["COMMENT"] = lkh_problem_type + ' Instance'
    specs["TYPE"] = lkh_problem_type

    specs["DIMENSION"] = num_locations
    # Weird LKH quirk: specifying the number of vehicles lets (D)CVRP hang.
    if lkh_problem_type not in ["CVRP", "DCVRP"]:
        specs["VEHICLES"] = num_locations - 1

    specs["CAPACITY"] = scale(instance["vehicle_capacity"], 40)
    if 'L' in lkh_problem_type:

        if not np.isinf(distance_limit := instance["distance_limit"]).any():
            specs["DISTANCE"] = scale(distance_limit, scaling_factor)
    # if 'TW' in lkh_problem_type:
    #     service_times = scale(instance['service_time'], scaling_factor)
    #     specs["SERVICE_TIME_SECTION"] = min(service_times[1:])
    specs["SCALE"] = scaling_factor

    specs["EDGE_WEIGHT_TYPE"] = "EUC_2D"
    # specs["EDGE_WEIGHT_FORMAT"] = "FULL_MATRIX"
    # specs["NODE_COORD_TYPE"] = "TWOD_COORDS"

    # NAME: C102
    # TYPE: CVRPTW
    # DIMENSION: 51
    # VEHICLES: 5
    # CAPACITY: 200
    # SERVICE_TIME: 90
    # SCALE: 10
    # EDGE_WEIGHT_TYPE: FLOOR_2D

    # LKH can only solve VRP variants that are explicitly supported (so no
    # arbitrary combinations between individual supported features). We can
    # support some open variants with some modeling tricks.

    # Data sections
    sections = {}
    sections["NODE_COORD_SECTION"] = scale(instance["locs"], scaling_factor)

    demand_linehaul = scale(instance["demand_linehaul"], 40)
    demand_backhaul = scale(instance["demand_backhaul"], 40)
    sections["DEMAND_SECTION"] = demand_linehaul + demand_backhaul
    if 'TW' in lkh_problem_type:
        time_windows = scale(instance["time_windows"], scaling_factor)
        sections["TIME_WINDOW_SECTION"] = time_windows

        service_times = scale(instance["service_time"], scaling_factor)
        sections["SERVICE_TIME_SECTION"] = service_times

    distances = instance["cost_matrix"]
    if 'B' in lkh_problem_type:
        backhaul_class = instance["backhaul_class"]

        if backhaul_class == 1:
            # VRPB has a backhaul section that specifies the backhaul nodes.
            backhaul_idcs = np.flatnonzero(instance["demand_backhaul"]).tolist()
            sections["BACKHAUL_SECTION"] = backhaul_idcs + [-1]

            # linehaul = np.flatnonzero(demand_linehaul > 0)
            # backhaul = np.flatnonzero(demand_backhaul > 0)
            # distances[np.ix_(backhaul, linehaul)] = time_windows.max()

        elif backhaul_class == 2:
            # VRPMPD has a pickup and delivery section that specifies the pickup
            # and delivery quantities for each node, as well as the time windows.
            # The regular time window section is redundant in this case.
            data = [
                [
                    0,  # dummy
                    time_windows[idx][0],
                    time_windows[idx][1],
                    service_times[idx],
                    demand_backhaul[idx],
                    demand_linehaul[idx],
                ]
                for idx in range(num_locations)
            ]
            sections["PICKUP_AND_DELIVERY_SECTION"] = data

    if instance["open_route"]:
        # Arcs to the depot are set to zero as vehicles don’t need to return.
        distances[:, 0] = 0

    # sections["EDGE_WEIGHT_SECTION"] = scale(distances, scaling_factor)

    # Convert to VRPLIB-like string.
    problem = "\n".join(f"{k} : {v}" for k, v in specs.items())
    problem += "\n" + "\n".join(_format(name, data) for name, data in sections.items())
    problem += "\n" + "\n".join(["DEPOT_SECTION", "1", "-1", "EOF"])

    if 'TW' in lkh_problem_type:
        return problem
    else:
        return lkh.LKHProblem.parse(problem)


def _is_1D(data) -> bool:
    for elt in data:
        if isinstance(elt, (list, tuple, np.ndarray)):
            return False
    return True


def _format(name: str, data) -> str:
    """
    Formats a data section.

    Parameters
    ----------
    name
        The name of the section.
    data
        The data to be formatted.

    Returns
    -------
    str
        A VRPLIB-formatted data section.
    """
    section = [name]
    include_idx = name not in ["EDGE_WEIGHT_SECTION", "BACKHAUL_SECTION"]

    if name == "BACKHAUL_SECTION":
        # Treat backhaul section as row vector.
        section.append("\t".join(str(val) for val in data))

    elif _is_1D(data):
        # Treat 1D arrays as column vectors, so each element is a row.
        for idx, elt in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            section.append(prefix + str(elt))
    else:
        for idx, row in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            rest = "\t".join([str(elt) for elt in row])
            section.append(prefix + rest)

    return "\n".join(section)


def routes2action(routes: list[list[int]]) -> list[int]:
    """
    Converts LKH routes to an action.
    """
    # LKH routes are location-indexed, which in turn are 1-indexed. The first
    # location is always the depot, so we subtract 2 to get client indices.
    # LKH routes are 1-indexed, so we subtract 1 to get client indices.
    routes_ = [[client - 1 for client in route] for route in routes]
    return [visit for route in routes_ for visit in route + [0]]
