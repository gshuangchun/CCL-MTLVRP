from functools import partial
from multiprocessing import Pool

from tensordict.tensordict import TensorDict
from torch import Tensor


class NoSolver:
    def solve(self, *args, **kwargs):
        pass


try:
    import routefinder.baselines.pyvrp as pyvrp
except ImportError:
    pyvrp = NoSolver()
try:
    import routefinder.baselines.lkh as lkh
except ImportError:
    lkh = NoSolver()
try:
    import routefinder.baselines.ortools as ortools
except ImportError:
    ortools = NoSolver()

from .utils import mtvrp2anyvrp


def solve(
        instances: TensorDict,
        max_runtime: float,
        num_procs: int = 1,
        data_type: str = "mtvrp",
        solver: str = "pyvrp",
        **kwargs,
) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instances with PyVRP.

    Parameters
    ----------
    instances
        TensorDict containing the AnyVRP instances to solve.
    max_runtime
        Maximum runtime for the solver.
    num_procs
        Number of processers to use to solve instances in parallel.
    data_type
        Environment mode. If "mtvrp", the instance data will be converted first.
    solver
        The solver to use. One of ["pyvrp", "ortools", "lkh"].

    Returns
    -------
    tuple[Tensor, Tensor]
        A Tensor containing the actions for each instance and a Tensor
        containing the corresponding costs.
    """
    if data_type == "mtvrp":
        instances = mtvrp2anyvrp(instances)

    if solver == "pyvrp" and isinstance(pyvrp, NoSolver):
        raise ImportError("PyVRP is not installed. Please install it using `pip install -e .[solvers]`.")
    if solver == "lkh" and isinstance(lkh, NoSolver):
        raise ImportError("LKH is not installed. Please install it using `pip install -e .[solvers]`")
    if solver == "ortools" and isinstance(ortools, NoSolver):
        raise ImportError("OR-Tools is not installed. Please install it using `pip install -e .[solvers]`.")

    solvers = {"pyvrp": pyvrp.solve, "ortools": ortools.solve, "lkh": lkh.solve}
    if solver not in solvers:
        raise ValueError(f"Unknown baseline solver: {solver}")

    _solve = solvers[solver]
    func = partial(_solve, max_runtime=max_runtime, **kwargs)

    # 将 TensorDict 拆分为单个实例列表
    # TensorDict 可以通过索引访问，也可以通过迭代访问
    # 为了确保多进程正常工作，我们显式地创建实例列表
    # 使用 unbind 方法或按索引访问来拆分
    try:
        # 尝试使用 unbind 方法（如果可用）
        if hasattr(instances, 'unbind'):
            instance_list = list(instances.unbind(0))
        elif hasattr(instances, 'batch_size') and len(instances.batch_size) > 0:
            # 如果 TensorDict 有 batch 维度，按索引拆分
            instance_list = [instances[i].clone() for i in range(instances.batch_size[0])]
        else:
            # 否则尝试直接迭代
            instance_list = [inst.clone() if hasattr(inst, 'clone') else inst for inst in instances]
    except (AttributeError, TypeError, IndexError):
        # 如果上述方法都失败，尝试直接迭代
        instance_list = list(instances) if hasattr(instances, '__iter__') else [instances]

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            results = pool.map(func, instance_list)
    else:
        results = [func(instance) for instance in instance_list]

    actions, costs = zip(*results)

    # Pad to ensure all actions have the same length.
    max_len = max(len(action) for action in actions)
    actions = [action + [0] * (max_len - len(action)) for action in actions]

    return Tensor(actions).long(), Tensor(costs)
