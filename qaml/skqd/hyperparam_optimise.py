import logging

from skopt import gp_minimize
from skopt.utils import use_named_args
from qaml.skqd import SKQDRunner
from typing import List, Dict, Any
from skopt.space import Real, Integer
import optuna
from scipy.optimize import OptimizeResult
from optuna.study import Study

logging.basicConfig()
logger = logging.getLogger(__name__)

def optimize_skqd_gp(
    search_space: List[Real | Integer],
    fixed_params: Dict[str, Any],
    gp_minimize_kwargs: Dict[str, Any],
) -> OptimizeResult:
    """Optimize the SKQD parameters, e.g. krylov dimension, number of trotter steps, the evolution time step with Bayesian optimization using Gaussian Processes (GP).
    Legacy library: scikit-optimize

    Args:
        search_space (List[Real | Integer]): Search space for the parameters to be optimized.
        fixed_params (Dict[str, Any]): Fixed parameters to run the SKQD.
        gp_minimize_kwargs (Dict[str, Any]): Hyper-parameters for the gp_minimize.

    Returns:
        OptimizeResult: Result of optimization.
    """

    # Objective function that wraps SKQDRunner
    @use_named_args(search_space)
    def objective(**opt_params):
        # Merge fixed and optimized parameters
        runner_params = {**fixed_params, **opt_params}
        runner = SKQDRunner(**runner_params)
        logger.info(f"Running with {runner.sampler.shots} shots for optimization of dt")
        logger.info(f"Evaluating SKQD with params: {opt_params}")
        gs_en, _, _ = runner.run()
        logger.info(f"GS energy: {gs_en}")
        return gs_en

    # Run the optimizer
    result = gp_minimize(func=objective, dimensions=search_space, **gp_minimize_kwargs)

    return result


def optimize_skqd_tpe(
    search_params: Dict[str, Dict[str, List[float]|str]],
    fixed_params: Dict[str, Any],
    optuna_kwargs: Dict[str, Any],
) -> Study:
    """Optimize the SKQD parameters, e.g. krylov dimension, number of trotter steps, the evolution time step with Tree-structured Parzen Estimator (TPE).
    Legacy library: Optuna

    Args:
        search_params (Dict[str, Dict[str, list[float] | str]]): Search parameters for the parameters to be optimized.
        fixed_params (Dict[str, Any]): Fixed parameters to run the SKQD.
        optuna_kwargs (Dict[str, Any]): Hyper-parameters for the optuna study.

    Returns:
        Study: Result of optimization.
    """

    # Objective function that wraps SKQDRunner
    def objective(trial):
        suggest_func = {
            "float": trial.suggest_float,
            "int": trial.suggest_int,
            "categorical": trial.suggest_categorical,
        }

        # Convert search_params to trial.suggest_*
        search_params_trial = {}
        for k, v in search_params.items():
            search_params_trial[k] = suggest_func[v["dtype"]](k, *v["bounds"])

        # Combine fixed and suggested parameters
        runner_params = {**fixed_params, **search_params_trial}

        runner = SKQDRunner(**runner_params)
        gs_en, _ = runner.run()

        return gs_en

    # Create a study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, **optuna_kwargs)  # Run up to 50 trials or 5 minutes

    return study
