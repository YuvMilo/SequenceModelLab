import torch
from typing import List, Callable, Any

from src.algorithms.ssm_init1D import get_diag_ssm_plus_noise_init, \
    get_rot_ssm_one_over_n_init, get_rot_ssm_equally_spaced_init
from src.models.deep_ssm import DeepSMMModel
import src.models.strategies.storing as storing_strat
import src.models.strategies.ssm_init as init_strat
import src.models.strategies.parametrization as param_strat
import src.models.strategies.calc as calc_strat


def _get_distinct_init_func_from_init_func(init_func):
    A_init_f = lambda n, input_dim=1, output_dim=1: init_func(n,
                                                              input_dim=input_dim,
                                                              output_dim=output_dim)[0]
    B_init_f = lambda n, input_dim=1, output_dim=1: init_func(n,
                                                              input_dim=input_dim,
                                                              output_dim=output_dim
                                                              )[1]
    C_init_f = lambda n, input_dim=1, output_dim=1: init_func(n,
                                                              input_dim=input_dim,
                                                              output_dim=output_dim)[2]
    D_init_f = lambda n, input_dim=1, output_dim=1: init_func(n,
                                                              input_dim=input_dim,
                                                              output_dim=output_dim)[3]
    return A_init_f, B_init_f, C_init_f, D_init_f


def _get_deep_ssm(init_func: Callable[[int], Any],
                  num_hidden_state: int,
                  trainable_param_list: List[str],
                  depth,
                  non_linearity: Callable[[Any], Any] = lambda x: x,
                  device: torch.device = None) -> DeepSMMModel:
    if device is None:
        device = torch.device("cpu")

    A_init_f, B_init_f, C_init_f, D_init_f = \
        _get_distinct_init_func_from_init_func(init_func)

    return DeepSMMModel(
        num_hidden_state=num_hidden_state,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=A_init_f,
                B_init_func=B_init_f,
                C_init_func=C_init_f,
                D_init_func=D_init_f,
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=trainable_param_list,
        device=device,
        non_linearity=non_linearity,
        depth=depth
    )


def get_full_ssm_diag_plus_noise1D(num_hidden_state: int,
                                   depth: int,
                                   A_diag: float = 0.9,
                                   A_noise_std: float = 0.001,
                                   B_init_std: float = 1e-1,
                                   C_init_std: float = 1e-1,
                                   non_linearity: Callable[[Any], Any] = lambda x: x,
                                   trainable_param_list: List[str] = ("A", "B", "C"),
                                   device: torch.device = None) -> DeepSMMModel:
    init_func = lambda n, output_dim, input_dim: get_diag_ssm_plus_noise_init(
        num_hidden_state=n,
        A_diag=A_diag,
        A_noise_std=A_noise_std,
        B_init_std=B_init_std,
        C_init_std=C_init_std,
        input_dim=input_dim,
        output_dim=output_dim
    )
    return _get_deep_ssm(init_func=init_func,
                         num_hidden_state=num_hidden_state,
                         device=device,
                         trainable_param_list=trainable_param_list,
                         non_linearity=non_linearity,
                         depth=depth)


def get_rot_ssm_equally_spaced(num_hidden_state: int,
                               depth: int,
                               radii: float = 0.99,
                               B_init_std: float = 2e-1,
                               C_init_std: float = 2e-1,
                               main_diagonal_diff: float = 0,
                               off_diagonal_ratio: float = 1,
                               non_linearity: Callable[[Any], Any] = lambda x: x,
                               trainable_param_list: List[str] = ("A", "B", "C"),
                               device: torch.device = None) -> DeepSMMModel:
    init_func = lambda n, output_dim, input_dim: get_rot_ssm_equally_spaced_init(
        num_hidden_state=n,
        radii=radii,
        B_init_std=B_init_std,
        C_init_std=C_init_std,
        main_diagonal_diff=main_diagonal_diff,
        off_diagonal_ratio=off_diagonal_ratio,
        angle_shift=2 ** 0.5,
        input_dim=input_dim,
        output_dim=output_dim
    )

    return _get_deep_ssm(init_func=init_func,
                         num_hidden_state=num_hidden_state,
                         device=device,
                         trainable_param_list=trainable_param_list,
                         non_linearity=non_linearity,
                         depth=depth)


def get_rot_ssm_one_over_n(num_hidden_state: int,
                           depth: int,
                           radii: float = 0.99,
                           B_init_std: float = 1e-1,
                           C_init_std: float = 1e-1,
                           main_diagonal_diff: float = 0,
                           off_diagonal_ratio: float = 1,
                           non_linearity: Callable[[Any], Any] = lambda x: x,
                           trainable_param_list: List[str] = ("A", "B", "C"),
                           device: torch.device = None) -> DeepSMMModel:
    init_func = lambda n, output_dim, input_dim: get_rot_ssm_one_over_n_init(
        num_hidden_state=n,
        radii=radii,
        B_init_std=B_init_std,
        C_init_std=C_init_std,
        main_diagonal_diff=main_diagonal_diff,
        off_diagonal_ratio=off_diagonal_ratio,
        input_dim=input_dim,
        output_dim=output_dim
    )

    return _get_deep_ssm(init_func=init_func,
                         num_hidden_state=num_hidden_state,
                         device=device,
                         trainable_param_list=trainable_param_list,
                         non_linearity=non_linearity,
                         depth=depth)
