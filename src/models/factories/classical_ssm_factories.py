import torch
from typing import List, Callable, Any

from src.algorithms.ssm_init1D import get_diag_ssm_plus_noise_init, \
    get_rot_ssm_one_over_n_init, get_rot_ssm_equally_spaced_init, get_hippo_disc_init
from src.models.ssm import SMMModel
import src.models.strategies.storing as storing_strat
import src.models.strategies.ssm_init as init_strat
import src.models.strategies.parametrization as param_strat
import src.models.strategies.calc as calc_strat


def _get_distinct_init_func_from_init_func(init_func):
    A_init_f = lambda n: init_func(n)[0]
    B_init_f = lambda n: init_func(n)[1]
    C_init_f = lambda n: init_func(n)[2]
    D_init_f = lambda n: init_func(n)[3]
    return A_init_f, B_init_f, C_init_f, D_init_f


def _get_classical_ssm(init_func: Callable[[int], Any],
                       num_hidden_state: int,
                       trainable_param_list: List[str],
                       device: torch.device = None):
    if device is None:
        device = torch.device("cpu")

    A_init_f, B_init_f, C_init_f, D_init_f = \
        _get_distinct_init_func_from_init_func(init_func)

    return SMMModel(
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
        device=device
    )


def get_full_ssm_diag_plus_noise1D(num_hidden_state: int,
                                   A_diag: float = 0.9,
                                   A_noise_std: float = 0.001,
                                   B_init_std: float = 1e-1,
                                   C_init_std: float = 1e-1,
                                   trainable_param_list: List[str] = ("A", "B", "C"),
                                   device: torch.device = None) -> SMMModel:
    init_func = lambda n: get_diag_ssm_plus_noise_init(num_hidden_state=n,
                                                       A_diag=A_diag,
                                                       A_noise_std=A_noise_std,
                                                       B_init_std=B_init_std,
                                                       C_init_std=C_init_std)
    return _get_classical_ssm(init_func=init_func,
                              num_hidden_state=num_hidden_state,
                              device=device,
                              trainable_param_list=trainable_param_list)


def get_full_ssm_hippo1D(num_hidden_state: int,
                         C_init_std: int = 1,
                         dt: float = 0.01,
                         trainable_param_list: List[str] = ("A", "B", "C"),
                         device: torch.device = None) -> SMMModel:
    init_func = lambda n: get_hippo_disc_init(num_hidden_state=n,
                                              C_init_std=C_init_std,
                                              dt=dt)

    return _get_classical_ssm(init_func=init_func,
                              num_hidden_state=num_hidden_state,
                              device=device,
                              trainable_param_list=trainable_param_list)


def get_rot_ssm_equally_spaced(num_hidden_state: int,
                               radii: float = 0.99,
                               B_init_std: float = 2e-1,
                               C_init_std: float = 2e-1,
                               trainable_param_list: List[str] = ("A", "B", "C"),
                               device: torch.device = None) -> SMMModel:
    init_func = lambda n: get_rot_ssm_equally_spaced_init(num_hidden_state=n,
                                                          radii=radii,
                                                          B_init_std=B_init_std,
                                                          C_init_std=C_init_std,
                                                          angle_shift=2 ** 0.5)

    return _get_classical_ssm(init_func=init_func,
                              num_hidden_state=num_hidden_state,
                              device=device,
                              trainable_param_list=trainable_param_list)


def get_rot_ssm_one_over_n(num_hidden_state: int,
                           radii: float = 0.99,
                           B_init_std: float = 1e-1,
                           C_init_std: float = 1e-1,
                           trainable_param_list: List[str] = ("A", "B", "C"),
                           device: torch.device = None) -> SMMModel:
    init_func = lambda n: get_rot_ssm_one_over_n_init(num_hidden_state=n,
                                                      radii=radii,
                                                      B_init_std=B_init_std,
                                                      C_init_std=C_init_std,
                                                      angle_shift=2 ** 0.5)

    return _get_classical_ssm(init_func=init_func,
                              num_hidden_state=num_hidden_state,
                              device=device,
                              trainable_param_list=trainable_param_list)
