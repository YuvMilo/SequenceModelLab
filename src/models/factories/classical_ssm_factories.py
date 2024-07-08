import torch
from typing import List, Callable, Any

from src.algorithms.ssm_init1D import get_diag_ssm_plus_noise_init, \
    get_rot_ssm_one_over_n_init, get_rot_ssm_equally_spaced_init, get_hippo_disc_init
from src.models.ssm import SMMModel
from src.models.strategies.base import BaseSSMStoringStrategy
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


def get_flexible_ssm(init_func: Callable[[int], Any],
                     num_hidden_state: int,
                     trainable_param_list: List[str] = ("A", "B", "C"),
                     non_linearity: Callable[[Any], Any] = lambda x: x,
                     device: torch.device = None,
                     ssm_storing_strategy: BaseSSMStoringStrategy = None):
    if device is None:
        device = torch.device("cpu")

    if ssm_storing_strategy is None:
        ssm_storing_strategy = storing_strat.RealArrayStoringStrategy

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
            ssm_storing_strategy=ssm_storing_strategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=trainable_param_list,
        device=device,
        non_linearity=non_linearity
    )

def get_flexible_diag_ssm(init_func: Callable[[int], Any],
                                     num_hidden_state: int,
                                     trainable_param_list: List[str] = ("A", "B", "C"),
                                     non_linearity: Callable[[Any], Any] = lambda x: x,
                                     device: torch.device = None,
                                     is_complex: bool = True,
                                     is_polar: bool = False,
                                     is_stable: bool = True):
    if device is None:
        device = torch.device("cpu")

    A_init_f, B_init_f, C_init_f, D_init_f = \
        _get_distinct_init_func_from_init_func(init_func)

    if is_complex:
        if not is_polar:
            if is_stable:
                ssm_storing_strategy = storing_strat.ComplexAs2DRealArrayStoringStrategyBoundedByOne()
            else:
                ssm_storing_strategy = storing_strat.ComplexAs2DRealArrayStoringStrategy()
        else:
            if is_stable:
                ssm_storing_strategy = storing_strat.ComplexAsPolarBoundedByOne()
            else:
                raise
    else:
        if is_stable:
            ssm_storing_strategy = storing_strat.RealArrayStoringStrategyBoudedByOne()
        else:
            ssm_storing_strategy = storing_strat.RealArrayStoringStrategy()

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
            ssm_storing_strategy=ssm_storing_strategy,
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=trainable_param_list,
        device=device,
        non_linearity=non_linearity
    )


def get_band_ssm(
        init_func: Callable[[int], Any],
        num_hidden_state: int,
        trainable_param_list: List[str] = ("A", "B", "C"),
        non_linearity: Callable[[Any], Any] = lambda x: x,
        device: torch.device = None,
):

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
            ssm_storing_strategy=storing_strat.BandComplexStoring(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=trainable_param_list,
        device=device,
        non_linearity=non_linearity
    )

def get_flexible_complex_diag_ssm(init_func: Callable[[int], Any],
                                  num_hidden_state: int,
                                  trainable_param_list: List[str] = ("A", "B", "C"),
                                  non_linearity: Callable[[Any], Any] = lambda x: x,
                                  device: torch.device = None,
                                  only_A_is_complex: bool = False):
    if device is None:
        device = torch.device("cpu")

    A_init_f, B_init_f, C_init_f, D_init_f = \
        _get_distinct_init_func_from_init_func(init_func)

    if only_A_is_complex:
        ssm_storing_strategy = storing_strat.ComplexAs2DRealArrayStoringStrategyOnlyA()
    else:
        ssm_storing_strategy = storing_strat.ComplexAs2DRealArrayStoringStrategy()

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
            ssm_storing_strategy=ssm_storing_strategy,
        ),
        ssm_calc_strategy=calc_strat.RecurrentDiagSMMCalcStrategy(),
        trainable_param_list=trainable_param_list,
        device=device,
        non_linearity=non_linearity
    )


def get_full_ssm_diag_plus_noise1D(num_hidden_state: int,
                                   A_diag: float = 0.9,
                                   A_noise_std: float = 0.001,
                                   B_init_std: float = 1e-1,
                                   C_init_std: float = 1e-1,
                                   non_linearity: Callable[[Any], Any] = lambda x: x,
                                   trainable_param_list: List[str] = ("A", "B", "C"),
                                   device: torch.device = None) -> SMMModel:
    init_func = lambda n, output_dim, input_dim: get_diag_ssm_plus_noise_init(
        num_hidden_state=n,
        A_diag=A_diag,
        A_noise_std=A_noise_std,
        B_init_std=B_init_std,
        C_init_std=C_init_std,
        input_dim=input_dim,
        output_dim=output_dim
    )
    return get_flexible_ssm(init_func=init_func,
                            num_hidden_state=num_hidden_state,
                            device=device,
                            trainable_param_list=trainable_param_list,
                            non_linearity=non_linearity)


def get_full_ssm_hippo1D(num_hidden_state: int,
                         C_init_std: int = 1,
                         dt: float = 0.01,
                         non_linearity: Callable[[Any], Any] = lambda x: x,
                         trainable_param_list: List[str] = ("A", "B", "C"),
                         device: torch.device = None) -> SMMModel:
    init_func = lambda n: get_hippo_disc_init(num_hidden_state=n,
                                              C_init_std=C_init_std,
                                              dt=dt)

    return get_flexible_ssm(init_func=init_func,
                            num_hidden_state=num_hidden_state,
                            device=device,
                            trainable_param_list=trainable_param_list,
                            non_linearity=non_linearity)


def get_rot_ssm_equally_spaced(num_hidden_state: int,
                               radii: float = 0.99,
                               B_init_std: float = 2e-1,
                               C_init_std: float = 2e-1,
                               main_diagonal_diff: float = 0,
                               off_diagonal_ratio: float = 1,
                               non_linearity: Callable[[Any], Any] = lambda x: x,
                               trainable_param_list: List[str] = ("A", "B", "C"),
                               device: torch.device = None,
                               block_diag: bool = False) -> SMMModel:
    init_func = lambda n, input_dim, output_dim: get_rot_ssm_equally_spaced_init(
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

    ssm_storing_strategy = None if not block_diag else\
        storing_strat.BlockDiagStoringStrategy

    return get_flexible_ssm(init_func=init_func,
                            num_hidden_state=num_hidden_state,
                            device=device,
                            trainable_param_list=trainable_param_list,
                            non_linearity=non_linearity,
                            ssm_storing_strategy=ssm_storing_strategy)


def get_rot_ssm_one_over_n(num_hidden_state: int,
                           radii: float = 0.99,
                           B_init_std: float = 1e-1,
                           C_init_std: float = 1e-1,
                           main_diagonal_diff: float = 0,
                           off_diagonal_ratio: float = 1,
                           non_linearity: Callable[[Any], Any] = lambda x: x,
                           trainable_param_list: List[str] = ("A", "B", "C"),
                           device: torch.device = None) -> SMMModel:
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

    return get_flexible_ssm(init_func=init_func,
                            num_hidden_state=num_hidden_state,
                            device=device,
                            trainable_param_list=trainable_param_list,
                            non_linearity=non_linearity)
