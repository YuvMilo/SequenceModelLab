import os


def test_EntityTrainingHistory_getitem():
    from src.logging.training_logs.base_log import EntityTrainingHistory

    e = EntityTrainingHistory()
    e.add_entity(10, 100)
    assert e[10] == 100


def test_BaseTrainingLog_logged_training_entities():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_training_entity("A", 0, 1)
    log.log_training_entity("A", 5, 7)
    log.log_training_entity("A", 2, 5)
    log.log_training_entity("B", 2, -5)

    assert len(log.logged_training_entities) == 2
    assert set(log.logged_training_entities) == {"A", "B"}


def test_BaseTrainingLog_saving_trained_params():
    from src.logging.training_logs.base_log import BaseTrainingLog

    d = {"A": 5}
    log = BaseTrainingLog(d)
    d["A"] = 6

    log.save("tmp")
    log = BaseTrainingLog()
    log.load("tmp")
    os.remove("tmp")  # TODO - There's probably a safer way to deal with tmp files.

    assert log.running_params["A"] == 5


def test_BaseTrainingLog_get_logged_entity_history():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_training_entity("A", 0, 1)
    log.log_training_entity("A", 5, 7)
    log.log_training_entity("A", 2, 5)

    entity_history = log.get_logged_entity_history(entity_name="A")

    # results should be sorted by epochs
    assert entity_history.entities == [1, 5, 7]
    assert entity_history.epochs == [0, 2, 5]


def test_BaseTrainingLog_history_augmentation():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_training_entity("A", 0, 1)
    log.log_training_entity("A", 5, 7)
    log.log_training_entity("A", 2, 5)
    log.log_training_entity("B", 2, -5)
    log.log_training_entity("B", 5, -6)
    log.log_training_entity("B", 10, -7)

    def augmentation_func(A, B) -> int:
        return A+B

    log.add_entity_history_by_augmentation(entity_name="C",
                                           parameters=["A", "B"],
                                           augmentation_func=augmentation_func)

    entity_history = log.get_logged_entity_history(entity_name="C")

    # Entity should be logged only where the two parameters are available
    assert entity_history.epochs == [2, 5]
    assert entity_history.entities == [0, 1]

    def augmentation_func(A, B) -> int:
        return A+B+2

    log.add_entity_history_by_augmentation(entity_name="C",
                                           parameters=["A", "B"],
                                           augmentation_func=augmentation_func,
                                           overwrite=False)
    entity_history = log.get_logged_entity_history(entity_name="C")
    assert entity_history.epochs == [2, 5]
    assert entity_history.entities == [0, 1]
    log.add_entity_history_by_augmentation(entity_name="C",
                                           parameters=["A", "B"],
                                           augmentation_func=augmentation_func,
                                           overwrite=True)
    entity_history = log.get_logged_entity_history(entity_name="C")
    assert entity_history.epochs == [2, 5]
    assert entity_history.entities == [2, 3]

def test_BaseTrainingLog_end_result_augmentation():
    import numpy as np
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_training_entity("A", 1, 1)
    log.log_training_entity("A", 2, 2)
    log.log_training_entity("A", 3, 3)
    log.log_training_entity("B", 2, -2)
    log.log_training_entity("B", 3, -3)
    log.log_training_entity("B", 4, -4)

    def augmentation_func(epochs, As, Bs) -> int:
        return min(np.min(As), np.min(Bs))

    log.add_entity_end_result_by_augmentation(entity_name="min",
                                              parameters=["A", "B"],
                                              augmentation_func=augmentation_func)

    end_result = log.get_end_result(end_result_name="min")

    # Entity should be calculated over when all the params are avalable
    assert end_result == -3

    def augmentation_func(epochs, As, Bs) -> int:
        return min(np.min(As), np.min(Bs))+3

    log.add_entity_end_result_by_augmentation(entity_name="min",
                                              parameters=["A", "B"],
                                              augmentation_func=augmentation_func,
                                              overwrite=False)
    end_result = log.get_end_result(end_result_name="min")
    assert end_result == -3

    log.add_entity_end_result_by_augmentation(entity_name="min",
                                              parameters=["A", "B"],
                                              augmentation_func=augmentation_func,
                                              overwrite=True)
    end_result = log.get_end_result(end_result_name="min")
    assert end_result == 0


def test_BaseTrainingLog_logged_end_results():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_end_result("A", 0)
    log.log_end_result("A", 1)
    log.log_end_result("B", 1)

    assert len(log.logged_end_results) == 2
    assert set(log.logged_end_results) == {"A", "B"}


def test_BaseTrainingLog_get_end_result():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_end_result("A", 0)
    log.log_end_result("A", 1)

    assert log.get_end_result("A") == 1


def test_BaseTrainingLog_bool():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_training_entity("A", 1, 0)
    log.log_training_entity("A", 2, 1)
    assert log

    log = BaseTrainingLog()
    log.log_training_entity("A", 1, 0)
    assert not log

    log = BaseTrainingLog()
    assert not log


def test_BaseTrainingLog_save_load():
    from src.logging.training_logs.base_log import BaseTrainingLog

    log = BaseTrainingLog()
    log.log_training_entity("A", 0, 1)
    log.log_training_entity("A", 5, 7)
    log.log_training_entity("A", 2, 5)
    log.log_training_entity("B", 2, -5)
    log.log_end_result("C", 0)
    log.log_end_result("C", 1)
    log.log_end_result("D", 1)

    log.save("tmp")
    log = BaseTrainingLog()
    log.load("tmp")
    entity_history = log.get_logged_entity_history(entity_name="A")
    os.remove("tmp")  # TODO - There's probably a safer way to deal with tmp files.

    assert len(log.logged_training_entities) == 2
    assert set(log.logged_training_entities) == {"A", "B"}
    assert entity_history.entities == [1, 5, 7]
    assert entity_history.epochs == [0, 2, 5]
    assert len(log.logged_end_results) == 2
    assert set(log.logged_end_results) == {"C", "D"} and len(log.logged_end_results) == 2


def test_SSMTrainingLogger():
    import torch
    import numpy as np
    from src.logging.training_logger.ssm_logger import SSMTrainingLogger
    from src.models.ssm import SMMModel
    import src.models.strategies.storing as storing_strat
    import src.models.strategies.ssm_init as init_strat
    import src.models.strategies.parametrization as param_strat
    import src.models.strategies.calc as calc_strat
    from src.logging.training_logs.base_log import BaseTrainingLog

    param_storing_freq = 2
    saving_freq = 10
    logger = SSMTrainingLogger(saving_path="tmp", param_storing_freq=param_storing_freq,
                               running_params={"lr": 0.001})

    # TODO - Use the output of a factory instead.
    dummy_SSM_model = SMMModel(
        num_hidden_state=1,
        input_dim=1,
        output_dim=1,
        ssm_param_strategy=param_strat.DiscreteSMMParametrizationStrategy(
            ssm_init_strategy=init_strat.FlexibleSSMInitStrategy(
                A_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
                B_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
                C_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
                D_init_func=lambda n: torch.zeros([1, 1], dtype=torch.float),
            ),
            ssm_storing_strategy=storing_strat.RealArrayStoringStrategy(),
        ),
        ssm_calc_strategy=calc_strat.RecurrentSMMCalcStrategy(),
        trainable_param_list=["A", "B", "C"],
        device=torch.device('cpu')
    )

    losses = []
    should_log_param_epochs = []
    for i in range(saving_freq*5+1):
        loss = 1/(i+1)
        logger.log(loss=loss, epoch_num=i, model=dummy_SSM_model)
        losses.append(loss)
        if i % param_storing_freq == 0:
            should_log_param_epochs.append(i)

    assert logger.get_loss_hist() == losses
    assert logger.training_log.get_logged_entity_history(entity_name="A").epochs == should_log_param_epochs

    logger.save()

    log = BaseTrainingLog()
    log.load("tmp")
    os.remove("tmp")  # TODO - There's probably a safer way to deal with tmp files.

    # Making
    assert np.max(log.get_logged_entity_history(entity_name="A").epochs) % saving_freq == 0
    assert log.running_params["lr"] == 0.001
