import os
import ray
import glob
from typing import List

from src.logging.training_logs.log_augmentation.augmentations import \
    hankel_augmentation, eig_augmentation, svd_augmentation, \
    effective_min_loss_augmentation, effective_train_len_augmentation
from src.logging.training_logs.base_log import BaseTrainingLog
from src.multiprocessing.utils import ProgressBar, ProgressBarActor


@ray.remote(num_cpus=60)
def save_all_augmentation(log_file_path: str,
                          progress_bar_actor: ProgressBarActor,
                          overwrite: bool = True):
    log = BaseTrainingLog()
    log.load(log_file_path)

    log.add_entity_end_result_by_augmentation(augmentation_func=effective_train_len_augmentation,
                                              entity_name="effective_train_len",
                                              parameters=["loss"],
                                              overwrite=overwrite)
    log.add_entity_end_result_by_augmentation(augmentation_func=effective_min_loss_augmentation,
                                              entity_name="min_loss",
                                              parameters=["loss"],
                                              overwrite=overwrite)
    log.add_entity_history_by_augmentation(
        entity_name="A_eig",
        parameters=["A"],
        augmentation_func=eig_augmentation,
        overwrite=overwrite
    )
    log.add_entity_history_by_augmentation(
        entity_name="A_svd",
        parameters=["A"],
        augmentation_func=svd_augmentation,
        overwrite=overwrite
    )
    log.add_entity_history_by_augmentation(
        entity_name="hankle",
        parameters=["A", "B", "C"],
        augmentation_func=hankel_augmentation,
        overwrite=overwrite
    )
    log.add_entity_history_by_augmentation(
        entity_name="hankle_eig",
        parameters=["hankle"],
        augmentation_func=eig_augmentation,
        overwrite=overwrite
    )
    log.add_entity_history_by_augmentation(
        entity_name="hankle_svd",
        parameters=["hankle"],
        augmentation_func=svd_augmentation,
        overwrite=overwrite
    )
    log.save(log_file_path)

    if progress_bar_actor is not None:
        progress_bar_actor.update.remote()


def augment_all_logs(result_paths: List[str],
                     overwrite: bool = True):
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    tasks = []
    for result_path in result_paths:
        result_path_reg = os.path.join(result_path, "*")

        for log_file_path in glob.glob(result_path_reg):
            task = save_all_augmentation.remote(log_file_path=log_file_path,
                                                overwrite=overwrite,
                                                progress_bar_actor=progress_bar_actor)
            tasks.append(task)

    pb.set_total(len(tasks))
    pb.print_until_done()


def run_aug():
    ray.init(num_cpus=120, ignore_reinit_error=True)
    # result_paths = ["../results/variance_mult",
    #                 "../results/changing_lag",
    #                 "../results/rot_abo"]
    result_paths = ["../results/non_linear/changing_lag"]
    augment_all_logs(result_paths=result_paths,
                     overwrite=False)


if __name__ == "__main__":
    run_aug()
