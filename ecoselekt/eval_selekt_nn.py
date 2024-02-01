import pickle
import time

import pandas as pd

from ecoselekt.eval_models import eval_metrics
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def eval_selekt(project_name):
    _LOGGER.info(f"Evaluating selekt for {project_name}")
    start = time.time()
    # load sliding windows splits
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    _LOGGER.info(
        f"Project: {project_name} with {len(windows)} windows loaded in {time.time() - start}"
    )

    evaluation_df = pd.DataFrame(
        columns=[
            "project",
            "window",
            "test_split",
            "precision",
            "recall",
            "f1",
            "auc",
            "gmean",
            "ap",
            "specifi",
        ]
    )

    selekt_pred_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_pred_nn.csv"
    )

    # start after model history is built
    # and ignore last window since there is no test data if we use it
    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        RF_result = selekt_pred_df[selekt_pred_df["window"] == i].copy()
        # drop extra columns dependent on `inference_selekt.py`
        RF_result = RF_result.drop(columns=["window"])

        RF_result.columns = [
            "defective_commit_prob",
            "defective_commit_pred",
            "label",
            "test_commit",
        ]  # for new result

        # test performance on each data split
        for x_split in range(i + 1, len(windows) - settings.C_TEST_WINDOWS + 1):
            split = pd.concat(
                [
                    windows[j][-settings.SHIFT :]
                    for j in range(x_split, x_split + settings.F_TEST_WINDOWS)
                ],
                ignore_index=True,
            )
            if settings.TEST_SIZE % settings.SHIFT != 0:
                split = pd.concat(
                    [
                        split,
                        windows[x_split + settings.F_TEST_WINDOWS][
                            -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                        ],
                    ],
                    ignore_index=True,
                )
            _LOGGER.info(f"Shape of test data: {split.shape}")
            test_commit = split.commit_id

            RF_df = pd.DataFrame()
            RF_df["commit_id"] = test_commit

            RF_final = pd.merge(
                RF_df, RF_result, how="inner", left_on="commit_id", right_on="test_commit"
            )
            (prec, rec, f1, auc, gmean, ap, specifi) = eval_metrics(RF_final)

            # store evaluation result
            evaluation_df = pd.concat(
                [
                    evaluation_df,
                    pd.DataFrame(
                        {
                            "project": [project_name],
                            "window": [i],
                            "test_split": [x_split],
                            "precision": [prec],
                            "recall": [rec],
                            "f1": [f1],
                            "auc": [auc],
                            "gmean": [gmean],
                            "ap": [ap],
                            "specifi": [specifi],
                        }
                    ),
                ],
                ignore_index=True,
            )

            _LOGGER.info(
                f"Ecoselekt [w{i}, {x_split}], Project: {project_name}"
                f", Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}, "
                f"GMean: {gmean:.2f}, AP: {ap:.2f}, Specifi: {specifi:.2f}"
            )
    # *[OUT]: save ecoselekt evaluation result
    evaluation_df.to_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_eval_nn.csv", index=False
    )
    _LOGGER.info(f"Selekt evaluation result saved for project {project_name}")


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            eval_selekt(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
        exit(1)
