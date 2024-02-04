import pickle
import time
import warnings

import pandas as pd
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

warnings.filterwarnings("ignore")
_LOGGER = get_logger()


def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]["cum_LOC"]
    buggy_line_k_percent = result_df_arg[result_df_arg["cum_LOC"] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent["label"] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort


def eval_metrics(result_df):
    pred = result_df["defective_commit_pred"]
    y_test = result_df["label"]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary"
    )  # at threshold = 0.5

    try:
        AUC = roc_auc_score(y_test, result_df["defective_commit_prob"])
    except ValueError:
        AUC = -1
    gmean = geometric_mean_score(y_test, result_df["defective_commit_pred"])
    ap = average_precision_score(y_test, result_df["defective_commit_prob"])
    specifi = specificity_score(y_test, result_df["defective_commit_pred"])

    return (prec, rec, f1, AUC, gmean, ap, specifi)


def eval_model_ckpts(project_name):
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    _LOGGER.info(f"Project: {project_name} with {len(windows)} windows loaded")

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

    pred_result_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result_lr.csv"
    )
    for i in range(len(windows) - settings.C_TEST_WINDOWS):
        RF_result = pred_result_df[pred_result_df["window"] == i].copy()
        # drop extra columns dependent on `train_models.py`
        RF_result = RF_result.drop(columns=["model_version", "window"])

        RF_result.columns = [
            "defective_commit_prob",
            "defective_commit_pred",
            "label",
            "test_commit",
        ]  # for new result

        # evaluate model on all further data splits
        for x_split in range(i + 1, len(windows) - settings.C_TEST_WINDOWS + 1):
            # get test data from further windows as per `settings.TEST_SIZE`
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
                        [[project_name, i, x_split, prec, rec, f1, auc, gmean, ap, specifi]],
                        columns=evaluation_df.columns,
                    ),
                ],
                ignore_index=True,
            )

            _LOGGER.info(
                f"{settings.EXP_ID}"
                f"Project: {project_name}, Test split: {x_split}, Window: {i}, "
                f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}, "
                f"GMean: {gmean:.2f}, AP: {ap:.2f}, Specificity: {specifi:.2f}"
            )
    # *[OUT]: save evaluation result
    evaluation_df.to_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_base_eval_lr.csv", index=False
    )
    _LOGGER.info(f"Project: {project_name} evaluation result saved")


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            eval_model_ckpts(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
