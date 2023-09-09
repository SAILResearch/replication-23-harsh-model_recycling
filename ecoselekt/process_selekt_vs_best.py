import pickle
import time

import pandas as pd

from ecoselekt.eval_models import eval_metrics
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def process_best_model_per_test_split(project_name="activemq"):
    _LOGGER.info(f"Starting best model processing for {project_name}")

    start = time.time()
    # load sliding windows splits
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    _LOGGER.info(
        f"Project: {project_name} with {len(windows)} windows loaded in {time.time() - start}"
    )

    pred_results_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result.csv"
    )

    best_model_df = pd.DataFrame(columns=["commit_id", "window", "test_split", "model_version"])
    best_model_eval_df = pd.DataFrame(
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
    # start after model history is built
    # and ignore last few windows (`settings.TEST_SIZE`) since there is no test data if we use it
    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        _LOGGER.info(f"Starting window {i} for {project_name}")

        all_pred_dfs = []
        # load all past model predictions including latest model prediction
        for j in range(i - settings.MODEL_HISTORY, i + 1):
            # window and model version are the same
            temp_df = pred_results_df[pred_results_df["window"] == j]
            temp_df["commit_id"] = temp_df["test_commit"]
            temp_df.drop("test_commit", axis=1, inplace=True)
            all_pred_dfs.append(temp_df)

        pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        _LOGGER.info(f"Prediction df shape: {pred_df.shape}")

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

            # filter out commit ids that are not in the current window
            xsplit_df = pred_df[pred_df["commit_id"].isin(split.commit_id)]

            # filter out non-significant model versions
            with open(
                settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_best_old_model.pkl",
                "rb",
            ) as f:
                best_old_model = pickle.load(f)
            xsplit_df = xsplit_df[xsplit_df["model_version"].isin([best_old_model, i])]

            xsplit_df["error"] = abs(xsplit_df["actual"] - xsplit_df["prob"])
            # deduplicate xsplit_df by commit_id keeping the row with the lowest error
            xsplit_df = xsplit_df.sort_values("error").drop_duplicates(
                subset="commit_id", keep="first"
            )
            _LOGGER.info(f"After dedup xsplit df shape: {xsplit_df.shape}")

            temp_df = pd.DataFrame(
                {"commit_id": xsplit_df.commit_id, "model_version": xsplit_df.model_version}
            )
            temp_df["window"] = i
            temp_df["test_split"] = x_split

            best_model_df = pd.concat([best_model_df, temp_df])

            # calculate evaluation metrics
            test_commit = split.commit_id

            RF_df = pd.DataFrame()
            RF_df["commit_id"] = test_commit

            RF_final = pd.merge(RF_df, xsplit_df, how="inner", on="commit_id")
            RF_final = RF_final[["prob", "pred", "actual", "commit_id"]]
            RF_final.columns = [
                "defective_commit_prob",
                "defective_commit_pred",
                "label",
                "test_commit",
            ]  # for new result
            (prec, rec, f1, auc, gmean, ap, specifi) = eval_metrics(RF_final)

            # store evaluation result
            best_model_eval_df = pd.concat(
                [
                    best_model_eval_df,
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
                f"Best select [w{i}, {x_split}], Project: {project_name}"
                f", Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}, "
                f"GMean: {gmean:.2f}, AP: {ap:.2f}, Specificity: {specifi:.2f}"
            )
        best_model_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_best_model_per_test_split.csv"
        )
        # *[OUT]: save best select evaluation result
        best_model_eval_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_best_eval.csv", index=False
        )
        _LOGGER.info(
            f"Saved best model per test split and its evaluation for window {i} for {project_name}"
        )


def process_selected_model_per_test_split(project_name="activemq"):
    _LOGGER.info(f"Starting selected model processing for {project_name}")

    start = time.time()
    # load sliding windows splits
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    _LOGGER.info(
        f"Project: {project_name} with {len(windows)} windows loaded in {time.time() - start}"
    )

    selekt_pred_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_pred.csv"
    )

    selected_model_df = pd.DataFrame(columns=["commit_id", "window", "test_split", "model_version"])

    # start after model history is built
    # and ignore last few windows (`settings.TEST_SIZE`) since there is no test data if we use it
    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        _LOGGER.info(f"Starting window {i} for {project_name}")

        # load model predictions of ecoselekt
        w_selekt_pred_df = selekt_pred_df[selekt_pred_df["window"] == i].copy()
        _LOGGER.info(f"selektperf df shape: {w_selekt_pred_df.shape}")

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

            # filter out commit ids that are not in the current window
            xsplit_df = w_selekt_pred_df[w_selekt_pred_df["commit_id"].isin(split.commit_id)]

            _LOGGER.info(f"After filtering xsplit df shape: {xsplit_df.shape}")

            temp_df = pd.DataFrame(
                {"commit_id": xsplit_df.commit_id, "model_version": xsplit_df.y_model_pred}
            )
            temp_df["window"] = i
            temp_df["test_split"] = x_split

            selected_model_df = pd.concat([selected_model_df, temp_df])

        selected_model_df.to_csv(
            settings.DATA_DIR
            / f"{settings.EXP_ID}_{project_name}_selected_model_per_test_split.csv"
        )
        _LOGGER.info(f"Saved selected model per test split for window {i} for {project_name}")


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            process_best_model_per_test_split(project_name)
            # process_selected_model_per_test_split(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
