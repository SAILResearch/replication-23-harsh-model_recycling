import pickle
import time

import numpy as np
import pandas as pd

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_models import get_combined_df

_LOGGER = get_logger()


def inference_selekt(project_name):
    _LOGGER.info(f"Inferencing selekt for {project_name}")
    start = time.time()
    # load sliding windows splits
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    pred_result_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result.csv"
    )

    _LOGGER.info(
        f"Project: {project_name} with {len(windows)} windows loaded in {time.time() - start}"
    )

    inf_performance_df = pd.DataFrame(
        columns=["window", "commit_id", "eco_pred_time", "base_pred_time"]
    )

    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        start = time.time()
        split = pd.concat(
            [windows[j][-settings.SHIFT :] for j in range(i + 1, i + 1 + settings.F_TEST_WINDOWS)],
            ignore_index=True,
        )
        if settings.TEST_SIZE % settings.SHIFT != 0:
            split = pd.concat(
                [
                    split,
                    windows[i + 1 + settings.F_TEST_WINDOWS][
                        -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                    ],
                ],
                ignore_index=True,
            )
        _LOGGER.info(f"Shape of test data: {split.shape}")

        test_feature, test_commit_id, new_test_label = get_combined_df(
            split.code,
            split.commit_id,
            split.label,
            split.drop(["code", "label"], axis=1),
        )

        all_pred_dfs = []
        # load all future model predictions
        for j in range(i + 1):
            temp_df = pred_result_df[pred_result_df["window"] == j].copy()
            temp_df.rename(columns={"test_commit": "commit_id"}, inplace=True)
            temp_df.drop("window", axis=1, inplace=True)
            # filter out commit ids that are not in the current window
            temp_df = temp_df[temp_df["commit_id"].isin(split.commit_id)]
            all_pred_dfs.append(temp_df)

        pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        _LOGGER.info(f"Prediction df shape: {pred_df.shape}")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_best_old_model.pkl",
            "rb",
        ) as f:
            best_old_model = pickle.load(f)

        pred_df = pred_df[pred_df["model_version"].isin([best_old_model, i])].reset_index(drop=True)

        pred_df["error"] = abs(pred_df["actual"] - pred_df["prob"])

        # deduplicate train_pred_df by commit_id keeping the row with the lowest error
        pred_df = pred_df.sort_values("error", ascending=True).drop_duplicates(
            "commit_id", keep="first"
        )
        pred_df.set_index("commit_id", inplace=True)
        pred_df.reindex(test_commit_id)
        pred_df.reset_index(inplace=True)
        _LOGGER.info(f"After dedup prediction df shape: {pred_df.shape}")

        # # load saved model selection model
        # start = time.time()
        # with open(
        #     settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_selekt_model.pkl", "rb"
        # ) as f:
        #     nn = pickle.load(f)
        # _LOGGER.info(f"Loaded selekt model in {time.time() - start}")

        def load_model(model_version):
            with open(
                settings.MODELS_DIR
                / f"{settings.EXP_ID}_{project_name}_w{model_version}_model.pkl",
                "rb",
            ) as f:
                return pickle.load(f)

        old_nn = load_model(best_old_model)
        new_nn = load_model(i)
        eco_runtimes = np.zeros(test_feature.shape[0])
        base_runtimes = np.zeros(test_feature.shape[0])
        _LOGGER.info(type(test_feature))
        for j in range(test_feature.shape[0]):
            temp_df = test_feature[j : j + 1].copy()
            # baseline
            start = time.time()
            new_nn.predict(temp_df)
            base_runtimes[j] = time.time() - start

            # eco
            start = time.time()
            _LOGGER.info(
                1
                if (old_nn.predict_proba(temp_df)[:, 1] + new_nn.predict_proba(temp_df)[:, 1]) / 2
                > 0.5
                else 0
            )
            eco_runtimes[j] = time.time() - start

        # *[OUT]: save ecoselekt inference latency results
        inf_performance_df = pd.concat(
            [
                inf_performance_df,
                pd.DataFrame(
                    {
                        "window": i,
                        "commit_id": test_commit_id,
                        "eco_pred_time": eco_runtimes,
                        "base_pred_time": base_runtimes,
                    }
                ),
            ],
            ignore_index=True,
        )
        inf_performance_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_inf_perf.csv",
            index=False,
        )
        _LOGGER.info(f"Saved inference latency results for window {i}")


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            inference_selekt(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
