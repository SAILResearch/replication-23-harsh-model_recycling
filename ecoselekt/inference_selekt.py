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

    selekt_pred_df = pd.DataFrame(
        columns=[
            "window",
            "y_pred_proba_eco",
            "y_pred_eco",
            "y_true",
            "commit_id",
            "y_model_pred_proba",
            "y_model_pred",
        ]
    )

    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        start = time.time()
        split = pd.concat(
            [windows[j].iloc[-settings.SHIFT :] for j in range(i + 1, len(windows))],
            ignore_index=True,
        )

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

        # load saved model selection model
        start = time.time()
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_selekt_model.pkl", "rb"
        ) as f:
            nn = pickle.load(f)
        _LOGGER.info(f"Loaded selekt model in {time.time() - start}")

        best_models = nn.predict(test_feature)
        best_models_proba = nn.predict_proba(test_feature)[:, 1]

        # create dataframe with shape of test_feature
        perf_df = pd.DataFrame(index=range(len(test_feature)))

        perf_df["y_model_pred"] = best_models
        perf_df["y_model_pred_proba"] = best_models_proba

        def load_model(model_version):
            with open(
                settings.MODELS_DIR
                / f"{settings.EXP_ID}_{project_name}_w{model_version}_model.pkl",
                "rb",
            ) as f:
                return pickle.load(f)

        for pred_model in pred_df["model_version"].unique():
            best_model = load_model(pred_model)
            # inference
            indices = best_models == pred_model

            perf_df.loc[indices, "y_pred_eco"] = best_model.predict(test_feature[indices])
            perf_df.loc[indices, "y_pred_proba_eco"] = best_model.predict_proba(
                test_feature[indices]
            )[:, 1]

            _LOGGER.info(f"Finished inference for model {pred_model} in window {i}")

        perf_df["window"] = i
        perf_df["commit_id"] = test_commit_id
        perf_df["y_true"] = new_test_label
        # fix types for saving, for some reason they are float due to indice assignment
        perf_df["y_pred_eco"] = perf_df["y_pred_eco"].astype(int)
        perf_df["y_model_pred"] = perf_df["y_model_pred"].astype(int)

        # *[OUT]: save ecoselekt prediction results
        # out of loop assign in batch and concat
        selekt_pred_df = pd.concat(
            [
                selekt_pred_df,
                perf_df,
            ],
            ignore_index=True,
        )
        selekt_pred_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_pred.csv", index=False
        )

        _LOGGER.info(f"Saved selekt model predictions for window {i}")


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            inference_selekt(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
