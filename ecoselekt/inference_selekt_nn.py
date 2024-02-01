import pickle
import time

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
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result_nn.csv"
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
        for j in range(i - settings.MODEL_HISTORY, i + 1):
            temp_df = pred_result_df[pred_result_df["window"] == j].copy()
            temp_df.rename(columns={"test_commit": "commit_id"}, inplace=True)
            temp_df.drop("window", axis=1, inplace=True)
            # filter out commit ids that are not in the current window
            temp_df = temp_df[temp_df["commit_id"].isin(split.commit_id)]
            all_pred_dfs.append(temp_df)

        pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        _LOGGER.info(f"Prediction df shape: {pred_df.shape}")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_stat_models_nn.pkl",
            "rb",
        ) as f:
            stat_models = pickle.load(f)

        pred_df = pred_df[pred_df["model_version"].isin(stat_models)].reset_index(drop=True)
        # add models probabilities as features
        for model_version in stat_models:
            prob_df = (
                pred_df[pred_df["model_version"] == model_version][["commit_id", "prob"]]
                .drop_duplicates(subset="commit_id", keep="first")
                .rename(columns={"prob": f"prob_{model_version}"})
                .reset_index(drop=True)
                .copy()
            )
            pred_df = pred_df.merge(prob_df, on="commit_id", how="left")

        model_prob_features = [f"prob_{model_version}" for model_version in stat_models]

        # deduplicate train_pred_df by commit_id keeping the row with the lowest error
        pred_df = pred_df.drop_duplicates(subset="commit_id", keep="first")
        pred_df.set_index("commit_id", inplace=True)
        pred_df.reindex(test_commit_id)
        pred_df.reset_index(inplace=True)
        _LOGGER.info(f"After dedup prediction df shape: {pred_df.shape}")

        final_test_feature = pred_df[model_prob_features].values

        # create dataframe with shape of test_feature
        perf_df = pd.DataFrame(index=range(len(final_test_feature)))

        perf_df["y_pred_eco"] = final_test_feature.mean(axis=1) > 0.5
        perf_df["y_pred_proba_eco"] = final_test_feature.mean(axis=1)

        _LOGGER.info(f"Finished inference for window {i}")

        perf_df["window"] = i
        perf_df["commit_id"] = test_commit_id
        perf_df["y_true"] = new_test_label
        # fix types for saving, for some reason they are float due to indice assignment
        perf_df["y_pred_eco"] = perf_df["y_pred_eco"].astype(int)

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
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_pred_nn.csv", index=False
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
