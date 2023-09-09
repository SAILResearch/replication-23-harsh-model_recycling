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
            "y_model_pred",
            "n_commit_ids",
        ]
    )

    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        _LOGGER.info(f"Starting window {i} for project {project_name}")

        # filter out "unavailable at the window time" future test commits
        past_split = pd.concat(
            [
                windows[j].iloc[-settings.SHIFT :]
                for j in range(i - settings.MODEL_HISTORY, i + settings.F_TEST_WINDOWS)
            ],
            ignore_index=True,
        )
        if settings.TEST_SIZE % settings.SHIFT != 0:
            past_split = pd.concat(
                [
                    past_split,
                    windows[i + settings.F_TEST_WINDOWS][
                        -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                    ],
                ],
                ignore_index=True,
            )

        _, past_commit_id, _ = get_combined_df(
            past_split.code,
            past_split.commit_id,
            past_split.label,
            past_split.drop(["code", "label"], axis=1),
        )

        all_past_dfs = []
        # load all past model predictions including latest model prediction
        for j in range(i - settings.MODEL_HISTORY, i + 1):
            temp_df = pred_result_df[pred_result_df["window"] == j].copy()
            temp_df.rename(columns={"test_commit": "commit_id"}, inplace=True)
            temp_df.drop("window", axis=1, inplace=True)
            # filter out commit ids that are not in the current window
            temp_df = temp_df[temp_df["commit_id"].isin(past_split.commit_id)]
            all_past_dfs.append(temp_df)

        pastk_df = pd.concat(all_past_dfs, ignore_index=True)
        _LOGGER.info(f"pastk df shape: {pastk_df.shape}")
        pastk_df.set_index("commit_id", inplace=True)
        _LOGGER.info(f"pastk df shape after index setting: {pastk_df.shape}")

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

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_knn.pkl",
            "rb",
        ) as f:
            knn = pickle.load(f)

        def load_model(model_version):
            with open(
                settings.MODELS_DIR
                / f"{settings.EXP_ID}_{project_name}_w{model_version}_model.pkl",
                "rb",
            ) as f:
                return pickle.load(f)

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_preprocess_knn.pkl", "rb"
        ) as f:
            preprocess_knn = pickle.load(f)

        knn_test_feature = preprocess_knn.transform(test_feature)

        indices = knn.kneighbors(
            knn_test_feature,
            n_neighbors=settings.CURRENT_KNEIGHBOURS,
            return_distance=False,
        )
        _LOGGER.info(f"KNN indices shape: {indices.shape}")

        # create dataframe with shape of test_feature
        perf_df = pd.DataFrame(index=range(len(test_feature)))

        perf_df["commit_id"] = test_commit_id
        perf_df["n_commit_ids"] = [past_commit_id[idxes] for idxes in indices]
        _LOGGER.info("Completed finding nn commit ids")

        for idx, row in enumerate(indices):
            if idx % 100 == 0:
                _LOGGER.info(f"Processing {idx} of {len(indices)}")
            commit_ids = past_commit_id[row]
            perf_df.at[idx, "n_commit_ids"] = commit_ids

            temp_pastk_df = pastk_df.loc[commit_ids]
            perf_df.loc[idx, "y_model_pred"] = (
                np.argmin(
                    [
                        np.mean(
                            (
                                temp_pastk_df[temp_pastk_df["model_version"] == j]["actual"]
                                - temp_pastk_df[temp_pastk_df["model_version"] == j]["prob"]
                            )
                            ** 2
                        )
                        for j in range(i - settings.MODEL_HISTORY, i + 1)
                    ]
                )
                + i
                - settings.MODEL_HISTORY
            )
        _LOGGER.info("Completed finding best model")

        # fix type
        perf_df["y_model_pred"] = perf_df["y_model_pred"].astype(int)

        _LOGGER.info("Starting inference with best model")

        for best_model in perf_df["y_model_pred"].unique():
            nn = load_model(best_model)
            perf_df.loc[perf_df["y_model_pred"] == best_model, "y_pred_eco"] = nn.predict(
                test_feature.loc[perf_df["y_model_pred"] == best_model]
            )
            perf_df.loc[
                perf_df["y_model_pred"] == best_model, "y_pred_proba_eco"
            ] = nn.predict_proba(test_feature.loc[perf_df["y_model_pred"] == best_model])[:, 1]
            _LOGGER.info(f"Finished inference with best model {best_model}")

        perf_df["window"] = i
        perf_df["y_true"] = new_test_label

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

        _LOGGER.info(f"Saved recycled model predictions for window {i}")


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            inference_selekt(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
