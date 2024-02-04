import pickle
import time

import numpy as np
import pandas as pd
import shap

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_models import get_combined_df

_LOGGER = get_logger()


def calc_drift(project_name="activemq"):
    _LOGGER.info(f"Starting drift calculation for {project_name}")
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    all_features = set()
    for i in range(len(windows) - settings.C_TEST_WINDOWS):
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_model.pkl", "rb"
        ) as f:
            mi = pickle.load(f)

        ii = mi.named_steps["selectkbest"].get_support()

        fi = mi.named_steps["columntransformer"].get_feature_names_out()[ii]
        assert len(set(fi)) == len(fi)

        all_features.update(fi)

    shap_mean_df = pd.DataFrame(columns=["project", "window", *all_features])
    shap_std_df = pd.DataFrame(columns=["project", "window", *all_features])
    shap_max_df = pd.DataFrame(columns=["project", "window", *all_features])

    # ignore last few windows (`settings.TEST_SIZE`) since there is no test data if we use it
    for i in range(len(windows) - (2 * settings.C_TEST_WINDOWS)):
        _LOGGER.info(f"Starting window {i} for {project_name}")
        # randomly shuffle the window to get random validation set
        # get test data from further windows as per `settings.TEST_SIZE`
        split = pd.concat(
            [windows[j][-settings.SHIFT :] for j in range(i + 2, i + 2 + settings.F_TEST_WINDOWS)],
            ignore_index=True,
        )
        if settings.TEST_SIZE % settings.SHIFT != 0:
            split = pd.concat(
                [
                    split,
                    windows[i + 2 + settings.F_TEST_WINDOWS][
                        -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                    ],
                ],
                ignore_index=True,
            )
        _LOGGER.info(f"Shape of test data: {split.shape}")

        test_feature, test_commit_id, test_label = get_combined_df(
            split.code,
            split.commit_id,
            split.label,
            split.drop(["code", "label"], axis=1),
        )
        _LOGGER.info(f"Test feature shape: {test_feature.shape}")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_model.pkl", "rb"
        ) as f:
            mi = pickle.load(f)

        ii = mi.named_steps["selectkbest"].get_support()

        fi = mi.named_steps["columntransformer"].get_feature_names_out()[ii]
        assert len(set(fi)) == len(fi)

        X = mi.named_steps["columntransformer"].transform(test_feature)
        X = mi.named_steps["selectkbest"].transform(X)

        explainer = shap.TreeExplainer(mi.named_steps["randomforestclassifier"])
        shap_values = explainer.shap_values(X.toarray())[0]
        # get mean and std of shap values
        shap_mean = np.mean(shap_values, axis=0)
        shap_std = np.std(shap_values, axis=0)
        # get absolute max of shap values and sign
        shap_max = np.argmax(np.abs(shap_values), axis=0)
        shap_max = np.array([shap_values[j, i] for i, j in enumerate(shap_max)])

        shap_mean_df = shap_mean_df.append(
            pd.Series(
                [project_name, i, *shap_mean],
                index=["project", "window", *fi],
            ),
            ignore_index=True,
        )
        shap_std_df = shap_std_df.append(
            pd.Series(
                [project_name, i, *shap_std],
                index=["project", "window", *fi],
            ),
            ignore_index=True,
        )
        shap_max_df = shap_max_df.append(
            pd.Series(
                [project_name, i, *shap_max],
                index=["project", "window", *fi],
            ),
            ignore_index=True,
        )

    shap_mean_df.to_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shap_mean.csv", index=False
    )
    shap_std_df.to_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shap_std.csv", index=False
    )
    shap_max_df.to_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shap_max.csv", index=False
    )


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            calc_drift(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
