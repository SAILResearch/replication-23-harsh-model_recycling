import ast
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_models import get_combined_df

_LOGGER = get_logger()

RESULTS_DIR = settings.DATA_DIR.parent.parent / "results" / f"exp_{settings.EXP_ID}"


def drift_check(project_name="activemq"):
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows_data = pickle.load(f)
    bestmodel_df = pd.read_csv(
        RESULTS_DIR / f"{settings.EXP_ID}_{project_name}_best_model_per_test_split.csv"
    )
    bestmodel_df.drop(columns=["Unnamed: 0"], inplace=True)
    bestmodel_df = bestmodel_df[bestmodel_df["test_split"] == bestmodel_df["window"] + 2]

    # find winning windows for each reused model version
    reused_model_versions = {}
    winning_model_perc = {}
    for i in bestmodel_df["window"].unique():
        temp_df = bestmodel_df[bestmodel_df["window"] == i]
        modelv = temp_df["model_version"].value_counts().index[0]
        if modelv == i:
            _LOGGER.info(f"Model version {modelv} is latest RFS model for window {i}")
            continue
        if reused_model_versions.get(modelv) is None:
            reused_model_versions[modelv] = [i]
            winning_model_perc[modelv] = [
                temp_df["model_version"].value_counts().values[0] / temp_df.shape[0]
            ]
        else:
            reused_model_versions[modelv].append(i)
            winning_model_perc[modelv].append(
                temp_df["model_version"].value_counts().values[0] / temp_df.shape[0]
            )

    # find shapdrift for each winning model version
    drift_results_df = pd.DataFrame(
        columns=[
            "project",
            "window",
            "model_version",
            "surv_rate",
            "perc",
            "drifted_columns",
            "total_columns",
            "first_window",
            "last_window",
        ]
    )

    shap_df = pd.read_csv(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shapdrift.csv")
    shap_df.fi = shap_df.fi.apply(lambda x: ast.literal_eval(x))
    shap_df.fj = shap_df.fj.apply(lambda x: ast.literal_eval(x))

    for i, windows in reused_model_versions.items():
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_model.pkl", "rb"
        ) as f:
            mi = pickle.load(f)

        ii = mi.named_steps["selectkbest"].get_support()

        features_i = mi.named_steps["columntransformer"].get_feature_names_out()[ii]

        # get test data from further windows as per `settings.TEST_SIZE`
        split_x = pd.concat(
            [
                windows_data[x][-settings.SHIFT :]
                for x in range(i + 2, i + 2 + settings.F_TEST_WINDOWS)
            ],
            ignore_index=True,
        )
        if settings.TEST_SIZE % settings.SHIFT != 0:
            split_x = pd.concat(
                [
                    split_x,
                    windows_data[i + 2 + settings.F_TEST_WINDOWS][
                        -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                    ],
                ],
                ignore_index=True,
            )
        _LOGGER.info(f"Shape of test data: {split_x.shape}")
        test_feature_x, test_commit_id_x, test_label_x = get_combined_df(
            split_x.code,
            split_x.commit_id,
            split_x.label,
            split_x.drop(["code", "label"], axis=1),
        )

        ALPHA = 0.05 / len(windows)

        columns_with_drift = []
        total_columns = []
        for j in windows:
            with open(
                settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{j}_model.pkl", "rb"
            ) as f:
                mj = pickle.load(f)

            ij = mj.named_steps["selectkbest"].get_support()
            features_j = mj.named_steps["columntransformer"].get_feature_names_out()[ij]

            temp_df = shap_df[shap_df["mi"] == i]
            temp_df = temp_df[temp_df["mj"] == j]
            fi = temp_df["fi"].values[0]
            fj = temp_df["fj"].values[0]

            _LOGGER.info(len(set(features_j).intersection(set(fj))))
            _LOGGER.info(len(set(fj)))

            common_features = set(fi).intersection(set(fj))
            common_features_i = [x for x in features_i if x in common_features]
            common_features_j = [x for x in features_j if x in common_features]
            # assert if order of features is same so later we can use mask created
            # from common features
            assert common_features_i == common_features_j

            j_mask = np.array([feat in common_features for feat in features_j])
            i_mask = np.array([feat in common_features for feat in features_i])

            # get test data from further windows as per `settings.TEST_SIZE`
            split_y = pd.concat(
                [
                    windows_data[y][-settings.SHIFT :]
                    for y in range(j + 2, j + 2 + settings.F_TEST_WINDOWS)
                ],
                ignore_index=True,
            )
            if settings.TEST_SIZE % settings.SHIFT != 0:
                split_y = pd.concat(
                    [
                        split_y,
                        windows_data[j + 2 + settings.F_TEST_WINDOWS][
                            -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                        ],
                    ],
                    ignore_index=True,
                )
            _LOGGER.info(f"Shape of test data: {split_y.shape}")
            test_feature_y, test_commit_id_y, test_label_y = get_combined_df(
                split_y.code,
                split_y.commit_id,
                split_y.label,
                split_y.drop(["code", "label"], axis=1),
            )

            # run test_feature_x through column transformer and selectkbest
            tf_x = mi.named_steps["columntransformer"].transform(test_feature_x)
            tf_x = mi.named_steps["selectkbest"].transform(tf_x)
            # select only features that are in both models
            tf_x = tf_x[:, i_mask]
            _LOGGER.info(f"Shape of test data after feature selection: {tf_x.shape}")

            # run test_feature_y through column transformer and selectkbest
            tf_y = mj.named_steps["columntransformer"].transform(test_feature_y)
            tf_y = mj.named_steps["selectkbest"].transform(tf_y)
            # select only features that are in both models
            tf_y = tf_y[:, j_mask]
            _LOGGER.info(f"Shape of test data after feature selection: {tf_y.shape}")

            count = 0
            for col in range(tf_x.shape[1]):
                x = tf_x[:, col].toarray().flatten()
                y = tf_y[:, col].toarray().flatten()
                if stats.ks_2samp(x, y).pvalue < ALPHA:
                    count += 1

            columns_with_drift.append(count)
            total_columns.append(tf_x.shape[1])

        temp_df = shap_df[shap_df["mi"] == i]
        temp_df = temp_df[temp_df["mj"] >= settings.MODEL_HISTORY]

        reused_temp_df = temp_df[temp_df["mj"].isin(windows)].copy()
        not_reused_temp_df = temp_df[(~temp_df["mj"].isin(windows)) & (temp_df["mj"] > i)].copy()

        reused_temp_df["surv_rate"] = reused_temp_df["surv_rate"].apply(lambda x: x * 100)
        reused_temp_df["surv_rate"] = reused_temp_df["surv_rate"].apply(lambda x: round(x, 2))
        not_reused_temp_df["surv_rate"] = not_reused_temp_df["surv_rate"].apply(lambda x: x * 100)
        not_reused_temp_df["surv_rate"] = not_reused_temp_df["surv_rate"].apply(
            lambda x: round(x, 2)
        )

        drift_results_df = pd.concat(
            [
                drift_results_df,
                pd.DataFrame(
                    {
                        "project": project_name,
                        "window": reused_temp_df["mj"],
                        "model_version": i,
                        "surv_rate": reused_temp_df["surv_rate"],
                        "perc": winning_model_perc[i],
                        "drifted_columns": columns_with_drift,
                        "total_columns": total_columns,
                        "first_window": min(reused_model_versions[i]),
                        "last_window": max(reused_model_versions[i]),
                    }
                ),
            ],
            ignore_index=True,
        )
        drift_results_df = pd.concat(
            [
                drift_results_df,
                pd.DataFrame(
                    {
                        "project": project_name,
                        "window": not_reused_temp_df["mj"],
                        "model_version": i,
                        "surv_rate": not_reused_temp_df["surv_rate"],
                        "perc": None,
                        "drifted_columns": None,
                        "total_columns": None,
                        "first_window": None,
                        "last_window": None,
                    }
                ),
            ],
            ignore_index=True,
        )
        drift_results_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_rq1_step1.csv",
            index=False,
        )


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            drift_check(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
