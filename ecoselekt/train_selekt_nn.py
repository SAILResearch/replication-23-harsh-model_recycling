import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import ecoselekt.utils as utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def save_selekt_model(project_name="activemq"):
    _LOGGER.info(f"Starting selekt model training for {project_name}")
    start = time.time()
    # get test train code changes
    (
        all_code,
        all_commit,
        all_label,
    ) = utils.prep_apachejit_data(project_name)
    _LOGGER.info(f"Loaded code changes in {time.time() - start}")

    # get commit metrics
    start = time.time()
    commit_metrics = utils.get_apachejit_commit_metrics(project_name)
    commit_metrics = commit_metrics.drop(
        ["fix", "year", "buggy"],
        axis=1,
    )
    commit_metrics = commit_metrics.fillna(value=0)
    _LOGGER.info(f"Loaded commit metrics in {time.time() - start}")

    # combine train and test code change without label because
    # label is model version which we add later
    df = pd.DataFrame(
        {
            "commit_id": all_commit,
            "code": all_code,
        }
    )

    # merge commit metrics to train code
    df = pd.merge(df, commit_metrics, on="commit_id")
    _LOGGER.info(f"Loaded total data size for project({project_name}): {df.shape}")

    start = time.time()
    # load sliding windows splits
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    _LOGGER.info(
        f"Project: {project_name} with {len(windows)} windows loaded in {time.time() - start}"
    )

    pred_result_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result_nn.csv"
    )

    selekt_train_df = pd.DataFrame(columns=["window", "commit_id", "model_version"])
    selekt_eval_df = pd.DataFrame(columns=["window", "prob", "pred", "actual", "test_commit"])
    # start after model history is built
    # and ignore last few windows (`settings.TEST_SIZE`) since there is no test data if we use it
    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        _LOGGER.info(f"Starting window {i} for {project_name}")

        # filter out "unavailable at the window time" future test commits
        split = pd.concat(
            [
                windows[j].iloc[-settings.SHIFT :]
                for j in range(i - settings.MODEL_HISTORY, i + settings.F_TEST_WINDOWS)
            ],
            ignore_index=True,
        )
        if settings.TEST_SIZE % settings.SHIFT != 0:
            split = pd.concat(
                [
                    split,
                    windows[i + settings.F_TEST_WINDOWS][
                        -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                    ],
                ],
                ignore_index=True,
            )

        all_pred_dfs = []
        # load all past model predictions including latest model prediction
        for j in range(i - settings.MODEL_HISTORY, i + 1):
            temp_df = pred_result_df[pred_result_df["window"] == j].copy()
            temp_df.rename(columns={"test_commit": "commit_id"}, inplace=True)
            temp_df.drop("window", axis=1, inplace=True)
            # filter out commit ids that are not in the current window
            temp_df = temp_df[temp_df["commit_id"].isin(split.commit_id)]
            all_pred_dfs.append(temp_df)

        pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        _LOGGER.info(f"Prediction df shape: {pred_df.shape}")

        pred_df["error"] = abs(pred_df["actual"] - pred_df["prob"])

        # use wilcoxon rank sum test to find similar model versions group
        # and keep only one model version from each group
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        og_model_versions = sorted(pred_df["model_version"].unique())
        _LOGGER.info(f"Total model versions: {len(og_model_versions)}")

        models_to_remove = []
        # deduplicate and copy train_pred_df by commit_id keeping the row with the lowest error
        temp_pred_df = (
            pred_df.sort_values("error").drop_duplicates(subset="commit_id", keep="first").copy()
        )

        # get all model versions that are not in the same group as the any other model versions
        _LOGGER.info(f"Filtering using wilcoxon rank sum test with alpha={settings.ALPHA}")
        for model_version_a in range(len(og_model_versions) - 1):
            if og_model_versions[model_version_a] in models_to_remove:
                continue
            for model_version_b in range(model_version_a + 1, len(og_model_versions)):
                _, p = wilcoxon(
                    pred_df[pred_df["model_version"] == og_model_versions[model_version_a]][
                        "error"
                    ].values,
                    pred_df[pred_df["model_version"] == og_model_versions[model_version_b]][
                        "error"
                    ].values,
                )
                if p >= settings.ALPHA:
                    # check frequncy of model version
                    if (
                        temp_pred_df[
                            temp_pred_df["model_version"] == og_model_versions[model_version_a]
                        ]["error"].count()
                        > temp_pred_df[
                            temp_pred_df["model_version"] == og_model_versions[model_version_b]
                        ]["error"].count()
                    ):
                        models_to_remove.append(og_model_versions[model_version_b])
                    else:
                        models_to_remove.append(og_model_versions[model_version_a])
                        break
        # filter out model versions that are similar to other model versions
        # and keep the latest model version
        model_versions = set(og_model_versions) - set(models_to_remove)
        if i not in model_versions:
            model_versions.add(i)

        pred_df = pred_df[pred_df["model_version"].isin(model_versions)]

        _LOGGER.info(f"After wilcoxon prediction df shape: {pred_df.shape}")
        _LOGGER.info(f"Model versions kept: {model_versions}")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_stat_models_nn.pkl", "wb"
        ) as f:
            f.write(pickle.dumps(model_versions))

        # add models probabilities as features
        for model_version in model_versions:
            prob_df = (
                pred_df[pred_df["model_version"] == model_version][["commit_id", "prob"]]
                .drop_duplicates(subset="commit_id", keep="first")
                .rename(columns={"prob": f"prob_{model_version}"})
                .reset_index(drop=True)
                .copy()
            )
            pred_df = pred_df.merge(prob_df, on="commit_id", how="left")

        # deduplicate train_pred_df by commit_id
        pred_df = pred_df.drop_duplicates(subset="commit_id", keep="first")
        _LOGGER.info(f"After dedup prediction df shape: {pred_df.shape}")
        # shuffle the dataframe
        pred_df = pred_df.sample(frac=1, random_state=settings.RANDOM_SEED).reset_index(drop=True)
        pred_df.drop("error", axis=1, inplace=True)
        _LOGGER.info(f"Label distribution: {Counter(pred_df['model_version'])}")

        # *[OUT]: save ecoselekt train data
        selekt_train_df = pd.concat(
            [
                selekt_train_df,
                pred_df[["commit_id", "model_version"]].assign(window=i),
            ],
            ignore_index=True,
        )
        selekt_train_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_train_data.csv",
            index=False,
        )

        pred_df = pd.merge(pred_df, df, on="commit_id", how="left")

        # # train using current window
        # train_split = windows[i].sample(
        #     frac=1, random_state=settings.RANDOM_SEED, ignore_index=True
        # )
        # # keep current window commit ids
        # pred_df = pred_df[pred_df["commit_id"].isin(train_split.commit_id)]

        # !: uncomment to save pred_df for experiment
        # pred_df.to_csv(
        #     settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_pred_train_exp.csv",
        #     index=False,
        # )

        model_prob_features = [f"prob_{model_version}" for model_version in model_versions]

        # train validation split
        (
            final_train_feature,
            valid_feature,
            final_train_label,
            valid_label,
        ) = train_test_split(
            pred_df[model_prob_features].values,
            pred_df.actual.values,
            test_size=0.2,
            random_state=settings.RANDOM_SEED,
            stratify=pred_df.actual.values,
        )

        _LOGGER.info(f"Selekt model training finished in {time.time() - start}")

        # evaluate model on train data
        prob = pred_df[model_prob_features].values.mean(axis=1)
        pred = np.where(prob > 0.5, 1, 0)
        _LOGGER.info(f"auc: {roc_auc_score(pred_df.actual.values, prob)}")

        # *[OUT]: save results for evaluation
        selekt_eval_df = pd.concat(
            [
                selekt_eval_df,
                pd.DataFrame(
                    {
                        "window": i,
                        "prob": prob.tolist(),
                        "pred": pred,
                        "actual": pred_df.actual.values.tolist(),
                        "test_commit": pred_df.commit_id.values.tolist(),
                    }
                ),
            ],
            ignore_index=True,
        )
        selekt_eval_df.to_csv(
            settings.DATA_DIR / (f"{settings.EXP_ID}_{project_name}_selekt_model_pred_nn.csv"),
            index=False,
        )
        _LOGGER.info(f"Selekt model prepared for {project_name} window {i}")


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            save_selekt_model(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
