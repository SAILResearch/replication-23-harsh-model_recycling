import pickle
import time
from collections import Counter

from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

pandas2ri.activate()

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from scipy.optimize import differential_evolution
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import ecoselekt.utils as utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_lr import get_combined_df_lr

_LOGGER = get_logger()


def objective_func_lr(k, train_feature, train_label, valid_feature, valid_label, estimators):
    pipeline = make_pipeline_imb(
        SMOTE(
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            k_neighbors=int(np.round(k)),
        ),
        LogisticRegression(
            random_state=settings.RANDOM_SEED,
        ),
    )

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=pipeline,
        n_jobs=-1,
        verbose=1,
        cv="prefit",
        stack_method="predict_proba",
        passthrough=False,
    )

    clf.fit(train_feature, train_label)

    prob = clf.predict_proba(valid_feature)[:, 1]
    auc = roc_auc_score(valid_label, prob)

    return -auc


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
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result_lr.csv"
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

        # use scottknott test to find similar model versions group
        og_model_versions = sorted(pred_df["model_version"].unique())
        _LOGGER.info(f"Total model versions: {len(og_model_versions)}")
        sk = importr("ScottKnottESD")
        data = pd.DataFrame(
            {
                model_version: pred_df[pred_df["model_version"] == model_version]["error"].values
                for model_version in pred_df["model_version"].unique()
            }
        )

        r_sk = sk.sk_esd(data, version="np", alpha=settings.ALPHA)
        column_order = list(r_sk[3] - 1)
        ranking = pd.DataFrame(
            {
                "model": [data.columns[i] for i in column_order],
                "rank": r_sk[1].astype("int"),
            }
        )  # long format
        _LOGGER.info(f"Ranking: {ranking}")
        # label each commit with the rank group
        pred_df = pd.merge(pred_df, ranking, left_on="model_version", right_on="model", how="left")
        pred_df.drop("model", axis=1, inplace=True)

        # deduplicate train_pred_df by commit_id keeping the row with the lowest error
        temp_df = pred_df.sort_values("error").drop_duplicates(subset="commit_id", keep="first")
        rank_to_model = {}
        # keep the most frequent model version for each rank group
        for rank in temp_df["rank"].unique():
            most_freq_model_version = (
                temp_df[temp_df["rank"] == rank]["model_version"].value_counts().index[0]
            )
            rank_to_model[rank] = most_freq_model_version
            temp_df["model_version"] = np.where(
                (temp_df["rank"] == rank) & (temp_df["model_version"] != most_freq_model_version),
                most_freq_model_version,
                temp_df["model_version"],
            )
        pred_df.drop("rank", axis=1, inplace=True)

        # filter out model versions that are similar to other model versions
        # and keep the latest model version
        model_versions = set(temp_df["model_version"].unique())
        if i not in model_versions:
            model_versions.add(i)

        pred_df = pred_df[pred_df["model_version"].isin(model_versions)]

        _LOGGER.info(f"After scottknott prediction df shape: {pred_df.shape}")
        _LOGGER.info(f"Model versions kept: {model_versions}")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_stat_models_lr.pkl", "wb"
        ) as f:
            f.write(pickle.dumps(model_versions))

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

        # stratified random subsampling to balance the dataset and reduce to `settings.WINDOW_SIZE`
        pred_df = pred_df.groupby("actual").apply(
            lambda x: x.sample(
                n=min(settings.WINDOW_SIZE // 2, x.shape[0]),
                random_state=settings.RANDOM_SEED,
                replace=False,
            )
        )
        _LOGGER.info(f"Train df shape: {pred_df.shape}")
        _LOGGER.info(f"Label distribution: {Counter(pred_df['actual'])}")

        # !: uncomment to save pred_df for experiment
        # pred_df.to_csv(
        #     settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_pred_train_exp.csv",
        #     index=False,
        # )
        train_feature, train_commit_id, train_label = get_combined_df_lr(
            pred_df.code,
            pred_df.commit_id,
            pred_df.actual,
            pred_df.drop(["pred", "prob", "actual", "code", "model_version"], axis=1),
        )

        # train validation split
        (
            final_train_feature,
            valid_feature,
            final_train_label,
            valid_label,
        ) = train_test_split(
            train_feature,
            train_label,
            test_size=0.2,
            random_state=settings.RANDOM_SEED,
            stratify=train_label,
        )

        def load_model(model_version):
            with open(
                settings.MODELS_DIR
                / f"{settings.EXP_ID}_{project_name}_w{model_version}_model_lr.pkl",
                "rb",
            ) as f:
                return pickle.load(f)

        estimators = [
            (
                f"model_{model_version}",
                load_model(model_version),
            )
            for model_version in model_versions
        ]

        start = time.time()

        positive_label_count = np.sum(final_train_label)
        bounds = [(1, positive_label_count - 1 if positive_label_count <= 20 else 20)]
        start = time.time()
        result = differential_evolution(
            objective_func_lr,
            bounds=bounds,
            args=(final_train_feature, final_train_label, valid_feature, valid_label, estimators),
            popsize=10,
            mutation=0.7,
            recombination=0.3,
            seed=0,
        )
        _LOGGER.info(f"Completed differential evolution for window {i} in {time.time() - start}")

        pipeline = make_pipeline_imb(
            SMOTE(
                random_state=settings.RANDOM_SEED,
                n_jobs=-1,
                k_neighbors=int(np.round(result.x)),
            ),
            LogisticRegression(random_state=settings.RANDOM_SEED),
        )

        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=pipeline,
            n_jobs=-1,
            verbose=1,
            cv="prefit",
            stack_method="predict_proba",
            passthrough=False,
        )

        clf.fit(final_train_feature, final_train_label)
        _LOGGER.info(f"Selekt model training finished in {time.time() - start}")

        # evaluate model on train data
        prob = clf.predict_proba(train_feature)
        pred = clf.predict(train_feature)
        _LOGGER.info(f"Validation score: {clf.score(valid_feature, valid_label)}")
        _LOGGER.info(f"Train score: {clf.score(final_train_feature, final_train_label)}")

        # *[OUT]: save results for evaluation
        selekt_eval_df = pd.concat(
            [
                selekt_eval_df,
                pd.DataFrame(
                    {
                        "window": i,
                        "prob": prob.tolist(),
                        "pred": pred,
                        "actual": train_label,
                        "test_commit": train_commit_id,
                    }
                ),
            ],
            ignore_index=True,
        )
        selekt_eval_df.to_csv(
            settings.DATA_DIR / (f"{settings.EXP_ID}_{project_name}_selekt_model_pred_lr.csv"),
            index=False,
        )

        # *[OUT]: save selekt model
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_selekt_model_lr.pkl", "wb"
        ) as f:
            pickle.dump(clf, f)
        _LOGGER.info(f"Selekt model saved for {project_name} window {i}")


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
