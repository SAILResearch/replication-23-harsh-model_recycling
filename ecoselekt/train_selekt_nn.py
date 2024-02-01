import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from scipy.optimize import differential_evolution
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import ecoselekt.utils as utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_models import get_combined_df, objective_func

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

        max_score = 0
        max_score_model_version = -1
        for model_version in range(i - settings.MODEL_HISTORY, i):
            temp_df = pred_df[pred_df["model_version"] == model_version]
            score = geometric_mean_score(temp_df["actual"], temp_df["pred"])
            if score > max_score:
                max_score = score
                max_score_model_version = model_version
            _LOGGER.info(f"Model version {model_version} Score: {score}")

        _LOGGER.info(f"Max gmean score: {max_score} for model version: {max_score_model_version}")

        pred_df = pred_df[pred_df["model_version"].isin([max_score_model_version, i])]

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_best_old_model_nn.pkl",
            "wb",
        ) as f:
            pickle.dump(max_score_model_version, f)

        # deduplicate train_pred_df by commit_id
        pred_df = pred_df.sort_values("error", ascending=True).drop_duplicates(
            subset=["commit_id"], keep="first"
        )
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

        train_feature, train_commit_id, train_label = get_combined_df(
            pred_df.code,
            pred_df.commit_id,
            pred_df.model_version,
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

        start = time.time()

        tokenize_ = make_column_transformer(
            (CountVectorizer(min_df=3, ngram_range=(1, 1)), "code"),
            remainder="passthrough",
        )
        preprocess_ = make_pipeline(
            tokenize_,
            SelectKBest(chi2, k=settings.K_FEATURES),
        )

        positive_label_count = np.sum(final_train_label)
        bounds = [(1, positive_label_count - 1 if positive_label_count < 20 else 20)]
        start = time.time()
        result = differential_evolution(
            objective_func,
            bounds=bounds,
            args=(
                preprocess_.fit_transform(final_train_feature, final_train_label),
                final_train_label,
                preprocess_.transform(valid_feature),
                valid_label,
            ),
            popsize=10,
            mutation=0.7,
            recombination=0.3,
            seed=0,
        )
        _LOGGER.info(f"Completed differential evolution for window {i} in {time.time() - start}")

        clf = make_pipeline_imb(
            tokenize_,
            SelectKBest(chi2, k=settings.K_FEATURES),
            SMOTE(
                random_state=settings.RANDOM_SEED,
                n_jobs=-1,
                k_neighbors=int(np.round(result.x)),
            ),
            RandomForestClassifier(
                n_estimators=300,
                random_state=settings.RANDOM_SEED,
                n_jobs=-1,
                verbose=1,
            ),
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
            settings.DATA_DIR / (f"{settings.EXP_ID}_{project_name}_selekt_model_pred_nn.csv"),
            index=False,
        )

        # *[OUT]: save selekt model
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_selekt_model_nn.pkl", "wb"
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
        exit(1)
