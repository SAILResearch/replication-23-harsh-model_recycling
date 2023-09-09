import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from scipy.optimize import differential_evolution
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import ecoselekt.utils as utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def get_combined_df(code_commit, commit_id, label, metrics_df):
    code_df = pd.DataFrame({"code": code_commit, "commit_id": commit_id, "label": label})

    code_df = code_df.sort_values(by="commit_id")

    metrics_df = metrics_df.sort_values(by="commit_id")
    # !: project is needed to be removed only for apachejit data
    metrics_df = metrics_df.drop(["commit_id", "project", "author_date"], axis=1)

    final_features = pd.concat([code_df["code"], metrics_df], axis=1)

    return final_features, np.array(code_df["commit_id"]), np.array(code_df["label"])


def objective_func(k, train_feature, train_label, valid_feature, valid_label):
    smote = SMOTE(random_state=42, k_neighbors=int(np.round(k)), n_jobs=32)
    train_feature_res, train_label_res = smote.fit_resample(train_feature, train_label)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train_feature_res, train_label_res)

    prob = clf.predict_proba(valid_feature)[:, 1]
    auc = roc_auc_score(valid_label, prob)

    return -auc


def save_model_ckpts(project_name="activemq"):
    _LOGGER.info(f"Starting checkpoint model training for {project_name}")
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
    # !: should we keep time information?
    commit_metrics = commit_metrics.drop(
        ["fix", "year", "buggy"],
        axis=1,
    )
    commit_metrics = commit_metrics.fillna(value=0)
    _LOGGER.info(f"Loaded commit metrics in {time.time() - start}")

    # combine train and test code changes
    df = pd.DataFrame(
        {
            "commit_id": all_commit,
            "code": all_code,
            "label": all_label,
        }
    )

    # merge commit metrics to train code
    df = pd.merge(df, commit_metrics, on="commit_id")
    _LOGGER.info(f"Loaded total data size for project({project_name}): {df.shape}")
    _LOGGER.info(f"Label distribution for project({project_name}): {Counter(df.label)}")

    # get data splits with sliding window
    df = df.sort_values(by="author_date").reset_index(drop=True)
    windows = utils.get_sliding_windows(df, settings.WINDOW_SIZE, settings.SHIFT)
    _LOGGER.info(f"Loaded windows: {len(windows)}")

    # save windows for evaluation
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "wb") as f:
        pickle.dump(windows, f)

    pred_results_df = pd.DataFrame(
        columns=["window", "prob", "pred", "actual", "test_commit", "model_version"]
    )

    # ignore last few windows (`settings.TEST_SIZE`) since there is no test data if we use it
    for i, window in enumerate(windows[: -settings.C_TEST_WINDOWS]):
        _LOGGER.info(f"Starting window {i} for {project_name}")
        start = time.time()

        # randomly shuffle the window to get random validation set
        split = window.sample(frac=1, random_state=settings.RANDOM_SEED, ignore_index=True)

        train_feature, train_commit_id, train_label = get_combined_df(
            split.code,
            split.commit_id,
            split.label,
            split.drop(["code", "label"], axis=1),
        )
        _LOGGER.info(f"Train feature shape: {train_feature.shape}")

        # train validation split
        final_train_feature, valid_feature, final_train_label, valid_label = train_test_split(
            train_feature, train_label, test_size=0.2, random_state=settings.RANDOM_SEED
        )
        _LOGGER.info(f"New train feature shape: {train_feature.shape}")

        full_split = pd.concat(windows, ignore_index=True)
        # get full features using the feature selector
        full_feature, full_commit_id, full_test_label = get_combined_df(
            full_split.code,
            full_split.commit_id,
            full_split.label,
            full_split.drop(["code", "label"], axis=1),
        )

        _LOGGER.info(f"Completed loading data for window {i} in {time.time() - start}")

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

        pipeline = make_pipeline_imb(
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

        # init and train model
        start = time.time()
        clf = pipeline.fit(final_train_feature, final_train_label)
        _LOGGER.info(f"fit model finish for window {i} in {time.time() - start}")
        _LOGGER.info(f"Model score: {clf.score(valid_feature, valid_label)}")

        # evaluate on full dataset and later filter for training ecoselekt based on window
        prob = clf.predict_proba(full_feature)[:, 1]
        pred = clf.predict(full_feature)

        # *[OUT]: save results for evaluation
        # *[OUT]: save results for training ecoselekt. Note: saving all but later filter.
        pred_results_df = pd.concat(
            [
                pred_results_df,
                pd.DataFrame(
                    {
                        "window": i,
                        "prob": prob,
                        "pred": pred,
                        "actual": full_test_label,
                        "test_commit": full_commit_id,
                        "model_version": i,
                    }
                ),
            ],
            ignore_index=True,
        )

        pred_results_df.to_csv(
            settings.DATA_DIR / (f"{settings.EXP_ID}_{project_name}_pred_result.csv"),
            index=False,
        )

        # *[OUT]: save model
        pickle.dump(
            clf,
            open(
                settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_model.pkl",
                "wb",
            ),
        )
        _LOGGER.info(f"Finished window {i} for {project_name}")


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            save_model_ckpts(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
