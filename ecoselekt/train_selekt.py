import pickle
import time

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

import ecoselekt.utils as utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_models import get_combined_df

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

        train_feature, train_commit_id, train_label = get_combined_df(
            split.code,
            split.commit_id,
            split.label,
            split.drop(["code", "label"], axis=1),
        )

        # train kNN model
        start = time.time()

        tokenize_ = make_column_transformer(
            (CountVectorizer(min_df=3, ngram_range=(1, 1)), "code"),
            remainder="passthrough",
        )
        preprocess_ = make_pipeline(
            tokenize_,
            SelectKBest(chi2, k=settings.K_FEATURES),
        )

        train_feature = preprocess_.fit_transform(train_feature, train_label)

        # save preprocess pipeline
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_preprocess_knn.pkl", "wb"
        ) as f:
            pickle.dump(preprocess_, f)

        clf = NearestNeighbors(n_neighbors=settings.MAX_KNEIGHBOURS, algorithm=settings.KNN_ALGO)
        clf.fit(train_feature, train_label)
        _LOGGER.info(f"Trained kNN model in {time.time() - start}")

        # save kNN model
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_knn.pkl", "wb"
        ) as f:
            pickle.dump(clf, f)
        _LOGGER.info(f"Saved kNN model for {project_name} for window {i}")


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
