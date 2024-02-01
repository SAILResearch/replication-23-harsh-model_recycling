import pickle
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

import ecoselekt.utils as utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()
warnings.filterwarnings("ignore")


def get_combined_df_lr(metrics_df):
    metrics_df = metrics_df.sort_values(by="commit_id")

    # !: project is needed to be removed only for apachejit data
    final_df = metrics_df.drop(["commit_id", "project", "author_date", "label"], axis=1)

    return final_df, np.array(metrics_df["commit_id"]), np.array(metrics_df["label"])


def objective_func_lr(k, train_feature, train_label, valid_feature, valid_label):
    smote = SMOTE(random_state=42, k_neighbors=int(np.round(k)), n_jobs=32)
    train_feature_res, train_label_res = smote.fit_resample(train_feature, train_label)

    clf = LogisticRegression(random_state=42, n_jobs=-1)
    clf.fit(train_feature_res, train_label_res)

    prob = clf.predict_proba(valid_feature)[:, 1]
    auc = roc_auc_score(valid_label, prob)

    return -auc


# https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447
# Stepwise Selection
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True

        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)

        if not changed:
            break

    return included


def save_model_ckpts(project_name="activemq"):
    _LOGGER.info(f"Starting checkpoint model training for {project_name}")
    # get commit metrics
    start = time.time()
    commit_metrics = utils.get_apachejit_commit_metrics(project_name)
    commit_metrics = commit_metrics.drop(
        ["fix", "year"],
        axis=1,
    )
    commit_metrics = commit_metrics.fillna(value=0)
    _LOGGER.info(f"Loaded commit metrics in {time.time() - start}")

    # combine train and test code changes
    df = commit_metrics.copy()
    df.rename(columns={"buggy": "label"}, inplace=True)

    _LOGGER.info(f"Loaded total data size for project({project_name}): {df.shape}")
    _LOGGER.info(f"Label distribution for project({project_name}): {Counter(df.label)}")

    # get data splits with sliding window
    df = df.sort_values(by="author_date").reset_index(drop=True)
    windows = utils.get_sliding_windows(df, settings.WINDOW_SIZE, settings.SHIFT)
    _LOGGER.info(f"Loaded windows: {len(windows)}")

    # save windows for evaluation
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows_lr.pkl", "wb") as f:
        pickle.dump(windows, f)

    # Uncomment to check correlation between features
    full_split = pd.concat(windows, ignore_index=True)
    # get full features using the feature selector
    full_feature, full_commit_id, full_test_label = get_combined_df_lr(full_split)

    # # Check correlation between features
    # corr = full_feature.corr()
    # # corr.to_csv(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_corr.csv")
    # # Find highly correlated features
    # corr = corr.abs()
    # corr = corr.unstack()
    # corr = corr.sort_values(ascending=False)
    # corr = corr[corr >= 0.9]
    # corr = corr[corr < 1]
    # corr = pd.DataFrame(corr).reset_index()
    # corr.columns = ["feature1", "feature2", "corr"]
    # _LOGGER.info(f"Correlation between features: {corr}")
    # # corr.to_csv(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_corr.csv")
    # _LOGGER.info(f"Correlation between features saved for {project_name}")
    # # correlation ends here

    full_feature = full_feature.drop(
        ["arexp"], axis=1
    )  # remove arexp as it is highly correlated with aexp

    # train validation split
    final_train_feature, valid_feature, final_train_label, valid_label = train_test_split(
        full_feature, full_test_label, test_size=0.2, random_state=settings.RANDOM_SEED
    )

    # # uncomment to check vif
    # vif_scores = [
    #     variance_inflation_factor(full_feature.values, feature)
    #     for feature in range(len(full_feature.columns))
    # ]
    # # find features with vif > 10
    # vif_features = full_feature.columns[np.array(vif_scores) > 10]
    # _LOGGER.info(f"Features with vif > 10: {vif_features}")
    # # vif ends here

    # # uncomment for stepwise selection
    # result = stepwise_selection(final_train_feature, final_train_label)
    # _LOGGER.info(f"Selected features: {result}")
    # clf = LogisticRegression(random_state=42, n_jobs=-1)
    # clf.fit(final_train_feature[result], final_train_label)
    # # check validation score
    # _LOGGER.info(f"Validation score: {clf.score(valid_feature[result], valid_label)}")

    # # find without stepwise selection
    # clf = LogisticRegression(random_state=42, n_jobs=-1)
    # clf.fit(final_train_feature, final_train_label)
    # # check validation score
    # _LOGGER.info(f"Validation score without: {clf.score(valid_feature, valid_label)}")
    # # stepwise selection ends here


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
