import numpy as np
import pandas as pd
from model_recycling_stats.log_util import get_logger
from model_recycling_stats.settings import settings

_LOGGER = get_logger()


def calc_median_improvements():
    rf_experiments = ["ms-exp8", "mr-exp4", "mst-exp8", "mv-exp2"]
    lr_experiments = ["ms-exp8", "mst-exp8", "mv-exp2"]
    nn_experiments = ["mst-exp8"]

    precision, recall, gmean, auc, f1 = [], [], [], [], []

    for exp in rf_experiments:
        # load baseline evaluation results
        base_eval_df = pd.concat(
            [
                pd.read_csv(
                    settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_base_eval.csv"
                )
                for project_name in settings.PROJECTS
            ]
        )
        base_eval_df["model"] = "base"

        # load ecoselekt evaluation results
        selekt_eval_df = pd.concat(
            [
                pd.read_csv(
                    settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_selekt_eval.csv"
                )
                for project_name in settings.PROJECTS
            ]
        )
        selekt_eval_df["model"] = "recycled"

        eval_df = pd.concat(
            [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
        )
        eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

        for metric in ["precision", "recall", "gmean", "auc", "f1"]:
            for project_name in settings.PROJECTS:
                median_improvements = np.median(
                    eval_df[
                        (eval_df["model"] == "recycled") & (eval_df["project"] == project_name)
                    ][metric].values
                    - eval_df[(eval_df["model"] == "base") & (eval_df["project"] == project_name)][
                        metric
                    ].values
                )
                _LOGGER.info(
                    f"{exp}, {project_name}: {metric} median improvement: {median_improvements*100:.2f}",
                )
                if metric == "precision":
                    precision.append(median_improvements * 100)
                elif metric == "recall":
                    recall.append(median_improvements * 100)
                elif metric == "gmean":
                    gmean.append(median_improvements * 100)
                elif metric == "auc":
                    auc.append(median_improvements * 100)
                elif metric == "f1":
                    f1.append(median_improvements * 100)
    for exp in lr_experiments:
        # load baseline evaluation results
        base_eval_df = pd.concat(
            [
                pd.read_csv(
                    settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_base_eval_lr.csv"
                )
                for project_name in settings.PROJECTS
            ]
        )
        base_eval_df["model"] = "base"

        # load ecoselekt evaluation results
        selekt_eval_df = pd.concat(
            [
                pd.read_csv(
                    settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_selekt_eval_lr.csv"
                )
                for project_name in settings.PROJECTS
            ]
        )
        selekt_eval_df["model"] = "recycled"

        eval_df = pd.concat(
            [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
        )
        eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

        for metric in ["precision", "recall", "gmean", "auc", "f1"]:
            for project_name in settings.PROJECTS:
                median_improvements = np.median(
                    eval_df[
                        (eval_df["model"] == "recycled") & (eval_df["project"] == project_name)
                    ][metric].values
                    - eval_df[(eval_df["model"] == "base") & (eval_df["project"] == project_name)][
                        metric
                    ].values
                )
                _LOGGER.info(
                    f"{exp}, {project_name}: {metric} median improvement: {median_improvements*100:.2f}",
                )
                if metric == "precision":
                    precision.append(median_improvements * 100)
                elif metric == "recall":
                    recall.append(median_improvements * 100)
                elif metric == "gmean":
                    gmean.append(median_improvements * 100)
                elif metric == "auc":
                    auc.append(median_improvements * 100)
                elif metric == "f1":
                    f1.append(median_improvements * 100)
    for exp in nn_experiments:
        # load baseline evaluation results
        base_eval_df = pd.concat(
            [
                pd.read_csv(
                    settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_base_eval_nn.csv"
                )
                for project_name in settings.PROJECTS
            ]
        )
        base_eval_df["model"] = "base"

        # load ecoselekt evaluation results
        selekt_eval_df = pd.concat(
            [
                pd.read_csv(
                    settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_selekt_eval_nn.csv"
                )
                for project_name in settings.PROJECTS
            ]
        )
        selekt_eval_df["model"] = "recycled"

        eval_df = pd.concat(
            [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
        )
        eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

        for metric in ["precision", "recall", "gmean", "auc", "f1"]:
            for project_name in settings.PROJECTS:
                median_improvements = np.median(
                    eval_df[
                        (eval_df["model"] == "recycled") & (eval_df["project"] == project_name)
                    ][metric].values
                    - eval_df[(eval_df["model"] == "base") & (eval_df["project"] == project_name)][
                        metric
                    ].values
                )
                _LOGGER.info(
                    f"{exp}, {project_name}: {metric} median improvement: {median_improvements*100:.2f}",
                )
                if metric == "precision":
                    precision.append(median_improvements * 100)
                elif metric == "recall":
                    recall.append(median_improvements * 100)
                elif metric == "gmean":
                    gmean.append(median_improvements * 100)
                elif metric == "auc":
                    auc.append(median_improvements * 100)
                elif metric == "f1":
                    f1.append(median_improvements * 100)

    print("min max precision", min(precision), max(precision))
    print("min max recall", min(recall), max(recall))
    print("min max gmean", min(gmean), max(gmean))
    print("min max auc", min(auc), max(auc))
    print("min max f1", min(f1), max(f1))


def calc_median_improvements_per_strategy():
    # lr model selection, prec, auc and f1 improv: exp 3, 26, 4, 27
    # lr model selection, recall degrad: exp 1, 20, 2, 28
    # rf model reuse, recall, g-mean and f1 improv and degrad: exp 5, 7, 6, 8
    # rf model stacking, recall and g-mean improv: exp 10, 30, 19, 11, 12, 29, 21
    # lr model stacking, precision, g-mean, auc and f1 improv and
    # recall degrad: exp 9, 19, 11, 17, 12, 29, 21
    # nn model stacking, g-mean, auc and f1 improv: exp 9, 19, 11, 17, 21
    # rf model voting, 13, 14
    # lr model voting, 13, 14, 24
    exp = "exp24"
    model = "lr"
    if model == "rf":
        base_eval_str = "base_eval.csv"
        selekt_eval_str = "selekt_eval.csv"
    else:
        base_eval_str = f"base_eval_{model}.csv"
        selekt_eval_str = f"selekt_eval_{model}.csv"

    precision, recall, gmean, auc, f1 = [], [], [], [], []

    # load baseline evaluation results
    base_eval_df = pd.concat(
        [
            pd.read_csv(
                settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_{base_eval_str}"
            )
            for project_name in settings.PROJECTS
        ]
    )
    base_eval_df["model"] = "base"

    # load ecoselekt evaluation results
    selekt_eval_df = pd.concat(
        [
            pd.read_csv(
                settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_{selekt_eval_str}"
            )
            for project_name in settings.PROJECTS
        ]
    )
    selekt_eval_df["model"] = "recycled"

    eval_df = pd.concat(
        [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
    )
    eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

    for metric in ["precision", "recall", "gmean", "auc", "f1"]:
        for project_name in settings.PROJECTS:
            median_improvements = np.median(
                eval_df[(eval_df["model"] == "recycled") & (eval_df["project"] == project_name)][
                    metric
                ].values
                - eval_df[(eval_df["model"] == "base") & (eval_df["project"] == project_name)][
                    metric
                ].values
            )
            if metric == "precision":
                precision.append(median_improvements * 100)
            elif metric == "recall":
                recall.append(median_improvements * 100)
            elif metric == "gmean":
                gmean.append(median_improvements * 100)
            elif metric == "auc":
                auc.append(median_improvements * 100)
            elif metric == "f1":
                f1.append(median_improvements * 100)
    print(f"{exp} {model}")
    print("min max precision", min(precision), max(precision))
    print("min max recall", min(recall), max(recall))
    print("min max gmean", min(gmean), max(gmean))
    print("min max auc", min(auc), max(auc))
    print("min max f1", min(f1), max(f1))


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        calc_median_improvements()
        calc_median_improvements_per_strategy()
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
