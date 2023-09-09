import numpy as np
import pandas as pd

from model_recycling_stats.log_util import get_logger
from model_recycling_stats.settings import settings

_LOGGER = get_logger()


def calc_median_improvements():
    experiments = ["ms-exp4", "mr-exp4", "mst-exp8", "mv-exp2"]

    recall, gmean, auc = [], [], []
    for exp in experiments:
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
        eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 1]

        for metric in ["recall", "gmean", "auc"]:
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
                if metric == "recall":
                    recall.append(median_improvements * 100)
                elif metric == "gmean":
                    gmean.append(median_improvements * 100)
                elif metric == "auc":
                    auc.append(median_improvements * 100)

    recall = [rec for rec in recall if rec > 0]
    gmean = [g for g in gmean if g > 0]
    auc = [a for a in auc if a > 0]

    print("min max recall", min(recall), max(recall))
    print("min max gmean", min(gmean), max(gmean))
    print("min max auc", min(auc), max(auc))


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        calc_median_improvements()
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
