import time

import pandas as pd
from scipy.stats import wilcoxon

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def calc_wilcoxon(project_name="activemq"):
    # load baseline evaluation results
    base_eval_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_base_eval.csv"
    )
    base_eval_df["model"] = "base"

    # load ecoselekt evaluation results
    selekt_eval_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_selekt_eval.csv"
    )
    selekt_eval_df["model"] = "recycled"

    eval_df = pd.concat(
        [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
    )
    eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 1]

    p_values = []
    # calculate wilcoxon
    for metric in ["precision", "recall", "f1", "auc", "gmean", "ap", "specifi"]:
        try:
            _, p = wilcoxon(
                eval_df[eval_df["model"] == "recycled"][metric],
                eval_df[eval_df["model"] == "base"][metric],
            )
        except ValueError:
            p = 1.0
        _LOGGER.info(f"{project_name} {metric} p-value: {p}")
        p_values.append(p)

    return p_values


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        stats_df = pd.DataFrame(
            columns=[
                "project",
                "precision",
                "recall",
                "f1",
                "auc",
                "gmean",
                "ap",
                "specifi",
            ]
        )
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            p_prec, p_rec, p_f1, p_auc, p_gmean, p_ap, p_spec = calc_wilcoxon(project_name)
            stats_df = stats_df.append(
                {
                    "project": project_name,
                    "precision": p_prec,
                    "recall": p_rec,
                    "f1": p_f1,
                    "auc": p_auc,
                    "gmean": p_gmean,
                    "ap": p_ap,
                    "specifi": p_spec,
                },
                ignore_index=True,
            )
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")

        stats_df.to_csv(settings.DATA_DIR / f"{settings.EXP_ID}_stats.csv", index=False)
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
