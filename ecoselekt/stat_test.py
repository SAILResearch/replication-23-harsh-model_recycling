import time

import pandas as pd
from cliffs_delta import cliffs_delta
from scipy.stats import wilcoxon

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def calc_wilcoxon(project_name="activemq"):
    # load baseline evaluation results
    base_eval_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_base_eval_nn.csv"
    )
    base_eval_df["model"] = "base"

    # load ecoselekt evaluation results
    base_eval_all_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_base_eval_nn_all.csv"
    )
    base_eval_all_df["model"] = "all"

    eval_df = pd.concat(
        [base_eval_df, base_eval_all_df],
    )

    p_values = []
    # calculate wilcoxon
    for metric in ["precision", "recall", "f1", "auc", "gmean", "ap", "specifi"]:
        try:
            _, p = wilcoxon(
                eval_df[eval_df["model"] == "all"][metric],
                eval_df[eval_df["model"] == "base"][metric],
            )
            d, res = cliffs_delta(
                eval_df[eval_df["model"] == "all"][metric],
                eval_df[eval_df["model"] == "base"][metric],
            )
            avg_all = eval_df[eval_df["model"] == "all"][metric].mean()
            avg_base = eval_df[eval_df["model"] == "base"][metric].mean()
            avg_diff = avg_all - avg_base
        except ValueError:
            p = 1.0

        is_significant = p < 0.05
        _LOGGER.info(
            f"{project_name} {metric} p-value: {p} significant: {is_significant} diff: {avg_diff} res: {res}"
        )
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

        stats_df.to_csv(settings.DATA_DIR / f"{settings.EXP_ID}_stats_nn_all.csv", index=False)
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
