import os
import time

import pandas as pd
from cliffs_delta import cliffs_delta
from model_recycling_stats.log_util import get_logger
from model_recycling_stats.settings import settings
from scipy.stats import wilcoxon

_LOGGER = get_logger()


def calc_wilcoxon(exp, project_name="activemq", base_model=None):
    # load baseline evaluation results
    base_eval_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_base_eval.csv"
        if base_model is None
        else settings.DATA_DIR
        / exp
        / f"{settings.EXP_ID}_{project_name}_base_eval_{base_model}.csv"
    )
    base_eval_df = pd.read_csv(base_eval_file)
    base_eval_df["model"] = "base"

    # load ecoselekt evaluation results
    selekt_eval_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_selekt_eval.csv"
        if base_model is None
        else settings.DATA_DIR
        / exp
        / f"{settings.EXP_ID}_{project_name}_selekt_eval_{base_model}.csv"
    )
    selekt_eval_df = pd.read_csv(selekt_eval_file)
    selekt_eval_df["model"] = "recycled"

    eval_df = pd.concat(
        [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
    )
    # accomodate for verification latency
    eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

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


def calc_cliffs_delta(exp, project_name="activemq", base_model=None):
    # load baseline evaluation results
    base_eval_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_base_eval.csv"
        if base_model is None
        else settings.DATA_DIR
        / exp
        / f"{settings.EXP_ID}_{project_name}_base_eval_{base_model}.csv"
    )
    base_eval_df = pd.read_csv(base_eval_file)
    base_eval_df["model"] = "base"

    # load ecoselekt evaluation results
    selekt_eval_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_selekt_eval.csv"
        if base_model is None
        else settings.DATA_DIR
        / exp
        / f"{settings.EXP_ID}_{project_name}_selekt_eval_{base_model}.csv"
    )
    selekt_eval_df = pd.read_csv(selekt_eval_file)
    selekt_eval_df["model"] = "recycled"

    eval_df = pd.concat(
        [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
    )
    # accomodate for verification latency
    eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

    delta_values = []
    # calculate cliffs delta
    for metric in ["precision", "recall", "f1", "auc", "gmean", "ap", "specifi"]:
        d, res = cliffs_delta(
            eval_df[eval_df["model"] == "recycled"][metric],
            eval_df[eval_df["model"] == "base"][metric],
        )
        _LOGGER.info(f"{project_name} {metric} delta value: {d}, {res}")
        delta_values.append(res)

    return delta_values


def calc_mean_metrics(exp, project_name="activemq", base_model=None):
    # load baseline evaluation results
    base_eval_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_base_eval.csv"
        if base_model is None
        else settings.DATA_DIR
        / exp
        / f"{settings.EXP_ID}_{project_name}_base_eval_{base_model}.csv"
    )
    base_eval_df = pd.read_csv(base_eval_file)
    base_eval_df["model"] = "base"

    # load ecoselekt evaluation results
    selekt_eval_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_{project_name}_selekt_eval.csv"
        if base_model is None
        else settings.DATA_DIR
        / exp
        / f"{settings.EXP_ID}_{project_name}_selekt_eval_{base_model}.csv"
    )
    selekt_eval_df = pd.read_csv(selekt_eval_file)
    selekt_eval_df["model"] = "recycled"

    eval_df = pd.concat(
        [base_eval_df[base_eval_df["window"] >= settings.MODEL_HISTORY], selekt_eval_df]
    )
    # accomodate for verification latency
    eval_df = eval_df[eval_df["window"] == eval_df["test_split"] - 2]

    mean_values = []
    # calculate mean values
    for metric in ["precision", "recall", "f1", "auc", "gmean", "ap", "specifi"]:
        recycle_mean = eval_df[eval_df["model"] == "recycled"][metric].mean()
        base_mean = eval_df[eval_df["model"] == "base"][metric].mean()
        mean_values.append(recycle_mean - base_mean)

    return mean_values


def calc_stats(exp, base_model=None):
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
    cliff_df = pd.DataFrame(
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
    mean_df = pd.DataFrame(
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
        p_prec, p_rec, p_f1, p_auc, p_gmean, p_ap, p_spec = calc_wilcoxon(
            exp, project_name, base_model
        )
        d_prec, d_rec, d_f1, d_auc, d_gmean, d_ap, d_spec = calc_cliffs_delta(
            exp, project_name, base_model
        )
        m_prec, m_rec, m_f1, m_auc, m_gmean, m_ap, m_spec = calc_mean_metrics(
            exp, project_name, base_model
        )
        stats_df = pd.concat(
            [
                stats_df,
                pd.DataFrame(
                    {
                        "project": [project_name],
                        "precision": [p_prec],
                        "recall": [p_rec],
                        "f1": [p_f1],
                        "auc": [p_auc],
                        "gmean": [p_gmean],
                        "ap": [p_ap],
                        "specifi": [p_spec],
                    }
                ),
            ],
            ignore_index=True,
        )

        cliff_df = pd.concat(
            [
                cliff_df,
                pd.DataFrame(
                    {
                        "project": [project_name],
                        "precision": [d_prec],
                        "recall": [d_rec],
                        "f1": [d_f1],
                        "auc": [d_auc],
                        "gmean": [d_gmean],
                        "ap": [d_ap],
                        "specifi": [d_spec],
                    }
                ),
            ],
            ignore_index=True,
        )

        mean_df = pd.concat(
            [
                mean_df,
                pd.DataFrame(
                    {
                        "project": [project_name],
                        "precision": [m_prec],
                        "recall": [m_rec],
                        "f1": [m_f1],
                        "auc": [m_auc],
                        "gmean": [m_gmean],
                        "ap": [m_ap],
                        "specifi": [m_spec],
                    }
                ),
            ],
            ignore_index=True,
        )

        _LOGGER.info(f"Finished {project_name} in {time.time() - start}")

    stats_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_stats.csv"
        if base_model is None
        else settings.DATA_DIR / exp / f"{settings.EXP_ID}_stats_{base_model}.csv"
    )
    stats_df.to_csv(stats_file, index=False)

    cliff_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_cliff.csv"
        if base_model is None
        else settings.DATA_DIR / exp / f"{settings.EXP_ID}_cliff_{base_model}.csv"
    )
    cliff_df.to_csv(cliff_file, index=False)

    mean_file = (
        settings.DATA_DIR / exp / f"{settings.EXP_ID}_mean.csv"
        if base_model is None
        else settings.DATA_DIR / exp / f"{settings.EXP_ID}_mean_{base_model}.csv"
    )
    mean_df.to_csv(mean_file, index=False)


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for exp in settings.EXPERIMENTS:
            _LOGGER.info(f"Starting {exp}")
            calc_stats(exp)
            calc_stats(exp, "lr")
            calc_stats(exp, "nn")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
