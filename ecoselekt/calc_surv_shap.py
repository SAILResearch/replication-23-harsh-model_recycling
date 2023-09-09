import time

import pandas as pd

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def calc_drift(project_name="activemq"):
    k = 20
    mean_df = pd.read_csv(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shap_mean.csv")
    std_df = pd.read_csv(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shap_std.csv")

    drift_results_df = pd.DataFrame(
        columns=[
            "project",
            "mi",
            "mj",
            "surv_rate",
            "fi",
            "fj",
        ]
    )

    for i in range(mean_df["window"].max() + 1):
        t_mean_df = mean_df[mean_df["window"] == i].T
        t_std_df = std_df[std_df["window"] == i].T
        t_mean_df.drop([t_mean_df.index[0], t_mean_df.index[1]], inplace=True)
        t_std_df.drop([t_std_df.index[0], t_std_df.index[1]], inplace=True)
        t_mean_df.columns = ["mean"]
        t_std_df.columns = ["std"]

        temp_df = pd.concat([t_mean_df, t_std_df], axis=1)
        temp_df["mean"] = abs(temp_df["mean"])
        temp_df.fillna(0, inplace=True)

        # find top 10 features with highest mean + std
        topk_features = (
            temp_df.sort_values(by=["mean", "std"], ascending=False).head(k).index.tolist()
        )

        for j in range(i, mean_df["window"].max() + 1):
            t_mean_df = mean_df[mean_df["window"] == j].T
            t_std_df = std_df[std_df["window"] == j].T
            t_mean_df.drop([t_mean_df.index[0], t_mean_df.index[1]], inplace=True)
            t_std_df.drop([t_std_df.index[0], t_std_df.index[1]], inplace=True)
            t_mean_df.columns = ["mean"]
            t_std_df.columns = ["std"]

            temp_df = pd.concat([t_mean_df, t_std_df], axis=1)
            temp_df["mean"] = abs(temp_df["mean"])
            temp_df.fillna(0, inplace=True)

            # find top 10 features with highest mean + std
            topk_features2 = (
                temp_df.sort_values(by=["mean", "std"], ascending=False).head(k).index.tolist()
            )

            # find survival rate of top 10 features
            surv_rate = len(set(topk_features).intersection(set(topk_features2))) / len(
                set(topk_features2)
            )

            drift_results_df = pd.concat(
                [
                    drift_results_df,
                    pd.DataFrame(
                        [
                            [
                                project_name,
                                i,
                                j,
                                surv_rate,
                                topk_features,
                                topk_features2,
                            ]
                        ],
                        columns=drift_results_df.columns,
                    ),
                ],
                ignore_index=True,
            )

    drift_results_df.to_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_shapdrift.csv", index=False
    )


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            calc_drift(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
