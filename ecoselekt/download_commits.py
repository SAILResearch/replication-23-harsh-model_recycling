import warnings
from pathlib import Path

import pandas as pd
from pydriller import Repository

from ecoselekt.log_util import get_logger

_LOGGER = get_logger()
warnings.filterwarnings("ignore")

APACHEJIT_DIR = Path(__file__).parent.parent / "apachejit" / "dataset"


def hadoop_save():
    _LOGGER.info("Starting collecting commits")
    df = pd.read_csv(APACHEJIT_DIR / "apachejit_total.csv")
    df = df[df["project"].isin(["apache/hadoop-hdfs", "apache/hadoop-mapreduce"])]
    count = 0
    for project, _ in df.groupby("project"):
        _LOGGER.info(f"Collecting commits for {project}")
        commits_df = pd.DataFrame(columns=["commit_id", "added", "removed"])
        commits = df[df["project"] == project]["commit_id"].tolist()
        for commit in Repository(
            "https://github.com/apache/hadoop.git", only_commits=commits
        ).traverse_commits():
            added = []
            removed = []
            for modified in commit.modified_files:
                added.extend(modified.diff_parsed["added"])
                removed.extend(modified.diff_parsed["deleted"])
            commits_df = commits_df.append(
                {
                    "commit_id": commit.hash,
                    "added": added,
                    "removed": removed,
                },
                ignore_index=True,
            )
            count += 1
            if count % 1000 == 0:
                _LOGGER.info(f"Processed {count} commits")

        for commit in Repository(
            f"https://github.com/{project}.git", only_commits=commits
        ).traverse_commits():
            added = []
            removed = []
            for modified in commit.modified_files:
                added.extend(modified.diff_parsed["added"])
                removed.extend(modified.diff_parsed["deleted"])
            commits_df = commits_df.append(
                {
                    "commit_id": commit.hash,
                    "added": added,
                    "removed": removed,
                },
                ignore_index=True,
            )
            count += 1
            if count % 1000 == 0:
                _LOGGER.info(f"Processed {count} commits")
        commits_df.to_pickle(APACHEJIT_DIR / f'{project.replace("/", "_")}_commits.pkl')
        _LOGGER.info(f"Collected commits for {project}")


def main():
    _LOGGER.info("Starting collecting commits")
    df = pd.read_csv(APACHEJIT_DIR / "apachejit_total.csv")
    count = 0
    for project, _ in df.groupby("project"):
        _LOGGER.info(f"Collecting commits for {project}")
        commits_df = pd.DataFrame(columns=["commit_id", "added", "removed"])
        commits = df[df["project"] == project]["commit_id"].tolist()
        for commit in Repository(
            f"https://github.com/{project}.git", only_commits=commits
        ).traverse_commits():
            added = []
            removed = []
            for modified in commit.modified_files:
                added.extend(modified.diff_parsed["added"])
                removed.extend(modified.diff_parsed["deleted"])
            commits_df = commits_df.append(
                {
                    "commit_id": commit.hash,
                    "added": added,
                    "removed": removed,
                },
                ignore_index=True,
            )
            count += 1
            if count % 1000 == 0:
                _LOGGER.info(f"Processed {count} commits")
        commits_df.to_pickle(APACHEJIT_DIR / f'{project.replace("/", "_")}_commits.pkl')
        _LOGGER.info(f"Collected commits for {project}")


if __name__ == "__main__":
    main()
    hadoop_save()
