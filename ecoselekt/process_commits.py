import pickle
import warnings
from pathlib import Path

import code_tokenize as ctok
import pandas as pd

from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()
warnings.filterwarnings("ignore")

APACHEJIT_DIR = Path(__file__).parent.parent / "apachejit" / "dataset"
APACHEJIT_COMMITS_DIR = Path(__file__).parent.parent / "apachejit" / "dataset" / "commits"


def tokenize_code(code_line):
    if not code_line:
        return ""
    return " ".join([str(x) for x in ctok.tokenize(code_line, lang="java", syntax_error="ignore")])


def main():
    commits_metrics = pd.read_csv(APACHEJIT_DIR / "apachejit_total.csv")
    for project in APACHEJIT_COMMITS_DIR.iterdir():
        if project.is_dir():
            raise ValueError(f"Unexpected directory {project}")

        project_name = f'{project.name.split("_")[0]}/{project.name.split("_")[1]}'
        project_commit_metrics = commits_metrics[commits_metrics["project"] == project_name]

        code = []
        commit_ids = []
        buggy = []
        commit_msgs = []

        _LOGGER.info(f"Processing {project.name}")

        with open(project, "rb") as f:
            commits = pickle.load(f)

        commits = commits.merge(
            project_commit_metrics[["commit_id", "buggy"]], on="commit_id", how="left"
        )

        _LOGGER.info(f"Loaded {commits.shape[0]} commits")

        for i, commit in enumerate(commits.itertuples()):
            if i % 100 == 0:
                _LOGGER.info(f"Processed {i} commits")

            code_added = [tokenize_code(x[1]) for x in commit.added]
            code_removed = [tokenize_code(x[1]) for x in commit.removed]
            code.append([{"added_code": code_added, "removed_code": code_removed}])
            commit_ids.append(commit.commit_id)
            buggy.append(int(commit.buggy))
            commit_msgs.append("dummy_commit_msg")

        data = [commit_ids, buggy, commit_msgs, code]

        # save the numpy array as a pickle file
        with open(settings.DATA_DIR / project.name, "wb") as f:
            pickle.dump(data, f)
