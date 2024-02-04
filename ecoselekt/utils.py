import pickle
import re
from pathlib import Path

import pandas as pd

import ecoselekt.consts as consts

DATA_DIR = Path(__file__).parent / "data"


def _load_data(data):
    commit_id = data[0]
    label = data[1]
    all_code_change = data[3]

    def _preprocess_code_line(code_line):
        code_line = (
            code_line.replace("(", " ")
            .replace(")", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace(".", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace(",", " ")
            .replace(" _ ", "_")
        )
        code_line = re.sub("``.*``", "<STR>", code_line)
        code_line = re.sub("'.*'", "<STR>", code_line)
        code_line = re.sub('".*"', "<STR>", code_line)
        code_line = re.sub("\d+", "<NUM>", code_line)

        # remove python common tokens
        new_code = ""

        for tok in code_line.split():
            if tok not in consts.PYTHON_COMMON_TOKENS:
                new_code = new_code + tok + " "

        return new_code.strip()

    all_added_code = [
        " \n ".join(
            list(
                set(
                    [
                        _preprocess_code_line(code_line)
                        for ch in code_change
                        if len(ch["added_code"]) > 0
                        for code_line in ch["added_code"]
                        # remove comments
                        if not code_line.startswith("#")
                    ]
                )
            )
        )
        for code_change in all_code_change
    ]
    all_removed_code = [
        " \n ".join(
            list(
                set(
                    [
                        _preprocess_code_line(code_line)
                        for ch in code_change
                        if len(ch["removed_code"]) > 0
                        for code_line in ch["removed_code"]
                        # remove comments
                        if not code_line.startswith("#")
                    ]
                )
            )
        )
        for code_change in all_code_change
    ]

    combined_code = [
        added_code + " " + removed_code
        for added_code, removed_code in zip(all_added_code, all_removed_code)
    ]

    return combined_code, commit_id, label


def prep_train_data(project_name):
    train_data = pickle.load(open(DATA_DIR / f"{project_name}_train.pkl", "rb"))

    # project_dict = pickle.load(open(DATA_DIR / f"{project_name}_dict.pkl", "rb"))[1]
    # print(project_dict)

    # max_idx = np.max(list(project_dict.values()))
    # print(max_idx)

    # project_dict["<STR>"] = max_idx + 1
    # print(project_dict)

    (
        train_combined_code,
        train_commit_id,
        train_label,
    ) = _load_data(train_data)

    return (
        train_combined_code,
        train_commit_id,
        train_label,
    )


def prep_test_data(project_name):
    test_data = pickle.load(open(DATA_DIR / f"{project_name}_test.pkl", "rb"))

    (
        test_combined_code,
        test_commit_id,
        test_label,
    ) = _load_data(test_data)

    return (
        test_combined_code,
        test_commit_id,
        test_label,
    )


def prep_apachejit_data(project_name):
    data = pickle.load(open(DATA_DIR / f"apache_{project_name}_commits.pkl", "rb"))

    return _load_data(data)


def get_apachejit_commit_metrics(project_name):
    df = pd.read_csv(DATA_DIR / "apachejit_total.csv")
    return df[df["project"] == f"apache/{project_name}"].reset_index(drop=True)


def get_sliding_windows(df, window_size=1000, shift=200):
    windows = []
    for i in range(0, ((df.shape[0] - window_size) // shift) + 1):
        windows.append(df.iloc[i * shift : i * shift + window_size])
    return windows
