import gc
import math
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import ecoselekt.consts as consts
import ecoselekt.dnn.cc2vec_utils as cc2vec_utils
import ecoselekt.dnn.deepjit_utils as deepjit_utils
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

DATA_DIR = Path(__file__).parent / "data"

random.seed(settings.RANDOM_SEED)

_LOGGER = get_logger()


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


def mini_batches(
    X_added_code, X_removed_code, Y, mini_batch_size=64, seed=settings.RANDOM_SEED, shuffled=True
):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    if shuffled == True:
        permutation = list(np.random.permutation(m))
        shuffled_X_added = X_added_code[permutation, :, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :, :]

        if len(Y.shape) == 1:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code
        shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size)
    )  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_added = shuffled_X_added[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :
        ]
        mini_batch_X_removed = shuffled_X_removed[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :
        ]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[
                k * mini_batch_size : k * mini_batch_size + mini_batch_size, :
            ]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_DExtended(X_ftr, X_msg, X_code, Y, mini_batch_size=64, seed=settings.RANDOM_SEED):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_ftr, shuffled_X_msg, shuffled_X_code, shuffled_Y = X_ftr, X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_ftr = shuffled_X_ftr[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :
        ]
        mini_batch_X_msg = shuffled_X_msg[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :
        ]
        mini_batch_X_code = shuffled_X_code[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :
        ]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[
                k * mini_batch_size : k * mini_batch_size + mini_batch_size, :
            ]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_ftr = shuffled_X_ftr[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size : m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_update_DExtended(X_ftr, X_code, Y, mini_batch_size=64, seed=settings.RANDOM_SEED):
    m = X_ftr.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_ftr, shuffled_X_code, shuffled_Y = X_ftr, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # change mini_batch_size in case pos is less than mini_batch_size / 2
    # since this is imbalanced data
    if len(Y_pos) < int(mini_batch_size / 2):
        _LOGGER.info(f"pos is less than mini_batch_size / 2: {len(Y_pos)}")
        mini_batch_size = len(Y_pos) * 2

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2))
            + random.sample(Y_neg, int(mini_batch_size / 2))
        )
        mini_batch_X_ftr = shuffled_X_ftr[indexes]
        mini_batch_X_code = shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
