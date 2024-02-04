import pickle
import time
from collections import Counter

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

import ecoselekt.utils as utils
from ecoselekt.dnn.deepjit import DeepJITExtended
from ecoselekt.dnn.deepjit_utils import optim_padding_code, optim_padding_msg
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings

_LOGGER = get_logger()


def get_combined_df(code_commit, message, commit_id, label, metrics_df):
    code_df = pd.DataFrame(
        {"code": code_commit, "commit_id": commit_id, "label": label, "message": message}
    )

    code_df = code_df.sort_values(by="commit_id")

    metrics_df = metrics_df.sort_values(by="commit_id")
    # !: project is needed to be removed only for apachejit data
    metrics_df = metrics_df.drop(["commit_id", "project", "author_date"], axis=1)

    final_features = pd.concat([code_df["code"], code_df["message"], metrics_df], axis=1)

    return final_features, np.array(code_df["commit_id"]), np.array(code_df["label"])


def save_model_ckpts(project_name="activemq"):
    _LOGGER.info(f"Starting checkpoint model training for {project_name}")
    start = time.time()
    # get test train code changes
    (all_code, all_commit, all_label, all_message) = utils.prep_apachejit_data(project_name)
    _LOGGER.info(f"Loaded code changes in {time.time() - start}")

    # get commit metrics
    start = time.time()
    commit_metrics = utils.get_apachejit_commit_metrics(project_name)
    # !: should we keep time information?
    commit_metrics = commit_metrics.drop(
        ["fix", "year", "buggy"],
        axis=1,
    )
    commit_metrics = commit_metrics.fillna(value=0)
    _LOGGER.info(f"Loaded commit metrics in {time.time() - start}")

    # TODO: add deleted and added tokens in code?
    # TODO: do we add commit message?
    # TODO: do we selectkbest like others?
    # combine train and test code changes
    df = pd.DataFrame(
        {
            "commit_id": all_commit,
            "code": all_code,
            "label": all_label,
            "message": all_message,
        }
    )

    # merge commit metrics to train code
    df = pd.merge(df, commit_metrics, on="commit_id")
    _LOGGER.info(f"Loaded total data size for project({project_name}): {df.shape}")
    _LOGGER.info(f"Label distribution for project({project_name}): {Counter(df.label)}")

    # get data splits with sliding window
    df = df.sort_values(by="author_date").reset_index(drop=True)
    windows = utils.get_sliding_windows(df, settings.WINDOW_SIZE, settings.SHIFT)
    _LOGGER.info(f"Loaded windows: {len(windows)}")

    # save windows for evaluation
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "wb") as f:
        pickle.dump(windows, f)

    pred_results_df = pd.DataFrame(
        columns=["window", "prob", "pred", "actual", "test_commit", "model_version"]
    )

    # ignore last few windows (`settings.TEST_SIZE`) since there is no test data if we use it
    for i, window in enumerate(windows[: -settings.C_TEST_WINDOWS]):
        _LOGGER.info(f"Starting window {i} for {project_name}")
        start = time.time()

        # randomly shuffle the window to get random validation set
        split = window.sample(frac=1, random_state=settings.RANDOM_SEED, ignore_index=True)

        train_feature, train_commit_id, train_label = get_combined_df(
            split.code,
            split.message,
            split.commit_id,
            split.label,
            split.drop(["code", "label", "message"], axis=1),
        )
        _LOGGER.info(f"Train feature shape: {train_feature.shape}")

        # train validation split
        final_train_feature, valid_feature, final_train_label, valid_label = train_test_split(
            train_feature, train_label, test_size=0.2, random_state=settings.RANDOM_SEED
        )
        _LOGGER.info(f"New train feature shape: {train_feature.shape}")

        # z-score normalization
        scaler = StandardScaler()
        scaler.fit(final_train_feature.drop(["code", "message"], axis=1).values)
        # *[OUT]: save scaler
        pickle.dump(scaler, open(settings.DATA_DIR / f"{project_name}_w{i}_scaler.pkl", "wb"))

        full_split = pd.concat(windows, ignore_index=True)
        _LOGGER.info(f"Full split shape: {full_split.shape}")
        # dedup by commit id
        full_split = full_split.drop_duplicates(subset=["commit_id"])
        _LOGGER.info(f"Full split shape after dedup: {full_split.shape}")

        # get full features using the feature selector
        full_feature, full_commit_id, full_test_label = get_combined_df(
            full_split.code,
            full_split.message,
            full_split.commit_id,
            full_split.label,
            full_split.drop(["code", "label", "message"], axis=1),
        )

        _LOGGER.info(f"Completed loading data for window {i} in {time.time() - start}")

        # prepare vocabulary for code and save it for use in transformer
        vectorizer = CountVectorizer(min_df=3, ngram_range=(1, 1))
        vectorizer.fit(final_train_feature.code)
        dict_code = vectorizer.vocabulary_
        dict_code["<NULL>"] = len(dict_code)
        # *[OUT]: save dict_code
        pickle.dump(dict_code, open(settings.DATA_DIR / f"{project_name}_w{i}_dict_code.pkl", "wb"))

        final_pad_code = optim_padding_code(project_name, i, final_train_feature.code, dict_code)

        valid_pad_code = optim_padding_code(project_name, i, valid_feature.code, dict_code)

        # prepare vocabulary for message and save it for use in transformer
        msg_vectorizer = CountVectorizer(min_df=3, ngram_range=(1, 1))
        msg_vectorizer.fit(final_train_feature.message)
        dict_msg = msg_vectorizer.vocabulary_
        dict_msg["<NULL>"] = len(dict_msg)
        # *[OUT]: save dict_msg
        pickle.dump(dict_msg, open(settings.DATA_DIR / f"{project_name}_w{i}_dict_msg.pkl", "wb"))

        final_pad_msg = optim_padding_msg(project_name, i, final_train_feature.message, dict_msg)

        valid_pad_msg = optim_padding_msg(project_name, i, valid_feature.message, dict_msg)

        # create and train the defect model
        start = time.time()
        model = DeepJITExtended(vocab_msg=len(dict_msg), vocab_code=len(dict_code))
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.LR)

        criterion = nn.BCELoss()
        for epoch in range(1, settings.NUM_EPOCHS + 1):
            total_loss = 0
            # building batches for training model
            batches = utils.mini_batches_update_DExtended(
                X_ftr=scaler.transform(
                    final_train_feature.drop(["code", "message"], axis=1).values
                ),
                X_msg=final_pad_msg,
                X_code=final_pad_code,
                Y=final_train_label,
                mini_batch_size=settings.BATCH_SIZE,
            )
            model.train()
            for bi, (batch) in enumerate(tqdm(batches)):
                X, X_msg, X_code, labels = batch
                X, X_msg, X_code, labels = (
                    torch.tensor(X).float(),
                    torch.tensor(X_msg).long(),
                    torch.tensor(X_code).long(),
                    torch.tensor(labels).float(),
                )

                optimizer.zero_grad()
                predict = model.forward(X, X_msg, X_code)
                loss = criterion(predict, labels)
                total_loss += loss
                loss.backward()
                optimizer.step()

            # calculate validation loss
            model.eval()
            predict = model.forward(
                torch.tensor(
                    scaler.transform(valid_feature.drop(["code", "message"], axis=1).values)
                ).float(),
                torch.tensor(valid_pad_msg).long(),
                torch.tensor(valid_pad_code).long(),
            )
            loss = criterion(predict, torch.tensor(valid_label).float())

            print(
                "Epoch %i / %i -- Total loss: %f -- Validation loss: %f"
                % (epoch, settings.NUM_EPOCHS, total_loss, loss)
            )

        # with torch.no_grad():
        #     # Plot the ROC curve
        #     y_pred = model.forward(
        #         torch.tensor(scaler.transform(valid_feature.drop(["code"], axis=1).values)).float(),
        #         torch.tensor(valid_pad_code).long(),
        #     )
        #     fpr, tpr, thresholds = roc_curve(valid_label, y_pred)
        #     plt.plot(fpr, tpr)  # ROC curve = TPR vs FPR
        #     plt.title("Receiver Operating Characteristics")
        #     plt.xlabel("False Positive Rate")
        #     plt.ylabel("True Positive Rate")
        #     plt.show()
        # calculate validation loss
        with torch.no_grad():
            predict = model.forward(
                torch.tensor(
                    scaler.transform(valid_feature.drop(["code", "message"], axis=1).values)
                ).float(),
                torch.tensor(valid_pad_msg).long(),
                torch.tensor(valid_pad_code).long(),
            )
            loss = criterion(predict, torch.tensor(valid_label).float())
            print("Validation loss: %f" % (loss))
            predict = predict.detach().numpy()
            print("Validation auc score: %f" % (roc_auc_score(valid_label, predict)))
            pred_labels = np.where(predict > 0.5, 1, 0)
            print("Validation f1 score: %f" % (f1_score(valid_label, pred_labels)))

        _LOGGER.info(f"fit model finish for window {i} in {time.time() - start}")

        full_pad_code = optim_padding_code(project_name, i, full_feature.code, dict_code)
        full_pad_msg = optim_padding_msg(project_name, i, full_feature.message, dict_msg)
        _LOGGER.info(f"Shape of full_pad_code: {full_pad_code.shape}")

        with torch.no_grad():
            # evaluate on full dataset and later filter for training ecoselekt based on window
            full_predict = model.forward(
                torch.tensor(
                    scaler.transform(full_feature.drop(["code", "message"], axis=1).values)
                ).float(),
                torch.tensor(full_pad_msg).long(),
                torch.tensor(full_pad_code).long(),
            )
            prob = full_predict.detach().numpy()
            pred = np.where(full_predict > 0.5, 1, 0)

        _LOGGER.info(f"Shape of prob: {prob.shape}")

        # *[OUT]: save results for evaluation
        # *[OUT]: save results for training ecoselekt. Note: saving all but later filter.
        pred_results_df = pd.concat(
            [
                pred_results_df,
                pd.DataFrame(
                    {
                        "window": i,
                        "prob": prob,
                        "pred": pred,
                        "actual": full_test_label,
                        "test_commit": full_commit_id,
                        "model_version": i,
                    }
                ),
            ],
            ignore_index=True,
        )

        pred_results_df.to_csv(
            settings.DATA_DIR / (f"{settings.EXP_ID}_{project_name}_pred_result_nn_all.csv"),
            index=False,
        )

        # *[OUT]: save model
        torch.save(
            model.state_dict(),
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_model_nn_all.pt",
        )
        _LOGGER.info(f"Finished window {i} for {project_name}")


def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            save_model_ckpts(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")


def main_parallel():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        Parallel(n_jobs=2)(
            delayed(save_model_ckpts)(project_name) for project_name in settings.PROJECTS
        )
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
