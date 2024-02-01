import pickle
import time

import numpy as np
import pandas as pd
import torch

from ecoselekt.dnn.deepjit import DeepJITExtended
from ecoselekt.dnn.deepjit_utils import optim_padding_code
from ecoselekt.log_util import get_logger
from ecoselekt.settings import settings
from ecoselekt.train_models import get_combined_df

_LOGGER = get_logger()


def inference_selekt(project_name):
    _LOGGER.info(f"Inferencing selekt for {project_name}")
    start = time.time()
    # load sliding windows splits
    with open(settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_windows.pkl", "rb") as f:
        windows = pickle.load(f)

    pred_result_df = pd.read_csv(
        settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_pred_result_nn.csv"
    )

    _LOGGER.info(
        f"Project: {project_name} with {len(windows)} windows loaded in {time.time() - start}"
    )

    inf_performance_df = pd.DataFrame(
        columns=["window", "commit_id", "eco_pred_time", "base_pred_time"]
    )

    for i in range(settings.MODEL_HISTORY, len(windows) - settings.C_TEST_WINDOWS):
        start = time.time()
        split = pd.concat(
            [windows[j][-settings.SHIFT :] for j in range(i + 1, i + 1 + settings.F_TEST_WINDOWS)],
            ignore_index=True,
        )
        if settings.TEST_SIZE % settings.SHIFT != 0:
            split = pd.concat(
                [
                    split,
                    windows[i + 1 + settings.F_TEST_WINDOWS][
                        -settings.SHIFT : (settings.TEST_SIZE % settings.SHIFT) - settings.SHIFT
                    ],
                ],
                ignore_index=True,
            )
        _LOGGER.info(f"Shape of test data: {split.shape}")

        test_feature, test_commit_id, new_test_label = get_combined_df(
            split.code,
            split.commit_id,
            split.label,
            split.drop(["code", "label"], axis=1),
        )

        all_pred_dfs = []
        # load all future model predictions
        for j in range(i - settings.MODEL_HISTORY, i + 1):
            temp_df = pred_result_df[pred_result_df["window"] == j].copy()
            temp_df.rename(columns={"test_commit": "commit_id"}, inplace=True)
            temp_df.drop("window", axis=1, inplace=True)
            # filter out commit ids that are not in the current window
            temp_df = temp_df[temp_df["commit_id"].isin(split.commit_id)]
            all_pred_dfs.append(temp_df)

        pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        _LOGGER.info(f"Prediction df shape: {pred_df.shape}")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_stat_models_nn.pkl",
            "rb",
        ) as f:
            stat_models = pickle.load(f)

        pred_df = pred_df[pred_df["model_version"].isin(stat_models)].reset_index(drop=True)
        # add models probabilities as features
        for model_version in stat_models:
            prob_df = (
                pred_df[pred_df["model_version"] == model_version][["commit_id", "prob"]]
                .drop_duplicates(subset="commit_id", keep="first")
                .rename(columns={"prob": f"prob_{model_version}"})
                .reset_index(drop=True)
                .copy()
            )
            pred_df = pred_df.merge(prob_df, on="commit_id", how="left")

        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_preprocess_nn.pkl", "rb"
        ) as f:
            preprocess_ = pickle.load(f)

        # deduplicate train_pred_df by commit_id keeping the row with the lowest error
        pred_df = pred_df.drop_duplicates(subset="commit_id", keep="first")
        pred_df.set_index("commit_id", inplace=True)
        pred_df.reindex(test_commit_id)
        pred_df.reset_index(inplace=True)
        _LOGGER.info(f"After dedup prediction df shape: {pred_df.shape}")

        # load saved model selection model
        start = time.time()
        with open(
            settings.MODELS_DIR / f"{settings.EXP_ID}_{project_name}_w{i}_selekt_model_nn.pkl", "rb"
        ) as f:
            nn = pickle.load(f)
        _LOGGER.info(f"Loaded selekt model in {time.time() - start}")

        def load_model(model_version, dict_code):
            model = DeepJITExtended(vocab_code=len(dict_code))
            model.load_state_dict(
                torch.load(
                    settings.MODELS_DIR
                    / f"{settings.EXP_ID}_{project_name}_w{model_version}_model_nn.pt"
                )
            )
            model.eval()
            return model

        # old_test_pad_code = optim_padding_code(
        #     project_name, best_old_model, test_feature.code, old_dict_code
        # )

        old_models = []
        for model in stat_models:
            dict_code = pickle.load(
                open(settings.DATA_DIR / f"{project_name}_w{model}_dict_code.pkl", "rb")
            )
            scaler = pickle.load(
                open(settings.DATA_DIR / f"{project_name}_w{model}_scaler.pkl", "rb")
            )
            old_models += [(load_model(model, dict_code), scaler, dict_code)]

        new_dict_code = pickle.load(
            open(settings.DATA_DIR / f"{project_name}_w{i}_dict_code.pkl", "rb")
        )
        new_scaler = pickle.load(open(settings.DATA_DIR / f"{project_name}_w{i}_scaler.pkl", "rb"))
        new_nn = load_model(i, new_dict_code)
        eco_runtimes = np.zeros(test_feature.shape[0])
        base_runtimes = np.zeros(test_feature.shape[0])

        for j in range(test_feature.shape[0]):
            temp_df = test_feature[j : j + 1].copy()
            # baseline
            start = time.time()
            temp_pad_code = optim_padding_code(project_name, i, temp_df.code, new_dict_code)
            with torch.no_grad():
                predict = new_nn.forward(
                    torch.tensor(
                        new_scaler.transform(temp_df.drop(["code"], axis=1).values)
                    ).float(),
                    torch.tensor(temp_pad_code).long(),
                )
            base_runtimes[j] = time.time() - start

            # eco
            start = time.time()
            with torch.no_grad():
                predicted_proba = []
                for old_model, old_scaler, old_dict_code in old_models:
                    temp_pad_code = optim_padding_code(project_name, i, temp_df.code, old_dict_code)
                    predict = old_model.forward(
                        torch.tensor(
                            old_scaler.transform(temp_df.drop(["code"], axis=1).values)
                        ).float(),
                        torch.tensor(temp_pad_code).long(),
                    )
                    predicted_proba.append(predict.detach().numpy()[0])
            nn.predict(
                np.concatenate(
                    [
                        preprocess_.transform(temp_df).toarray(),
                        np.array(predicted_proba).reshape(1, -1),
                    ],
                    axis=1,
                )
            )
            eco_runtimes[j] = time.time() - start
            print(f"Base: {base_runtimes[j]}, Eco: {eco_runtimes[j]}")

        # *[OUT]: save ecoselekt inference latency results
        inf_performance_df = pd.concat(
            [
                inf_performance_df,
                pd.DataFrame(
                    {
                        "window": i,
                        "commit_id": test_commit_id,
                        "eco_pred_time": eco_runtimes,
                        "base_pred_time": base_runtimes,
                    }
                ),
            ],
            ignore_index=True,
        )
        inf_performance_df.to_csv(
            settings.DATA_DIR / f"{settings.EXP_ID}_{project_name}_inf_perf_nn.csv",
            index=False,
        )
        _LOGGER.info(f"Saved inference latency results for window {i}")


def main():
    try:
        for project_name in settings.PROJECTS:
            _LOGGER.info(f"Starting {project_name}")
            start = time.time()
            inference_selekt(project_name)
            _LOGGER.info(f"Finished {project_name} in {time.time() - start}")
    except Exception:
        _LOGGER.exception("Unexpected error occurred.")
