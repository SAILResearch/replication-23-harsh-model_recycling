import datetime
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.helper import SliceDict
from tqdm import tqdm

import ecoselekt.utils as utils
from ecoselekt.settings import settings

torch.manual_seed(settings.RANDOM_SEED)


class DeepJITExtended(nn.Module):
    def __init__(self, vocab_msg, vocab_code):
        super(DeepJITExtended, self).__init__()

        V_msg = vocab_msg
        V_code = vocab_code
        Dim = settings.EMBEDDING_DIM
        Class = settings.CLASS_NUM

        Ci = 1  # input of convolutional layer
        Co = settings.NUM_FILTERS  # output of convolutional layer
        Ks = settings.FILTER_SIZES  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(settings.DROP_OUT)
        # commits metrics are added to the code representation with 12 features
        self.fc1 = nn.Linear((2 * len(Ks) * Co) + 12, settings.HIDDEN_UNITS)  # hidden units
        self.fc2 = nn.Linear(settings.HIDDEN_UNITS, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, settings.NUM_FILTERS * len(settings.FILTER_SIZES))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward(self, x_metrics, x_msg, x_code):
        x_msg = self.embed_msg(x_msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_code = self.embed_code(x_code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)

        x_commit = torch.cat((x_metrics, x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out


def tune_model(train_data, valid_data):
    # !When using this make DeepJITExtended init accept keyword arguments
    # TODO: add params
    params = {
        "module__num_filters": [16, 32, 64, 128],
        "module__filter_sizes": [[1, 2, 3]],
        "optimizer__lr": [1e-5],
        "module__embedding_dim": [16, 32, 64, 128],
        "module__dropout_keep_prob": [0.5],
        "module__hidden_units": [512],
    }

    cc2ftr, data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = train_data
    valid_cc2ftr, valid_data_pad_msg, valid_data_pad_code, valid_data_labels, _, _ = valid_data

    # combine data
    cc2ftr = torch.tensor(cc2ftr).long()
    data_pad_msg = torch.tensor(data_pad_msg).long()
    data_pad_code = torch.tensor(data_pad_code).long()
    data_labels = torch.tensor(data_labels).float()

    valid_cc2ftr = torch.tensor(valid_cc2ftr).long()
    valid_data_pad_msg = torch.tensor(valid_data_pad_msg).long()
    valid_data_pad_code = torch.tensor(valid_data_pad_code).long()
    valid_data_labels = torch.tensor(valid_data_labels).float()

    cc2ftr = torch.cat((cc2ftr, valid_cc2ftr), 0)
    data_pad_msg = torch.cat((data_pad_msg, valid_data_pad_msg), 0)
    data_pad_code = torch.cat((data_pad_code, valid_data_pad_code), 0)
    data_labels = torch.cat((data_labels, valid_data_labels), 0)

    params["module__vocab_msg"], params["module__vocab_code"] = [len(dict_msg)], [len(dict_code)]
    params["module__embedding_ftr"] = [cc2ftr.shape[1]]

    if len(data_labels.shape) == 1:
        params["module__class_num"] = [1]
    else:
        params["module__class_num"] = [data_labels.shape[1]]

    # create model with skorch
    model = NeuralNetClassifier(
        DeepJITExtended,
        criterion=nn.BCELoss,
        optimizer=torch.optim.Adam,
        max_epochs=50,
        batch_size=64,
        verbose=False,
    )

    grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3, verbose=2)
    print(cc2ftr.shape, data_pad_msg.shape, data_pad_code.shape, data_labels.shape)
    X_dict = SliceDict(
        ftr=cc2ftr,
        msg=data_pad_msg,
        code=data_pad_code,
    )

    grid_result = grid.fit(X_dict, data_labels)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def train_model(train_data, valid_data, project_name, window):
    # TODO: add params
    params = {
        "num_filters": 16,
        "filter_sizes": "1, 2, 3",
        "l2_reg_lambda": 1e-5,
        "num_epochs": 50,
        "embedding_dim": 16,
        "no_cuda": False,
        "dropout_keep_prob": 0.5,
        "hidden_units": 512,
        "batch_size": 64,
    }

    cc2ftr, data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code, commit_ids = train_data
    (
        valid_cc2ftr,
        valid_data_pad_msg,
        valid_data_pad_code,
        valid_data_labels,
        _,
        _,
        valid_commit_ids,
    ) = valid_data

    # set up parameters
    params["cuda"] = (not params["no_cuda"]) and torch.cuda.is_available()
    del params["no_cuda"]
    params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(",")]

    params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
    params["embedding_ftr"] = cc2ftr.shape[1]

    if len(data_labels.shape) == 1:
        params["class_num"] = 1
    else:
        params["class_num"] = data_labels.shape[1]
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create and train the defect model
    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["l2_reg_lambda"])

    criterion = nn.BCELoss()
    for epoch in range(1, params["num_epochs"] + 1):
        total_loss = 0
        # building batches for training model
        batches = utils.mini_batches_update_DExtended(
            X_ftr=cc2ftr,
            X_msg=data_pad_msg,
            X_code=data_pad_code,
            Y=data_labels,
            mini_batch_size=params["batch_size"],
        )
        model.train()
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = (
                    torch.tensor(pad_msg).cuda(),
                    torch.tensor(pad_code).cuda(),
                    torch.cuda.FloatTensor(labels),
                )
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, labels = (
                    torch.tensor(pad_msg).long(),
                    torch.tensor(pad_code).long(),
                    torch.tensor(labels).float(),
                )

            optimizer.zero_grad()
            predict = model.forward(ftr, pad_msg, pad_code)
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
        # add validation loss
        batches = utils.mini_batches_update_DExtended(
            X_ftr=valid_cc2ftr,
            X_msg=valid_data_pad_msg,
            X_code=valid_data_pad_code,
            Y=valid_data_labels,
            mini_batch_size=8,
        )
        valid_loss = 0
        model.eval()
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, labels = batch
            ftr = torch.tensor(ftr).long()
            pad_msg, pad_code, labels = (
                torch.tensor(pad_msg).long(),
                torch.tensor(pad_code).long(),
                torch.tensor(labels).float(),
            )

            predict = model.forward(ftr, pad_msg, pad_code)
            loss = criterion(predict, labels)
            valid_loss += loss

        print(
            "Epoch %i / %i -- Total loss: %f, Val loss: %f"
            % (epoch, params["num_epochs"], total_loss, valid_loss)
        )

    torch.save(model.state_dict(), settings.MODELS_DIR / f"{project_name}_w{window}_dnn.pt")


def evaluation_model(data, project_name, window):
    # TODO: add params
    params = {
        "num_filters": 16,
        "filter_sizes": "1, 2, 3",
        "embedding_dim": 16,
        "no_cuda": False,
        "dropout_keep_prob": 0.5,
        "hidden_units": 512,
        "batch_size": 64,
    }
    cc2ftr, pad_msg, pad_code, all_labels, dict_msg, dict_code, commit_ids = data
    batches = utils.mini_batches_DExtended(
        X_ftr=cc2ftr,
        X_msg=pad_msg,
        X_code=pad_code,
        Y=all_labels,
        mini_batch_size=params["batch_size"],
    )

    params["cuda"] = (not params["no_cuda"]) and torch.cuda.is_available()
    del params["no_cuda"]
    params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(",")]

    params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
    params["embedding_ftr"] = cc2ftr.shape[1]

    if len(all_labels.shape) == 1:
        params["class_num"] = 1
    else:
        params["class_num"] = all_labels.shape[1]
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(settings.MODELS_DIR / f"{project_name}_w{window}_dnn.pt"))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = (
                    torch.tensor(pad_msg).cuda(),
                    torch.tensor(pad_code).cuda(),
                    torch.cuda.FloatTensor(label),
                )
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, labels = (
                    torch.tensor(pad_msg).long(),
                    torch.tensor(pad_code).long(),
                    torch.tensor(label).float(),
                )
            if torch.cuda.is_available():
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    all_pred_label = np.array(all_predict) > 0.5
    print("distribution of predicted labels:", Counter(all_pred_label))
    print("distribution of actual labels:", Counter(all_label))

    auc_score = roc_auc_score(y_true=all_label, y_score=all_predict)
    print("Test data -- AUC score:", auc_score)

    return np.array(all_predict), all_pred_label, commit_ids, np.array(all_label)
