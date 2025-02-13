import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import log_loss
import optuna
import torch

seed_num=0
np.random.seed(seed_num)

x_train = torch.load("./dataset/training/x_train.pt") # [5888, 4775]
y_train = torch.load("./dataset/training/y_train.pt") # 
x_test = torch.load("./dataset/training/x_test.pt")
y_test = torch.load("./dataset/training/y_test.pt")

x_train = x_train.detach().numpy()
y_train = y_train.detach().numpy()
x_test = x_test.detach().numpy()
y_test = y_test.detach().numpy()


def objective(trial):
    params = {
        "objective": 'binary:logistic',
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
        "reg_lambda":trial.suggest_int("reg_lambda", 0, 20),
        "alpha":trial.suggest_int("alpha", 0, 20),
        "gamma":trial.suggest_int("gamma", 0, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 4)
    }

    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, verbose=False)
    predictions = model.predict_proba(x_test)
    loss = log_loss(y_test, predictions)
    return loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

print('\nBest hyperparameters:', study.best_params)
print('Best loss:', study.best_value)

