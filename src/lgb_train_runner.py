import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

from src.pipelines.artifacts import ParsePlayerData


if __name__ == "__main__":
    # Load data
    raw_data = pd.read_csv("data/train.csv")
    target_enc = ParsePlayerData("nextDayPlayerEngagement", ["target1", "target2", "target3", "target4"])
    train_index = target_enc.fit_transform(raw_data).reset_index(drop=False)
    print(train_index.shape)

    feature_flags = ['1', '2', '3', '31', '32', '33', '34', '4', '41', '5', '6', '7']  # , '3', '4', '5', '6']
    Xtr = []
    for flag in feature_flags:
        if flag == '1':
            Xtr.append(np.load(f"data/features/X_train{flag}.npy")[:, np.r_[0:8, 16:24]])
        else:
            Xtr.append(np.load(f"data/features/X_train{flag}.npy"))
    X_tra = np.hstack(Xtr)

    targets = ['target1', 'target2', 'target3', 'target4']
    y_tr = train_index[targets].values
    # print(np.unique(X_tra[:, 235]))
    tr1 = lgb.Dataset(X_tra, y_tr[:, 0])
    tr2 = lgb.Dataset(X_tra, y_tr[:, 1])
    tr3 = lgb.Dataset(X_tra, y_tr[:, 2])
    tr4 = lgb.Dataset(X_tra, y_tr[:, 3])

    n_ests = [4000, 700, 2000, 3000]
    fixed_params = {
        'learning_rate': 0.05,
        'num_leaves': 255,
        'min_data_in_leaf': 2,
        'colsample_bytree': 0.4,
        'subsample': 0.95,
        'bagging_freq': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.01,
        'max_bin': 127,
        'device': 'gpu',
        'gpu_use_dp': False,
        'gpu_device_id': 0,
        'boost_from_average': True,
        'reg_sqrt': True,
        'objective': 'mae',
        'metric': 'mae',
        'verbose': -1,
        'num_threads': 16

    }

    params = {
        'n_estimators': n_ests[0],
        **fixed_params
    }

    bst1 = lgb.train(params, tr1, verbose_eval=50)
    bst1.save_model("data/models/bst1_train_v2.pkl")

    params = {
        'n_estimators': n_ests[1],
        **fixed_params
        }

    bst2 = lgb.train(params, tr2, verbose_eval=50)
    bst2.save_model("data/models/bst2_train_v2.pkl")

    params = {
        'n_estimators': n_ests[2],
        **fixed_params
    }

    bst3 = lgb.train(params, tr3, verbose_eval=50)
    bst3.save_model("data/models/bst3_train_v2.pkl")

    params = {
        'n_estimators': n_ests[3],
        **fixed_params
    }

    bst4 = lgb.train(params, tr4, verbose_eval=50)
    bst4.save_model("data/models/bst4_train_v2.pkl")

    print("DONE")
