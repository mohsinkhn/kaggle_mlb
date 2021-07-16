import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

from src.pipelines.data_preparation import ParsePlayerData


if __name__ == "__main__":
    # Load data
    # raw_data = pd.read_csv("data/train.csv")
    # tr = raw_data.loc[raw_data.date < 20210401]
    # val = raw_data.loc[raw_data.date >= 20210401]
    # print(raw_data.shape, val.shape)

    # roster_2021 = pd.read_csv("data/players.csv")
    # roster_2021 = roster_2021.loc[roster_2021.playerForTestSetAndFuturePreds == True]
    # target_enc = ParsePlayerData("nextDayPlayerEngagement", ["target1", "target2", "target3", "target4"])
    # tr_index = target_enc.fit_transform(tr).reset_index(drop=False)
    # vl_index = target_enc.fit_transform(val).reset_index(drop=False)
    # vl_index = vl_index.loc[vl_index.playerId.isin(roster_2021.playerId.astype(str))]
    # tr_index.to_csv("data/tr_index.csv", index=False)
    # vl_index.to_csv("data/vl_index.csv", index=False)

    tr_index = pd.read_csv("data/tr_index.csv")
    vl_index = pd.read_csv("data/vl_index.csv")

    print(tr_index.shape, vl_index.shape)

    feature_flags = ['1', '2', '3', '31', '32', '33', '34', '4', '41', '5', '6']  # , '3', '4', '5', '6']
    Xtr, Xvl = [], []
    for flag in feature_flags:
        if flag == '1':
            Xtr.append(np.load(f"data/features/X_tr{flag}.npy")[:, np.r_[0:8, 16:24]])
            Xvl.append(np.load(f"data/features/X_vl{flag}.npy")[:, np.r_[0:8, 16:24]])
        else:
            Xtr.append(np.load(f"data/features/X_tr{flag}.npy"))
            Xvl.append(np.load(f"data/features/X_vl{flag}.npy"))
    X_tra, X_vla = np.hstack(Xtr)[:, :], np.hstack(Xvl)[:, :]

    targets = ['target1', 'target2', 'target3', 'target4']
    y_tr = tr_index[targets].values
    y_vl = vl_index[targets].values
    # print(np.unique(X_tra[:, 235]))
    tr1 = lgb.Dataset(X_tra, y_tr[:, 0])
    tr2 = lgb.Dataset(X_tra, y_tr[:, 1])
    tr3 = lgb.Dataset(X_tra, y_tr[:, 2])
    tr4 = lgb.Dataset(X_tra, y_tr[:, 3])

    vl1 = lgb.Dataset(X_vla, y_vl[:, 0], reference=tr1, )
    vl2 = lgb.Dataset(X_vla, y_vl[:, 1], reference=tr2, )
    vl3 = lgb.Dataset(X_vla, y_vl[:, 2], reference=tr3, )
    vl4 = lgb.Dataset(X_vla, y_vl[:, 3], reference=tr4, )

    # params = {
    #     'n_estimators': 4000,
    #     'learning_rate': 0.08,
    #     'num_leaves': 31,
    #     'colsample_bytree': 0.3,
    #     'subsample': 0.5,
    #     'reg_alpha': 0.1,
    #     'reg_lambda': 0.1,
    #     'max_bin': 255,
    #     'objective': 'mae',
    #     'metric': 'mae'
    # }

    params = {
        'n_estimators': 4000,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'min_data_in_leaf': 5,
        'colsample_bytree': 0.4,
        'subsample': 0.5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_bin': 255,
        'objective': 'mae',
        'metric': 'mae'
    }
    bst1 = lgb.train(params, tr1, valid_sets=[vl1], early_stopping_rounds=200, verbose_eval=50)
    pred1 = bst1.predict(X_vla)
    bst1.save_model("data/models/bst1_v1.pkl")

    bst2 = lgb.train(params, tr2, valid_sets=[vl2], early_stopping_rounds=200, verbose_eval=50)
    pred2 = bst2.predict(X_vla)
    bst2.save_model("data/models/bst2_v1.pkl")

    bst3 = lgb.train(params, tr3, valid_sets=[vl3], early_stopping_rounds=200, verbose_eval=50)
    pred3 = bst3.predict(X_vla)
    bst3.save_model("data/models/bst3_v1.pkl")

    bst4 = lgb.train(params, tr4, valid_sets=[vl4], early_stopping_rounds=200, verbose_eval=50)
    pred4 = bst4.predict(X_vla)
    bst4.save_model("data/models/bst4_v1.pkl")

    print("Final validation score")
    preds = np.vstack((pred1, pred2, pred3, pred4)).T
    np.save("data/vl_preds2.npy", preds)
    print(mae(y_vl, preds))
