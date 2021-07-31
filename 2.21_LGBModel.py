"""Train LGB models."""
import sys
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.pipeline import make_pipeline, make_union

from mllib.transformers import (
    DateLagN,
    ExpandingCount,
    ExpandingMean,
    ExpandingSum,
    FunctionTransfomer,
    LagN,
)
from src.constants import (
    TARGETS,
    awards_artifact,
    event_artifact,
    player_twitter_artifact,
    rosters_artifact,
    scores1_mean_artifact,
    scores2_mean_artifact,
    scores3_mean_artifact,
    scores4_mean_artifact,
    scores5_mean_artifact,
    targets_artifact,
    transactions_artifact,
)
from src.pipelines.artifacts import DataLoader, MapToCol, ParseJsonField
from src.pipelines.utils import batch_list


if __name__ == "__main__":

    # Generate index
    TRAIN_FILE = "data/train_updated.csv"
    PLAYERS_FILE = "data/players.csv"
    VAL_START_DATE = 20210601
    DEVICE = "gpu"
    device = DEVICE
    artifacts_path = "data/artifacts/v01"
    SAVE_FEATURES = True
    LOAD_FEATURES = False
    TRAIN_SEASON_ONLY = True
    SEED1 = 786
    SEED2 = 20201102

    # raw_data = pd.read_csv(TRAIN_FILE)
    # tr = raw_data.loc[raw_data.date < VAL_START_DATE]
    # val = raw_data.loc[raw_data.date >= VAL_START_DATE]
    # print(raw_data.shape, val.shape)

    # roster_2021 = pd.read_csv(PLAYERS_FILE)
    # roster_2021 = roster_2021.loc[roster_2021.playerForTestSetAndFuturePreds == True]
    # target_enc = ParseJsonField(
    #     date_field="date", data_field="nextDayPlayerEngagement", use_cols=TARGETS+['playerId']
    # )
    # tr_index = target_enc.transform(tr).reset_index(drop=False)
    # tr_index = tr_index.loc[tr_index.playerId.isin(roster_2021.playerId.astype(str))]
    # del tr

    # vl_index = target_enc.transform(val).reset_index(drop=False)
    # vl_index = vl_index.loc[vl_index.playerId.isin(roster_2021.playerId.astype(str))]
    # del raw_data, val
    # # tr_index.to_csv("data/tr_index_smallv01.csv", index=False)
    # # vl_index.to_csv("data/vl_index_smallv01.csv", index=False)

    tr_index = pd.read_csv("data/tr_index_smallv01.csv")
    vl_index = pd.read_csv("data/vl_index_smallv01.csv")
    print(tr_index.shape, vl_index.shape)

    seasons = pd.read_csv("data/seasons_formatted.csv")

    target_stats_train = (
        make_union(
            *[
                ExpandingMean(
                    key_cols=[0, 1, 2, 3],
                    hist_data_path=f"{artifacts_path}/{targets_artifact}",
                    N=j,
                    skip=10,
                    device=DEVICE,
                    fill_value=0,
                )
                for j in [7, 30, 60, 300]
            ],
        ),
    )

    target_stats_test = (
        make_union(
            *[
                ExpandingMean(
                    key_cols=[0, 1, 2, 3],
                    hist_data_path=f"{artifacts_path}/{targets_artifact}",
                    N=j,
                    skip=0,
                    device=DEVICE,
                    fill_value=0,
                )
                for j in [7, 30, 60, 300]
            ],
            verbose=True,
        ),
    )

    other_features = make_union(
        LagN(
            key_cols=[0],
            hist_data_path=f"{artifacts_path}/{awards_artifact}",
            fill_value=-1,
            N=1,
            skip=0,
            device=device,
        ),
        ExpandingCount(
            key_cols=[0],
            hist_data_path=f"{artifacts_path}/{awards_artifact}",
            fill_value=0,
            N=365,
            skip=0,
            device=device,
        ),
        LagN(
            key_cols=[0, 1, 2],
            hist_data_path=f"{artifacts_path}/{transactions_artifact}",
            fill_value=-1,
            N=1,
            skip=0,
            device=device,
        ),
        LagN(
            key_cols=[0],
            hist_data_path=f"{artifacts_path}/{rosters_artifact}",
            fill_value=-1,
            N=1,
            skip=0,
            device=device,
        ),
        *[
            ExpandingCount(
                key_cols=[0],
                hist_data_path=f"{artifacts_path}/{rosters_artifact}",
                fill_value=0,
                N=j,
                skip=0,
                device=device,
            )
            for j in [30, 300]
        ],
        make_pipeline(
            LagN(
                key_cols=[0],
                hist_data_path=f"{artifacts_path}/{player_twitter_artifact}",
                fill_value=0,
                N=1,
                skip=0,
                device=device,
            ),
            FunctionTransfomer(np.log1p),
        ),
    )

    scores1_cols = [0, 1, 2, 3, 4]
    scores2_cols = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]
    scores3_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores4_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    scores5_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    scores_lags = []
    for score_cols, score_artifact in [
        (scores1_cols, scores1_mean_artifact),
        (scores2_cols, scores2_mean_artifact),
        (scores3_cols, scores3_mean_artifact),
        (scores4_cols, scores4_mean_artifact),
        (scores5_cols, scores5_mean_artifact),
    ]:
        for j in range(2):
            for cols in batch_list(score_cols, 4):
                scores_lags.append(
                    LagN(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=-1,
                        N=j + 1,
                        skip=0,
                        device=device,
                    )
                )
    scores_lags = make_union(*scores_lags, verbose=True)

    scores_mean = []
    for score_cols, score_artifact in [
        (scores2_cols, scores2_mean_artifact),
        (scores4_cols, scores4_mean_artifact),
        (scores5_cols, scores5_mean_artifact),
    ]:
        for j in range(2):
            for cols in batch_list(score_cols, 4):
                scores_mean.append(
                    ExpandingMean(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=-1,
                        N=j + 1,
                        skip=0,
                        device=device,
                    )
                )
    scores_mean = make_union(*scores_mean, verbose=True)

    scores_sum = []
    for score_cols, score_artifact in [
        (scores2_cols, scores2_mean_artifact),
        (scores3_cols, scores3_mean_artifact),
        (scores4_cols, scores4_mean_artifact),
        (scores5_cols, scores5_mean_artifact),
    ]:
        for j in range(2):
            for cols in batch_list(score_cols, 4):
                scores_sum.append(
                    ExpandingSum(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=-1,
                        N=j + 1,
                        skip=0,
                        device=device,
                    )
                )
    scores_sum = make_union(*scores_sum, verbose=True)

    scores_extra = make_union(
        *[
            LagN(
                key_cols=[0, 4],
                hist_data_path=f"{artifacts_path}/{scores1_mean_artifact}",
                fill_value=-1,
                N=j + 1,
                skip=0,
                device=device,
            )
            for j in range(2, 14)
        ],
        *[
            ExpandingCount(
                key_cols=[0],
                hist_data_path=f"{artifacts_path}/{scores1_mean_artifact}",
                fill_value=0,
                N=j,
                skip=0,
                device=device,
            )
            for j in [30, 150]
        ],
    )

    all_players_features = make_union(
        *[
            DateLagN(
                date_col="date",
                key_cols=cols,
                hist_data_path=f"{artifacts_path}/{scores2_mean_artifact}",
                N=j + 1,
                skip=0,
                device=device,
            )
            for j in range(1)
            for cols in batch_list(scores2_cols)
        ],
        *[
            DateLagN(
                date_col="date",
                key_cols=cols,
                hist_data_path=f"{artifacts_path}/{scores4_mean_artifact}",
                N=j + 1,
                skip=0,
                device=device,
            )
            for j in range(1)
            for cols in batch_list(scores4_cols)
        ],
    )

    event_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    events_features = make_union(
        *[
            LagN(
                key_cols=cols,
                hist_data_path=f"{artifacts_path}/{event_artifact}",
                fill_value=0,
                N=j + 1,
                skip=0,
                device=device,
            )
            for j in range(2)
            for cols in batch_list(event_cols)
        ],
        MapToCol(
            map_col="date",
            attr="seasonflag",
            mapper_input="seasons_formatted.csv",
            mapper_pipeline=DataLoader(artifacts_path, ftype="csv"),
        ),
    )

    feature_pipeline_tr2 = make_union(
        target_stats_train,
        other_features,
        scores_lags,
        scores_mean,
        scores_sum,
        scores_extra,
        all_players_features,
        events_features,
        verbose=True,
    )
    feature_pipeline_te2 = make_union(
        target_stats_test,
        other_features,
        scores_lags,
        scores_mean,
        scores_sum,
        scores_extra,
        all_players_features,
        events_features,
        verbose=True,
    )

    if not LOAD_FEATURES:
        X_tr = feature_pipeline_tr2.transform(tr_index)
        X_vl = feature_pipeline_te2.transform(vl_index)
    else:
        X_tr = np.load("data/X_tr_v201_skip10.npy")
        X_vl = np.load("data/X_vl_v201_skip10.npy")
    y_tr = tr_index[TARGETS].values
    y_vl = vl_index[TARGETS].values
    print(X_tr.shape, X_vl.shape)

    if SAVE_FEATURES:
        np.save("data/X_tr_v201_skip10.npy", X_tr)
        np.save("data/X_vl_v201_skip10.npy", X_vl)

    if TRAIN_SEASON_ONLY:
        cond = X_tr[:, -1] > 0
        X_tr = X_tr[cond]
        y_tr = y_tr[cond]

        cond = X_vl[:, -1] > 0
        X_vl = X_vl[cond]
        y_vl = y_vl[cond]
        print(X_tr.shape, X_vl.shape, y_tr.shape, y_vl.shape)

    tr1 = lgb.Dataset(X_tr, y_tr[:, 0])
    tr2 = lgb.Dataset(X_tr, y_tr[:, 1])
    tr3 = lgb.Dataset(X_tr, y_tr[:, 2])
    tr4 = lgb.Dataset(X_tr, y_tr[:, 3])

    vl1 = lgb.Dataset(X_vl, y_vl[:, 0], reference=tr1)
    vl2 = lgb.Dataset(X_vl, y_vl[:, 1], reference=tr2)
    vl3 = lgb.Dataset(X_vl, y_vl[:, 2], reference=tr3)
    vl4 = lgb.Dataset(X_vl, y_vl[:, 3], reference=tr4)

    params1 = {
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "colsample_bytree": 0.5,
        "subsample": 0.95,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "extra_trees": False,
        "max_bin": 127,
        # 'device': 'gpu',
        # 'gpu_use_dp': False,
        # 'gpu_device_id': 0,
        "boost_from_average": True,
        "reg_sqrt": True,
        "objective": "mae",
        "metric": "mae",
        "verbose": -1,
        "seed": SEED1,
        "min_data_per_group": 10,
        "cat_l2": 10,
        "cat_smooth": 10,
        "num_threads": 16,
    }

    params2 = {
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "colsample_bytree": 0.55,
        "subsample": 0.95,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "extra_trees": False,
        "max_bin": 127,
        # 'device': 'gpu',
        # 'gpu_use_dp': False,
        # 'gpu_device_id': 0,
        "boost_from_average": True,
        "reg_sqrt": True,
        "objective": "mae",
        "metric": "mae",
        "verbose": -1,
        "seed": SEED2,
        "min_data_per_group": 10,
        "cat_l2": 10,
        "cat_smooth": 10,
        "num_threads": 16,
    }

    for i, params in enumerate([params1, params2]):
        bst1 = lgb.train(params, tr1, valid_sets=[vl1], early_stopping_rounds=200, verbose_eval=50)
        pred21 = bst1.predict(X_vl)
        print(mae(y_vl[:, 0], pred21))

        bst2 = lgb.train(params, tr2, valid_sets=[vl2], early_stopping_rounds=200, verbose_eval=50)
        pred22 = bst2.predict(X_vl)
        print(mae(y_vl[:, 1], pred22))

        bst3 = lgb.train(params, tr3, valid_sets=[vl3], early_stopping_rounds=200, verbose_eval=50)
        pred23 = bst3.predict(X_vl)
        print(mae(y_vl[:, 2], pred23))

        bst4 = lgb.train(params, tr4, valid_sets=[vl4], early_stopping_rounds=200, verbose_eval=50)
        pred24 = bst4.predict(X_vl)
        print(mae(y_vl[:, 3], pred24))

        preds_2 = np.vstack((pred21, pred22, pred23, pred24)).T
        print(f"Overall score for params {i} -> f{mae(y_vl, preds_2):6.4f}")
        bst1.save_model(f"artifacts/bst1_train_v401_{i+1}.pkl")
        bst2.save_model(f"artifacts/bst2_train_v401_{i+1}.pkl")
        bst3.save_model(f"artifacts/bst3_train_v401_{i+1}.pkl")
        bst4.save_model(f"artifacts/bst4_train_v401_{i+1}.pkl")

        np.save(f"data/lgb_t1_logv401_skip10_{i+1}.npy", pred21)
        np.save(f"data/lgb_t2_logv401_skip10_{i+1}.npy", pred22)
        np.save(f"data/lgb_t3_logv401_skip10_{i+1}.npy", pred23)
        np.save(f"data/lgb_t4_logv401_skip10_{i+1}.npy", pred24)
