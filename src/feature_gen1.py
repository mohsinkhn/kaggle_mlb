import joblib

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder

from mllib.transformers import (
    ExpandingMean,
    ExpandingMedian,
    ExpandingMin,
    ExpandingMax,
    ExpandingVar,
    ExpandingQ05,
    ExpandingQ25,
    ExpandingQ75,
    ExpandingQ95,
    ExpandingSum,
    LagN,
    DateTimeFeatures,
    SelectCols,
    UnqLastN,
    LastNSum,
    LastNMean,
    MapAttributes,
    Astype,
    DateDiff,
)
from src.pipelines.data_preparation import ParsePlayerData


def get_save_features(pipe, tr_index, vl_index, indicator):
    X_tr = pipe.fit_transform(tr_index)
    X_vl = pipe.transform(vl_index)
    np.save(f"data/features/X_tr{indicator}.npy", X_tr)
    np.save(f"data/features/X_vl{indicator}.npy", X_vl)
    joblib.dump(pipe, f"data/features/pipe{indicator}.pkl")


if __name__ == "__main__":
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

    # features1 = make_union(
    #     ExpandingMean(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingMedian(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingMax(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingMin(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingQ25(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingQ75(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingQ95(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingQ05(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingMin(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingVar(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, skip=30
    #     ),
    #     ExpandingMean(
    #         "date", "playerId", [0, 1, 2, 3], "data/tr_targets/", "data/playerid_mappings.json", fill_value=-1, N=120, skip=30,
    #     ), verbose=True
    # )
    # get_save_features(features1, tr_index, vl_index, 1)
    # print("Done features 1")

    # features2 = make_union(
    #     *[
    #         LagN(
    #             "date",
    #             "playerId",
    #             [1],
    #             "data/tr_rosters/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=i,
    #             skip=0,
    #         )
    #         for i in range(1, 4)
    #     ],
    #     *[
    #         LagN(
    #             "date",
    #             "playerId",
    #             [0],
    #             "data/tr_rosters/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=i,
    #             skip=0,
    #         )
    #         for i in range(1, 2)
    #     ],
    #     *[
    #         UnqLastN(
    #             "date",
    #             "playerId",
    #             [1],
    #             "data/tr_rosters/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=i,
    #             skip=0,
    #         )
    #         for i in range(2, 5)
    #     ],
    #     verbose=True,
    # )
    # get_save_features(features2, tr_index, vl_index, 2)
    # print("Done features 2")

    # features3 = make_union(
    #     *[
    #         LagN(
    #             "date",
    #             "playerId",
    #             list(range(4 * i, 4 * (i + 1))),
    #             "data/tr_scores_mean/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=j,
    #         )
    #         for i in range(17)
    #         for j in range(1, 3)
    #     ], verbose=True
    # )
    # get_save_features(features3, tr_index, vl_index, 3)
    # print("Done features 3")

    # features31 = make_union(
    #     *[
    #         LagN(
    #             "date",
    #             "playerId",
    #             list(range(4 * i, 4 * (i + 1))),
    #             "data/tr_scores_max/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=j,
    #         )
    #         for i in range(17)
    #         for j in range(1, 3)
    #     ], verbose=True
    # )
    # get_save_features(features31, tr_index, vl_index, 31)
    # print("Done features 31")

    # features32 = make_union(
    #     *[
    #         LagN(
    #             "date",
    #             "playerId",
    #             list(range(4 * i, 4 * (i + 1))),
    #             "data/tr_scores_sum/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=j,
    #         )
    #         for i in range(17)
    #         for j in range(1, 3)
    #     ], verbose=True
    # )
    # get_save_features(features32, tr_index, vl_index, 32)
    # print("Done features 32")

    # features33 = make_union(
    #     *[
    #         LastNMean(
    #             "date",
    #             "playerId",
    #             list(range(4 * i, 4 * (i + 1))),
    #             "data/tr_scores_mean/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=j,
    #         )
    #         for i in range(17)
    #         for j in [7, 30]
    #     ], verbose=True
    # )
    # get_save_features(features33, tr_index, vl_index, 33)
    # print("Done features 32")

    # features34 = make_union(
    #     *[
    #         LastNSum(
    #             "date",
    #             "playerId",
    #             list(range(4 * i, 4 * (i + 1))),
    #             "data/tr_scores_sum/",
    #             "data/playerid_mappings.json",
    #             fill_value=-1,
    #             N=j,
    #         )
    #         for i in range(17)
    #         for j in [7, 30]
    #     ], verbose=True
    # )
    # get_save_features(features34, tr_index, vl_index, 34)
    # print("Done features 34")


    # features4 = make_union(
    #     make_pipeline(
    #         MapAttributes('data/players.csv', 'csv', 'playerId', 'primaryPositionCode'),
    #         Astype('int32', {'O': 11, 'I': 12})
    #     ),
    #     MapAttributes('data/players.csv', 'csv', 'playerId', 'weight'),
    #     MapAttributes('data/players.csv', 'csv', 'playerId', 'heightInches'),
    # )
    # get_save_features(features4, tr_index, vl_index, 4)

    features41 = DateDiff(date_col='date', user_col='playerId', diff_col='mlbDebutDate', data_filepath='data/players.csv', format='%Y%m%d')
    get_save_features(features41, tr_index, vl_index, 41)
    # features4 = make_union(
    #     LagN(
    #         "date",
    #         "playerId",
    #         [0, 1, 2, 3],
    #         "data/tr_scores/",
    #         "data/playerid_mappings.json",
    #         fill_value=-1,
    #         N=1,
    #     ),
    #     verbose=True,
    # )
    # get_save_features(features4, tr_index, vl_index, 4)
    # print("Done features 4")

    features5 = make_union(
        make_pipeline(
            LastNSum("date", "playerId", [0], "data/tr_awards/", "data/playerid_mappings.json", fill_value=-1, N=1000),
        ), verbose=True
    )
    get_save_features(features5, tr_index, vl_index, 5)
    print("Done features 5")

    features6 = make_union(
        LagN(
            "date",
            "playerId",
            [0],
            "data/tr_transactions/",
            "data/playerid_mappings.json",
            fill_value=-1,
            N=1,
        ),
        verbose=True,
    )
    get_save_features(features6, tr_index, vl_index, 6)
    print("Done features 6")
    # features6 = make_pipeline(SelectCols("date"), DateTimeFeatures(attrs=["dayofweek"]), verbose=True)
    # get_save_features(features6, tr_index, vl_index, 6)
    # print("Done features 6")
