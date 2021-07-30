"""Train LGB models."""
import itertools

import numpy as np
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
from src.pipelines.artifacts import DataLoader, MapToCol
from src.pipelines.utils import batch_list


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
event_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def get_feature_pipeline1(
    artifacts_path,
    device="gpu",
    target_windows=[7, 30, 150, 1500],
    score_windows=[30, 150],
    score5_windows=[30, 150],
):
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
    target_stats_train1 = make_union(
        *[
            ExpandingMean(
                key_cols=[0, 1, 2, 3],
                hist_data_path=f"{artifacts_path}/{targets_artifact}",
                N=j,
                skip=10,
                device=device,
                fill_value=0,
            )
            for j in target_windows
        ],
    )

    target_stats_test1 = make_union(
        *[
            ExpandingMean(
                key_cols=[0, 1, 2, 3],
                hist_data_path=f"{artifacts_path}/{targets_artifact}",
                N=j,
                skip=0,
                device=device,
                fill_value=0,
            )
            for j in target_windows
        ],
        verbose=True,
    )

    other_features1 = make_union(
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
            for j in score_windows
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

    scores_lags1 = make_union(
        *[
            make_union(
                *[
                    LagN(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=-1,
                        N=j + 1,
                        skip=0,
                        device=device,
                    )
                    for j in range(2)
                    for cols in batch_list(score_cols, 4)
                ]
            )
            for score_cols, score_artifact in [
                (scores1_cols, scores1_mean_artifact),
                (scores2_cols, scores2_mean_artifact),
                (scores3_cols, scores3_mean_artifact),
                (scores4_cols, scores4_mean_artifact),
                (scores5_cols, scores5_mean_artifact),
            ]
        ]
    )

    scores_mean1 = make_union(
        *[
            make_union(
                *[
                    ExpandingMean(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=0,
                        N=j,
                        skip=0,
                        device=device,
                    )
                    for j in score_windows
                    for cols in batch_list(score_cols, 4)
                ]
            )
            for score_cols, score_artifact in [
                (scores2_cols, scores2_mean_artifact),
                (scores4_cols, scores4_mean_artifact),
            ]
        ]
    )

    scores5_mean1 = make_union(
        *[
            make_union(
                *[
                    ExpandingMean(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=0,
                        N=j,
                        skip=0,
                        device=device,
                    )
                    for j in score5_windows
                    for cols in batch_list(score_cols, 4)
                ]
            )
            for score_cols, score_artifact in [
                (scores5_cols, scores5_mean_artifact),
            ]
        ]
    )

    scores_sum1 = make_union(
        *[
            make_union(
                *[
                    ExpandingSum(
                        key_cols=cols,
                        hist_data_path=f"{artifacts_path}/{score_artifact}",
                        fill_value=0,
                        N=j,
                        skip=0,
                        device=device,
                    )
                    for j in score_windows
                    for cols in batch_list(score_cols, 4)
                ]
            )
            for score_cols, score_artifact in [
                (scores2_cols, scores2_mean_artifact),
                (scores3_cols, scores3_mean_artifact),
                (scores4_cols, scores4_mean_artifact),
                (scores5_cols, scores5_mean_artifact),
            ]
        ]
    )

    scores_extra1 = make_union(
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
            for j in score5_windows
        ],
    )

    feature_pipeline_tr1 = make_union(
        target_stats_train1,
        other_features1,
        scores_lags1,
        scores_mean1,
        scores5_mean1,
        scores_sum1,
        scores_extra1,
        all_players_features,
        events_features,
        verbose=True,
    )
    feature_pipeline_te1 = make_union(
        target_stats_test1,
        other_features1,
        scores_lags1,
        scores_mean1,
        scores5_mean1,
        scores_sum1,
        scores_extra1,
        all_players_features,
        events_features,
        verbose=True,
    )
    return feature_pipeline_tr1, feature_pipeline_te1
