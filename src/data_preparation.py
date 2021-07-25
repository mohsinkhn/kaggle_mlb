"""Prepare artifacts for creating features."""
import argparse

import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_union, make_pipeline

from src.constants import (
    TARGETS,
    PLTWITTER,
    SCORES1,
    SCORES2,
    SCORES3,
    SCORES4,
    SCORES5,
    TEAM_SCORES1,
    TEAM_SCORES2,
    TEAM_SCORES3,
    TEAM_STANDINGS,
    AWARDS,
    ROSTERS,
    TRANSACTIONS,
    AWARDID_DICT
)
from src.constants import (
    playerid_mapping,
    teamid_mapping,
    targets_artifact,
    scores1_mean_artifact,
    scores1_first_artifact,
    scores1_last_artifact,
    scores2_mean_artifact,
    scores2_first_artifact,
    scores2_last_artifact,
    scores3_mean_artifact,
    scores3_first_artifact,
    scores3_last_artifact,
    scores4_mean_artifact,
    scores4_first_artifact,
    scores4_last_artifact,
    scores5_mean_artifact,
    scores5_first_artifact,
    scores5_last_artifact,
    team_scores1_mean_artifact,
    team_scores2_mean_artifact,
    team_scores3_mean_artifact,
    awards_artifact,
    rosters_artifact,
    player_twitter_artifact,
    transactions_artifact,
    team_standings_artifact
)
from src.pipelines.artifacts import (
    DataLoader,
    FilterDf,
    GetUnique,
    CreateArtifact,
    PivotbyDateUser,
    Update3DArtifact,
    ParseJsonField,
    MapCol,
    GroupByAggDF,
)


def get_dataprep_pipelines(save_data):
    prepare_targets = make_pipeline(
        ParseJsonField(data_field="nextDayPlayerEngagement", use_cols=["playerId", *TARGETS]),
        PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
        Update3DArtifact(
            load_path=str(Path(save_data) / targets_artifact),
            save_path=save_data,
            artifact_name=targets_artifact,
        ),
    )

    prepare_scores1 = make_pipeline(
        ParseJsonField(data_field="playerBoxScores", use_cols=["playerId", *SCORES1]),
        MapCol(
            "positionType",
            mapping={
                "Pitcher": 1,
                "Infielder": 2,
                "Outfielder": 3,
                "Hitter": 4,
                "Catcher": 5,
                "Runner": 6,
            },
        ),
        make_union(
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "mean" for sc in SCORES1}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores1_mean_artifact),
                    save_path=save_data,
                    artifact_name=scores1_mean_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "first" for sc in SCORES1}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores1_first_artifact),
                    save_path=save_data,
                    artifact_name=scores1_first_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "last" for sc in SCORES1}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores1_last_artifact),
                    save_path=save_data,
                    artifact_name=scores1_last_artifact,
                ),
            ),
        )
    )

    prepare_scores2 = make_pipeline(
        ParseJsonField(data_field="playerBoxScores", use_cols=["playerId", *SCORES2]),
        make_union(
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "mean" for sc in SCORES2}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores2_mean_artifact),
                    save_path=save_data,
                    artifact_name=scores2_mean_artifact,
                )
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "first" for sc in SCORES2}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores2_first_artifact),
                    save_path=save_data,
                    artifact_name=scores2_first_artifact,
                )
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "last" for sc in SCORES2}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores2_last_artifact),
                    save_path=save_data,
                    artifact_name=scores2_last_artifact,
                )
            ),
        )
    )

    prepare_scores3 = make_pipeline(
        ParseJsonField(data_field="playerBoxScores", use_cols=["playerId", *SCORES3]),
        make_union(
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "mean" for sc in SCORES3}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores3_mean_artifact),
                    save_path=save_data,
                    artifact_name=scores3_mean_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "first" for sc in SCORES3}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores3_first_artifact),
                    save_path=save_data,
                    artifact_name=scores3_first_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "last" for sc in SCORES3}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores3_last_artifact),
                    save_path=save_data,
                    artifact_name=scores3_last_artifact,
                ),
            ),

        )
    )

    prepare_scores4 = make_pipeline(
        ParseJsonField(data_field="playerBoxScores", use_cols=["playerId", *SCORES4]),
        make_union(
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "mean" for sc in SCORES4}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores4_mean_artifact),
                    save_path=save_data,
                    artifact_name=scores4_mean_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "first" for sc in SCORES4}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores4_first_artifact),
                    save_path=save_data,
                    artifact_name=scores4_first_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "last" for sc in SCORES4}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores4_last_artifact),
                    save_path=save_data,
                    artifact_name=scores4_last_artifact,
                ),
            ),
        )
    )

    prepare_scores5 = make_pipeline(
        ParseJsonField(data_field="playerBoxScores", use_cols=["playerId", *SCORES5]),
        make_union(
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "mean" for sc in SCORES5}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores5_mean_artifact),
                    save_path=save_data,
                    artifact_name=scores5_mean_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "first" for sc in SCORES5}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores5_first_artifact),
                    save_path=save_data,
                    artifact_name=scores5_first_artifact,
                ),
            ),
            make_pipeline(
                GroupByAggDF(["playerId", "date"], agg_dict={sc: "last" for sc in SCORES5}),
                PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
                Update3DArtifact(
                    load_path=str(Path(save_data) / scores5_last_artifact),
                    save_path=save_data,
                    artifact_name=scores5_last_artifact,
                ),
            ),
        )
    )

    prepare_awards = make_pipeline(
        ParseJsonField(data_field="awards", use_cols=["playerId", *AWARDS]),
        MapCol("awardId", AWARDID_DICT),
        PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
        Update3DArtifact(
            load_path=str(Path(save_data) / awards_artifact),
            save_path=save_data,
            artifact_name=awards_artifact,
        ),
    )

    prepare_rosters = make_pipeline(
        ParseJsonField(data_field="rosters", use_cols=["playerId", *ROSTERS]),
        MapCol(
            "statusCode",
            {
                "A": 1,
                "RM": 2,
                "D60": 3,
                "D10": 4,
                "D7": 5,
                "PL": 6,
                "SU": 7,
                "BRV": 8,
                "FME": 9,
                "RES": 10,
                "DEC": 11,
            },
        ),
        PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
        Update3DArtifact(
            load_path=str(Path(save_data) / rosters_artifact),
            save_path=save_data,
            artifact_name=rosters_artifact,
        ),
    )

    prepare_transactions = make_pipeline(
        ParseJsonField(data_field="transactions", use_cols=["playerId", *TRANSACTIONS]),
        MapCol(
            "typeCode",
            {
                "SFA": 1,
                "TR": 2,
                "NUM": 3,
                "ASG": 4,
                "DES": 5,
                "CLW": 6,
                "OUT": 7,
                "REL": 8,
                "SC": 9,
                "OPT": 10,
                "RTN": 11,
                "SGN": 12,
                "SE": 13,
                "CU": 14,
                "DFA": 15,
                "RET": 16,
            },
        ),
        PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
        Update3DArtifact(
            load_path=str(Path(save_data) / transactions_artifact),
            save_path=save_data,
            artifact_name=transactions_artifact,
        ),
    )

    prepare_pltwitter = make_pipeline(
        ParseJsonField(data_field="playerTwitterFollowers", use_cols=["playerId", *PLTWITTER]),
        PivotbyDateUser(schema_file=str(Path(save_data) / playerid_mapping)),
        Update3DArtifact(
            load_path=str(Path(save_data) / player_twitter_artifact),
            save_path=save_data,
            artifact_name=player_twitter_artifact,
        ),
    )

    prepare_team_scores1_mean = make_pipeline(
        ParseJsonField(data_field="teamBoxScores", use_cols=["teamId", *TEAM_SCORES1]),
        GroupByAggDF(["teamId", "date"], agg_dict={sc: "mean" for sc in TEAM_SCORES1}),
        PivotbyDateUser(
            user_col="teamId", schema_file=str(Path(save_data) / teamid_mapping)
        ),
        Update3DArtifact(
            user_col="teamId",
            load_path=str(Path(save_data) / team_scores1_mean_artifact),
            save_path=save_data,
            artifact_name=team_scores1_mean_artifact,
        ),
    )

    prepare_team_scores2_mean = make_pipeline(
        ParseJsonField(data_field="teamBoxScores", use_cols=["teamId", *TEAM_SCORES2]),
        GroupByAggDF(["teamId", "date"], agg_dict={sc: "mean" for sc in TEAM_SCORES2}),
        PivotbyDateUser(
            user_col="teamId", schema_file=str(Path(save_data) / teamid_mapping)
        ),
        Update3DArtifact(
            user_col="teamId",
            load_path=str(Path(save_data) / team_scores2_mean_artifact),
            save_path=save_data,
            artifact_name=team_scores2_mean_artifact,
        ),
    )
    prepare_team_scores3_mean = make_pipeline(
        ParseJsonField(data_field="teamBoxScores", use_cols=["teamId", *TEAM_SCORES3]),
        GroupByAggDF(["teamId", "date"], agg_dict={sc: "mean" for sc in TEAM_SCORES3}),
        PivotbyDateUser(
            user_col="teamId", schema_file=str(Path(save_data) / teamid_mapping)
        ),
        Update3DArtifact(
            user_col="teamId",
            load_path=str(Path(save_data) / team_scores3_mean_artifact),
            save_path=save_data,
            artifact_name=team_scores3_mean_artifact,
        ),
    )

    prepare_team_standings = make_pipeline(
        ParseJsonField(data_field="standings", use_cols=["teamId", *TEAM_STANDINGS]),
        PivotbyDateUser(
            user_col="teamId", schema_file=str(Path(save_data) / teamid_mapping)
        ),
        Update3DArtifact(
            user_col="teamId",
            load_path=str(Path(save_data) / team_standings_artifact),
            save_path=save_data,
            artifact_name=team_standings_artifact,
        ),
    )
    dataprep_pipeline1 = make_union(
        prepare_targets,
        prepare_scores1,
        prepare_scores2,
        prepare_scores3,
        prepare_scores4,
        prepare_scores5,
        prepare_awards,
        prepare_rosters,
        prepare_transactions,
        prepare_pltwitter,
        prepare_team_scores1_mean,
        prepare_team_scores2_mean,
        prepare_team_scores3_mean,
        prepare_team_standings,
    )

    dataprep_pipeline2 = make_union(
        prepare_targets,
        prepare_scores1,
        prepare_scores2,
        prepare_scores3,
        prepare_scores4,
        prepare_scores5,
        prepare_awards,
        prepare_rosters,
        prepare_transactions,
        prepare_pltwitter,
        prepare_team_scores1_mean,
        prepare_team_scores2_mean,
        prepare_team_scores3_mean,
        prepare_team_standings,
    )
    return dataprep_pipeline1, dataprep_pipeline2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_data", type=str)
    parser.add_argument("--val_start_date", type=int)
    args = parser.parse_args()

    READ_ROOT = "./data"
    TRAIN_FILE = "./data/train.csv"
    VAL_START_DATE = args.val_start_date
    SAVE_DATA = args.save_data

    # Map playerids to ints
    playerid_mapper = make_pipeline(
        DataLoader(READ_ROOT, "csv"),
        FilterDf(filter_query="playerForTestSetAndFuturePreds == True"),
        GetUnique("playerId"),
        CreateArtifact(SAVE_DATA, playerid_mapping, "joblib"),
    )
    playerid_mapper.transform("players.csv")
    print("Playerid integer mapping created!")

    teamid_mapper = make_pipeline(
        DataLoader(READ_ROOT, "csv"),
        GetUnique("id"),
        CreateArtifact(SAVE_DATA, teamid_mapping, "joblib"),
    )
    teamid_mapper.transform("teams.csv")
    print("Teamid integer mapping created!")

    raw_data = pd.read_csv(TRAIN_FILE)
    tr = raw_data.loc[raw_data.date < VAL_START_DATE]
    vl = raw_data.loc[raw_data.date >= VAL_START_DATE]
    print(tr.shape, vl.shape)

    dataprep_pipeline1, dataprep_pipeline2 = get_dataprep_pipelines(SAVE_DATA)
    status = dataprep_pipeline1.transform(tr)
    print(f"Status for fitting datapipeline on train data: {status}")
    status = dataprep_pipeline2.transform(vl)
    print(f"Status for fitting datapipeline on val data: {status}")
