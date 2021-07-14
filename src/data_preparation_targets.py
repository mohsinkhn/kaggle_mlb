import joblib

import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline

from mllib.transformers import OrdinalTransformer
from src.constants import root_path, playerid_mapping_file, VAL_START_DATE
from src.pipelines.data_preparation import ParsePlayerData, CreateUpdateArtifact


if __name__ == "__main__":
    filepath = str(Path(root_path) / playerid_mapping_file)
    raw_data = pd.read_csv(Path(root_path) / "train.csv")
    tr_data = raw_data.loc[raw_data["date"] < VAL_START_DATE]
    # pipe1 = make_pipeline(
    #     ParsePlayerData("nextDayPlayerEngagement", ["target1", "target2", "target3", "target4"]),
    #     CreateUpdateArtifact("./data", None, "tr_targets", filepath, True),
    # )
    # pipe1.fit_transform(tr_data)
    # joblib.dump(pipe1, 'data/pipeline_dataprep_pipe1.pkl')

    pipe2 = make_pipeline(
        ParsePlayerData("rosters", ["teamId", "statusCode"]),
        OrdinalTransformer(["teamId", "statusCode"]),
        CreateUpdateArtifact("./data", None, "tr_rosters", filepath, True),
    )
    pipe2.fit_transform(raw_data)
    joblib.dump(pipe2, 'data/pipeline_dataprep_pipe2.pkl')

    # pipe3 = make_pipeline(
    #     ParsePlayerData("awards", ["awardId"]),
    #     OrdinalTransformer(["awardId"]),
    #     CreateUpdateArtifact("./data", None, "tr_awards", filepath, True),
    # )
    # pipe3.fit_transform(raw_data)
    # joblib.dump(pipe3, 'data/pipeline_dataprep_pipe3.pkl')

    # pipe4 = make_pipeline(
    #     ParsePlayerData(
    #         "playerBoxScores",
    #         [
    #             "jerseyNum",
    #             "positionCode",
    #             "positionType",
    #             "battingOrder",
    #         ],
    #     ),
    #     OrdinalTransformer(["jerseyNum", "positionCode", "positionType"]),
    #     CreateUpdateArtifact("./data", None, "tr_scores", filepath, True),
    # )
    # pipe4.fit_transform(raw_data)
    # joblib.dump(pipe4, 'data/pipeline_dataprep_pipe4.pkl')

    # pipe42 = make_pipeline(
    #     ParsePlayerData(
    #         "playerBoxScores",
    #         [   
    #             "gamesPlayedBatting",
    #             "flyOuts",
    #             "groundOuts",
    #             "runsScored",
    #             "doubles",
    #             "triples",
    #             "homeRuns",
    #             "strikeOuts",
    #             "baseOnBalls",
    #             "intentionalWalks",
    #             "hits",
    #             "hitByPitch",
    #             "atBats",
    #             "caughtStealing",
    #             "stolenBases",
    #             "groundIntoDoublePlay",
    #             "groundIntoTriplePlay",
    #             "plateAppearances",
    #             "totalBases",
    #             "rbi",
    #             "leftOnBase",
    #             "sacBunts",
    #             "sacFlies",
    #             "catchersInterference",
    #             "pickoffs",
    #             "gamesPlayedPitching",
    #             "gamesStartedPitching",
    #             "completeGamesPitching",
    #             "shutoutsPitching",
    #             "winsPitching",
    #             "lossesPitching",
    #             "flyOutsPitching",
    #             "airOutsPitching",
    #             "groundOutsPitching",
    #             "runsPitching",
    #             "doublesPitching",
    #             "triplesPitching",
    #             "homeRunsPitching",
    #             "strikeOutsPitching",
    #             "baseOnBallsPitching",
    #             "intentionalWalksPitching",
    #             "hitsPitching",
    #             "hitByPitchPitching",
    #             "atBatsPitching",
    #             "caughtStealingPitching",
    #             "stolenBasesPitching",
    #             "inningsPitched",
    #             "saveOpportunities",
    #             "earnedRuns",
    #             "battersFaced",
    #             "outsPitching",
    #             "pitchesThrown",
    #             "balls",
    #             "strikes",
    #             "hitBatsmen",
    #             "balks",
    #             "wildPitches",
    #             "pickoffsPitching",
    #             "rbiPitching",
    #             "gamesFinishedPitching",
    #             "inheritedRunners",
    #             "inheritedRunnersScored",
    #             "catchersInterferencePitching",
    #             "sacBuntsPitching",
    #             "sacFliesPitching",
    #             "saves",
    #             "holds",
    #             "blownSaves",
    #             "assists",
    #             "putOuts",
    #             "errors",
    #             "chances",
    #         ], agg=True,
    #     ),
    #     CreateUpdateArtifact("./data", None, "tr_scores2", filepath, True),
    # )
    # pipe42.fit_transform(raw_data)
    # joblib.dump(pipe42, 'data/pipeline_dataprep_pipe42.pkl')

    # pipe5 = make_pipeline(
    #     ParsePlayerData("playerTwitterFollowers", ["numberOfFollowers"]),
    #     CreateUpdateArtifact("./data", None, "tr_twitter", filepath, True),
    # )
    # pipe5.fit_transform(raw_data)
    # joblib.dump(pipe5, 'data/pipeline_dataprep_pipe5.pkl')

    # pipe6 = make_pipeline(
    #     ParsePlayerData("transactions", ["typeCode"]),
    #     OrdinalTransformer(["typeCode"]),
    #     CreateUpdateArtifact("./data", None, "tr_transactions", filepath, True),
    # )
    # pipe6.fit_transform(raw_data)
    # joblib.dump(pipe6, 'data/pipeline_dataprep_pipe6.pkl')
