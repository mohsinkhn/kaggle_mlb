import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline

from mllib.transformers import OrdinalTransformer
from src.utils.io import load_json
from src.constants import root_path, playerid_mapping_file
from src.pipelines.data_preparation import ParsePlayerData, CreateUpdateArtifact


if __name__ == "__main__":
    filepath = str(Path(root_path) / playerid_mapping_file)
    playerid_mappings = load_json(filepath)

    raw_data = pd.read_csv(Path(root_path) / "train.csv")
    pipe1 = make_pipeline(
        ParsePlayerData("nextDayPlayerEngagement", ["date", "target1", "target2", "target3", "target4"]),
        CreateUpdateArtifact("./data", None, "tr_targets", playerid_mappings, True),
    )
    pipe1.transform(raw_data)

    pipe2 = make_pipeline(
        ParsePlayerData("rosters", ["date", "teamId", "statusCode"]),
        OrdinalTransformer(["statusCode"]),
        CreateUpdateArtifact("./data", None, "tr_rosters", playerid_mappings, True),
    )
    pipe2.transform(raw_data)

    pipe3 = make_pipeline(
        ParsePlayerData("awards", ["date", "awardId"]),
        OrdinalTransformer(["awardId"]),
        CreateUpdateArtifact("./data", None, "tr_awards", playerid_mappings, True),
    )
    pipe3.transform(raw_data)

    pipe4 = make_pipeline(
        ParsePlayerData(
            "playerBoxScores",
            [
                "date",
                "jerseyNum",
                "positionCode",
                "positionType",
                "battingOrder",
                "gamesPlayedBatting",
                "flyOuts",
                "groundOuts",
                "runsScored",
                "doubles",
                "triples",
                "homeRuns",
                "strikeOuts",
                "baseOnBalls",
                "intentionalWalks",
                "hits",
                "hitByPitch",
                "atBats",
                "caughtStealing",
                "stolenBases",
                "groundIntoDoublePlay",
                "groundIntoTriplePlay",
                "plateAppearances",
                "totalBases",
                "rbi",
                "leftOnBase",
                "sacBunts",
                "sacFlies",
                "catchersInterference",
                "pickoffs",
                "gamesPlayedPitching",
                "gamesStartedPitching",
                "completeGamesPitching",
                "shutoutsPitching",
                "winsPitching",
                "lossesPitching",
                "flyOutsPitching",
                "airOutsPitching",
                "groundOutsPitching",
                "runsPitching",
                "doublesPitching",
                "triplesPitching",
                "homeRunsPitching",
                "strikeOutsPitching",
                "baseOnBallsPitching",
                "intentionalWalksPitching",
                "hitsPitching",
                "hitByPitchPitching",
                "atBatsPitching",
                "caughtStealingPitching",
                "stolenBasesPitching",
                "inningsPitched",
                "saveOpportunities",
                "earnedRuns",
                "battersFaced",
                "outsPitching",
                "pitchesThrown",
                "balls",
                "strikes",
                "hitBatsmen",
                "balks",
                "wildPitches",
                "pickoffsPitching",
                "rbiPitching",
                "gamesFinishedPitching",
                "inheritedRunners",
                "inheritedRunnersScored",
                "catchersInterferencePitching",
                "sacBuntsPitching",
                "sacFliesPitching",
                "saves",
                "holds",
                "blownSaves",
                "assists",
                "putOuts",
                "errors",
                "chances",
            ],
        ),
        OrdinalTransformer(["positionCode", "positionType"]),
        CreateUpdateArtifact("./data", None, "tr_awards", playerid_mappings, True),
    )
    pipe4.transform(raw_data)

    pipe5 = make_pipeline(
        ParsePlayerData("playerTwitterFollowers", ["date", "numberOfFollowers"]),
        CreateUpdateArtifact("./data", None, "tr_twitter", playerid_mappings, True),
    )
    pipe5.transform(raw_data)

    pipe6 = make_pipeline(
        ParsePlayerData("transactions", ["date", "typeCode"]),
        OrdinalTransformer(["typeCode"]),
        CreateUpdateArtifact("./data", None, "tr_transactions", playerid_mappings, True),
    )
    pipe6.transform(raw_data)
