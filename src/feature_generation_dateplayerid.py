import joblib

import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline, make_union

from mllib.transformers import ExpandingMean, ExpandingMedian
from src.utils.io import load_json
from src.constants import root_path, playerid_mapping_file, VAL_START_DATE
from src.pipelines.data_preparation import ParsePlayerData, CreateUpdateArtifact


if __name__ == "__main__":
    tr_index = 
    pipe1 = make_pipeline(
        ParsePlayerData("nextDayPlayerEngagement", ["date", "target1", "target2", "target3", "target4"]),
        CreateUpdateArtifact("./data", None, "tr_targets", playerid_mappings, True),
    