import json
import pandas as pd
from pathlib import Path


from src.constants import root_path, playerid_mapping_file

if __name__ == "__main__":
    df = pd.read_csv(Path(root_path) / "players.csv")
    playerids = df["playerId"].unique()
    playerid_mapping = {str(pid): i for i, pid in enumerate(playerids)}
    with open(str(Path(root_path) / playerid_mapping_file), "w") as fp:
        json.dump(playerid_mapping, fp)
