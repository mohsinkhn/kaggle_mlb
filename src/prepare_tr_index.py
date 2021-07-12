from pathlib import Path
import pandas as pd

from src.constants import playerid_mapping_file, root_path, VAL_START_DATE
from src.utils.io import load_json


if __name__ == "__main__":
    dates = pd.read_csv(Path(root_path) / "train.csv", usecols=["date"])
    dates = dates.loc[dates.date < VAL_START_DATE]
    filepath = str(Path(root_path) / playerid_mapping_file)
    playerid_mappings = load_json(filepath)
    dates = dates.date.unique()
    playerids = playerid_mappings.keys()

    date_playerid, dts, pids = [], [], []
    for dt in dates:
        for pid in playerids:
            dts.append(dt)
            pids.append(pid)
            date_playerid.append(f"{dt}_{pid}")
    df = pd.DataFrame({"date_playerId": date_playerid, "date": dts, "playerId": pids})
    df.to_parquet("data/tr_index.pq")
