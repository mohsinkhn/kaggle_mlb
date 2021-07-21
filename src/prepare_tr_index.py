from pathlib import Path
import pandas as pd

from src.pipelines.data_preparation import ParsePlayerData



if __name__ == "__main__":
    raw_data = pd.read_csv("data/train.csv")
    tr = raw_data.loc[raw_data.date < 20210401]
    val = raw_data.loc[raw_data.date >= 20210401]
    print(raw_data.shape, val.shape)

    roster_2021 = pd.read_csv("data/players.csv")
    roster_2021 = roster_2021.loc[roster_2021.playerForTestSetAndFuturePreds == True]
    target_enc = ParsePlayerData("nextDayPlayerEngagement", ["target1", "target2", "target3", "target4"])
    tr_index = target_enc.fit_transform(tr).reset_index(drop=False)
    # tr_index['debutdate'] = tr_index.map()
    vl_index = target_enc.fit_transform(val).reset_index(drop=False)
    vl_index = vl_index.loc[vl_index.playerId.isin(roster_2021.playerId.astype(str))]
    tr_index.to_csv("data/tr_index.csv", index=False)
    vl_index.to_csv("data/vl_index.csv", index=False)

    tr_index = pd.read_csv("data/tr_index.csv")
    vl_index = pd.read_csv("data/vl_index.csv")
