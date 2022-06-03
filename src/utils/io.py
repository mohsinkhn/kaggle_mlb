import json
from dataclasses import dataclass

import pandas as pd
from pathlib import Path


@dataclass
class Files:
    root_path: str
    train: str
    awards: str
    test: str
    players: str
    teams: str
    seasons: str
    submission: str

    def _update_fields(self):
        self.__dict__ = {k: str(Path(self.root_path) / v) for k, v in self.__dict__.items() if k != 'root_path'}


class MLB(object):
    def __init__(self, df, player_ids):
        self.df = df.copy()
        self.df.set_index('date', inplace=True)
        self.player_ids = player_ids
        self.preds = []
        self.move_to_next = True
        self.dates = self.df.index.unique()
        self.pred_df = pd.DataFrame({'playerId': self.player_ids})
        self.pred_df['target1'] = 0
        self.pred_df['target2'] = 0
        self.pred_df['target3'] = 0
        self.pred_df['target4'] = 0

    def predict(self, pred_df):
        self.preds.append(pred_df)
        self.move_to_next = True

    def iter_test(self):
        for date in self.dates:
            test_df = self.df.loc[date]
            if 'nextDayPlayerEngagement' in test_df:
                del test_df['nextDayPlayerEngagement']

            sample_pred_df = self.pred_df.copy()
            sample_pred_df['date_playerId'] = sample_pred_df['playerId'].apply(lambda x: f"{date}_{x}")
            del sample_pred_df['playerId']
            sample_pred_df['date'] = date
            sample_pred_df.set_index('date', inplace=True)
            self.move_to_next = False
            yield test_df, sample_pred_df
            if self.move_to_next:
                continue
            else:
                print("Call `predict()` first.")

    def __iter__(self):
        return self


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def save_json(obj, filepath):
    with open(filepath, "w") as f:
        json.dump(obj, f)
