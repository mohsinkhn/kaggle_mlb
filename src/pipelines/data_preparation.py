import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from mllib.transformers import BaseTransformer
from src.utils.io import load_json, save_json


class ParsePlayerData(BaseTransformer):
    def __init__(self, field_name, use_cols):
        self.field_name = field_name
        self.use_cols = use_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dfs = []
        for _, row in tqdm(X.iterrows(), total=len(X)):
            data = row[self.field_name]
            date = row['date']
            if data == 'nan':
                continue
                # return None
            df = pd.read_json(data)
            df['playerId'] = df['playerId'].astype(str)
            df.set_index('playerId', inplace=True)
            df['date'] = date
            dfs.append(df[self.use_cols])
        return pd.concat(dfs)


class CreateUpdateArtifact(BaseTransformer):
    def __init__(self, root_path, load_artifact, save_artifact, player_mappings, renew=True):
        self.root_path = root_path
        self.renew = renew
        self.load_artifact = load_artifact
        self.artifact_load_path = None
        if self.load_artifact:
            self.artifact_load_path = Path(root_path) / self.load_artifact

        self.save_artifact = save_artifact
        self.artifact_save_path = Path(root_path) / self.save_artifact
        if self.renew:
            self.artifact_save_path.rmdir()
        self.artifact_save_path.mkdir(parents=True, exist_ok=True)
        self.playerid_mappings = player_mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        mapping_df = pd.DataFrame.from_dict(self.playerid_mappings, orient='index', columns=['playeridx'])

        if (self.artifact_load_path / 'data.npy').exists():
            prev_arr, date_mapping = self._load()
        else:
            prev_arr, date_mapping = None, {}

        dates = df.date.unique()
        for i, date in tqdm(enumerate(dates), total=len(dates)):
            tmp = df.loc[df.date == date]
            arr = pd.merge(mapping_df, tmp, left_index=True, right_index=True, how='left')
            del arr['date'], arr['playeridx']
            arr = np.expand_dims(arr.values, 0)
            if prev_arr is None:
                prev_arr = arr
            else:    
                prev_arr = np.append(prev_arr, arr, 0)
            if not date_mapping:
                date_mapping[str(date)] = 0
            else:
                date_mapping[str(date)] = max(date_mapping.values()) + i

        self._save(prev_arr, date_mapping)

    def _load(self):
        prev_arr = np.load(str(self.artifact_load_path / 'data.npy'), allow_pickle=True)
        date_mapping = load_json(str(self.artifact_load_path / 'date_mapping.json'))
        return prev_arr, date_mapping

    def _save(self, arr, date_mapping):
        np.save(str(self.artifact_save_path / 'data.npy'), arr)
        save_json(date_mapping, str(self.artifact_save_path / 'date_mapping.json'))

