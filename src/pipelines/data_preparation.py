import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from mllib.transformers import BaseTransformer
from src.utils.io import load_json, save_json


class ParsePlayerData(BaseTransformer):
    def __init__(self, field_name, use_cols, agg=None):
        self.field_name = field_name
        self.use_cols = use_cols
        self.agg = agg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dfs = []
        if not isinstance(X, pd.DataFrame):
            return None
        if (self.field_name not in X.columns) or ("date" not in X.columns):
            return None
        for _, row in tqdm(X.iterrows(), total=len(X)):
            data = row[self.field_name]
            if (
                (str(data) == "nan")
                or (str(data) == "")
                or (str(data) == "NaN")
                or (str(data) == "null")
                or (str(data) == "<NA>")
            ):
                continue
            date = row["date"]
            try:
                df = pd.read_json(data)
                df["playerId"] = df["playerId"].astype(str)
                if self.agg is not None:
                    df = getattr(df.groupby("playerId")[self.use_cols], self.agg)()
                else:
                    df.drop_duplicates(subset=["playerId"], inplace=True)
                    df.set_index("playerId", inplace=True)
                    df = df[self.use_cols]
                df["date"] = date
                dfs.append(df)
            except (ValueError, KeyError):
                continue
        if len(dfs) == 0:
            return None
        return pd.concat(dfs)


class CreateUpdateArtifact(BaseTransformer):
    def __init__(
        self,
        root_path,
        load_artifact,
        save_artifact,
        playerid_mappings_file,
        renew=True,
    ):
        self.root_path = root_path
        self.renew = renew
        self.load_artifact = load_artifact
        self.artifact_load_path = None
        if self.load_artifact:
            self.artifact_load_path = Path(root_path) / self.load_artifact

        self.save_artifact = save_artifact
        self.artifact_save_path = Path(root_path) / self.save_artifact
        if self.renew:
            if self.artifact_save_path.exists():
                shutil.rmtree(self.artifact_save_path)
        self.artifact_save_path.mkdir(parents=True, exist_ok=True)
        self.playerid_mappings_file = playerid_mappings_file

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            playerid_mappings = load_json(self.playerid_mappings_file)
            df = X.copy()
            mapping_df = pd.DataFrame.from_dict(
                playerid_mappings, orient="index", columns=["playeridx"]
            )

            if (self.artifact_load_path is not None) and (
                self.artifact_load_path / "data.npy"
            ).exists():
                prev_arr, date_mapping = self._load()
            else:
                prev_arr, date_mapping = None, {}

            dates = df.date.unique()
            for i, date in tqdm(enumerate(dates), total=len(dates)):
                tmp = df.loc[df.date == date]
                arr = pd.merge(mapping_df, tmp, left_index=True, right_index=True, how="left")
                del arr["date"], arr["playeridx"]
                arr = np.expand_dims(arr.values, 0).astype(np.float32)
                if prev_arr is None:
                    prev_arr = arr
                    date_mapping[str(date)] = 0
                else:
                    if str(date) in date_mapping:
                        prev_arr[date_mapping[str(date)]] = arr
                    else:
                        prev_arr = np.append(prev_arr, arr, 0)
                        date_mapping[str(date)] = max(date_mapping.values()) + 1
            self._save(prev_arr, date_mapping)
        except:
            pass


    def _load(self):
        prev_arr = np.load(str(self.artifact_load_path / "data.npy"), allow_pickle=True)
        date_mapping = load_json(str(self.artifact_load_path / "date_mapping.json"))
        return prev_arr, date_mapping

    def _save(self, arr, date_mapping):
        np.save(str(self.artifact_save_path / "data.npy"), arr)
        save_json(date_mapping, str(self.artifact_save_path / "date_mapping.json"))
