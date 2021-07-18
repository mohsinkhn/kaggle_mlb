import json
import shutil

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from mllib.transformers import BaseTransformer
from src.utils.io import load_json, save_json


def save_data(data, save_path, artifact_name, save_type):
    save_path = Path(save_path)
    try:
        save_path.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        return False, "Permission denied to save artifact"

    save_filepath = str(save_path / artifact_name)
    if save_type == 'npy':
        try:
            np.save(save_filepath, data)
        except TypeError:
            return False, "Not able to save as numpy file"

    elif save_type == 'json':
        try:
            with open(save_filepath, "w") as fh:
                json.dump(data, fh)
        except TypeError:
            return False, "Not able to save as json file"
    else:
        joblib.dump(data, save_filepath)
    return True, None


def load_data(load_filepath, file_type):
    if not Path(load_filepath).exists():
        return None

    load_filepath = str(load_filepath)

    if file_type == "csv":
        data = pd.read_csv(load_filepath) 

    elif file_type == "npy":
        data = np.load(load_filepath, allow_pickle=True)

    elif file_type == 'json':
        with open(load_filepath, "r") as fh:
            data = json.load(fh)

    else:
        data = joblib.load(load_filepath)
    return data


class DataLoader(BaseTransformer):
    """Load data from filepaths."""
    def __init__(self, load_path, ftype='csv'):
        self.load_path = load_path
        self.ftype = ftype

    def _transform(self, filename):
        filepath = str(Path(self.load_path) / filename)
        data = load_data(filepath, file_type=self.ftype)
        if data is None:
            print(f"ERROR!!! Not able to load {filename}")
        return data


class CreateArtifact(BaseTransformer):
    def __init__(self, save_path=None, artifact_name=None, save_type='joblib'):
        """Dump data toprovided filepaths."""
        self.save_path = save_path
        self.artifact_name = artifact_name
        self.save_type = save_type

    def _transform(self, data):
        if data is None:
            return False, "No input data found."
        flag, msg = save_data(data, self.save_path, self.artifact_name, self.save_type)
        if not flag:
            print(msg)
            return False
        return True


class Update3DArtifact(BaseTransformer):
    def __init__(self, date_col='date',
                 user_col='playerId',
                 load_path=None,
                 save_path=None,
                 artifact_name=None,
                 save_type='joblib'):
        """Dump data to provided filepaths."""
        self.date_col = date_col
        self.user_col = user_col
        self.load_path = load_path
        self.save_path = save_path
        self.artifact_name = artifact_name
        self.save_type = save_type

    def _transform(self, data):
        if data is None:
            return None
        artifact = load_data(self.load_path, self.save_type)
        prev_arr, dates, users = artifact["data"], artifact[self.date_col], artifact[self.user_col]
        curr_arr, curr_dates, curr_users = data["data"], data[self.date_col], data[self.user_col]
        print("Loaded data ...")
        date_idx_map = {d: i for i, d in enumerate(dates)}
        user_idx_map = {u: i for i, u in enumerate(users)}

        curr_users_idx = np.array([user_idx_map[u] for u in curr_users if u in user_idx_map])
        valid_users_idx = np.array([i for i, u in enumerate(curr_users) if u in user_idx_map])
        for i, date in tqdm(enumerate(curr_dates)):
            if date in date_idx_map:
                prev_arr[date_idx_map[date], curr_users_idx] = curr_arr[i, valid_users_idx]
            else:
                prev_arr = np.append(prev_arr, curr_arr[i, curr_users_idx])
                dates.append(date)
        data = {'data': prev_arr, self.date_col: dates, self.user_col: users}
        print("Updating data, Saving ...")
        save_data(data, self.save_path, self.artifact_name, self.save_type)
        return True


class DfTransformer(BaseTransformer):
    """Abstract class for validating dataframe transformations."""
    def transform(self, X, y=None):
        if not self._validate_input(X):
            return None

        return self._transform(X)

    def _validate_input(self, X):
        if X is None:
            print("Input is None")
            return False
        if not isinstance(X, pd.DataFrame):
            print("Input is not a pandas DataFrame")
            return False
        return True


class FilterDf(DfTransformer):
    def __init__(self, filter_query=None):
        self.filter_query = filter_query

    def _transform(self, X):
        try:
            Xt = X.query(self.filter_query)
        except ValueError:
            print("Not able to apply filter query.")
            Xt = X.copy()
        return Xt


class GetUnique(DfTransformer):
    def __init__(self, field_name=None):
        self.field_name = field_name

    def _transform(self, X: pd.DataFrame, y=None):
        return X[self.field_name].unique()


class MapCol(DfTransformer):
    def __init__(self, field_name=None, mapping=None, fill_value=0):
        self.field_name = field_name
        self.mapping = mapping
        self.fill_value = fill_value

    def _transform(self, X: pd.DataFrame, y=None):
        X[self.field_name] = np.array([self.mapping.get(val, self.fill_value) for val in X[self.field_name].values])
        return X


class ParseJsonField(DfTransformer):
    def __init__(self, date_field="date", data_field=None, use_cols=None):
        self.date_field = date_field
        self.data_field = data_field
        self.use_cols = use_cols

    def _transform(self, X):
        if (self.data_field not in X.columns) or (self.data_field not in X.columns):
            return None

        data = []
        for _, row in tqdm(X.iterrows(), total=len(X)):
            row_data = row[self.data_field]
            try:
                row_df = pd.read_json(row_data)[self.use_cols]
            except (ValueError, KeyError):
                continue

            row_df[self.date_field] = row[self.date_field]
            data.append(row_df)

        if len(data) == 0:
            return None
        return pd.concat(data)


class GroupByAggDF(DfTransformer):
    def __init__(self, grouper=None, agg_dict=None):
        self.grouper = grouper
        self.agg_dict = agg_dict

    def _transform(self, X, y=None):
        try:
            return X.groupby(self.grouper).agg(self.agg_dict).reset_index(drop=False)
        except (KeyError, AttributeError):
            return None


class PivotbyDateUser(DfTransformer):
    def __init__(self, date_col="date", user_col="playerId", schema_file=None, dtype='float32', fill_value=np.nan):
        self.date_col = date_col
        self.user_col = user_col
        self.schema_file = schema_file
        self.dtype = dtype
        self.fill_value = fill_value

    def _transform(self, X, y=None):
        if (self.date_col not in X.columns) or (self.user_col not in X.columns):
            print("Warning! Could not find date/user column in dataframe")
            return None

        if not Path(self.schema_file).exists():
            return None

        schema_users = load_data(self.schema_file, 'joblib')
        schema_user_idx = {u: i for i, u in enumerate(schema_users)}
        valid_rows = np.array([True if u in schema_user_idx else False for u in X[self.user_col].values])
        unq_dates, indices = np.unique(X[self.date_col].values[valid_rows], return_inverse=True)  # sunique()
        feature_cols = [col for col in X.columns if col not in set([self.date_col, self.user_col])]
        user_indices = np.array([schema_user_idx[u] for u in X[self.user_col].values[valid_rows] if u in schema_user_idx])
        features = X[feature_cols].values[valid_rows]
        features[features == ''] = np.nan
        del X
        arr = np.ones(shape=(len(unq_dates), len(schema_users), len(feature_cols)), dtype=self.dtype) * self.fill_value
        for i, date in tqdm(enumerate(unq_dates), total=len(unq_dates)):
            try:
                arr[i, user_indices[indices == i], :] = features[indices == i]
            except TypeError:
                print(f"Gut typerror in {date}")
                continue
            except IndexError:
                print("Got index error")
                continue
        return {'data': arr, self.date_col: unq_dates, self.user_col: schema_users}


class ParsePlayerData(DfTransformer):
    def __init__(self, field_name, use_cols, agg=None):
        self.field_name = field_name
        self.use_cols = use_cols
        self.agg = agg

    def _transform(self, X, y=None):
        dfs = []
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
