import os
import json
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import utils

log = utils.get_logger(__name__)


def prepare_data(config: DictConfig):
    """Prepare all data artifacts required for feature engineering.

    Care need to be taken to avoid temporal information leakage.
    """
    # Map playerid to integers representing columns in pivoted data.
    playerid_mapper = hydra.utils.instantiate(config.prepare_playeridartifact)
    playerid_mapper.transform('players.csv')

    # Create teamId mappings
    artifacts_3d = hydra.utils.instantiate(config.prepare_3Dartifacts)
    artifacts_3d.transform('train.csv')

    # Prepare 3D matrix of previous targets

    # Prepare 3D matrix of awards

    # Prepare 3D matrix of playerBoxScore

    # Prepare 3D matrix of rosters

    # Prepare 3D matrix of teamScores

    # Prepare 3D matrix of twitterFollowers

    # Prepare 3D matrix of  