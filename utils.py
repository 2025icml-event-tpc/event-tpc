import re
import numpy as np
import polars as pl
import torch

from ruamel.yaml import YAML
yaml = YAML()

import logging
logging.basicConfig(format="%(asctime)s[%(levelname)s]: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def load_yaml(path: str):
    with open(path, 'r') as f:
        data = yaml.load(f)

    return data


def load_estp(data_dir: str):
    import os
    import datetime
    from preprocessing import EventTPTokenizer

    tokenizer = EventTPTokenizer.load(os.path.join(data_dir, "event_tokens.json"))
    
    with torch.serialization.safe_globals([
        datetime.date, np.ndarray, np.core.multiarray._reconstruct, 
        np.dtype, np.dtypes.ObjectDType, np.dtypes.Int64DType, np.dtypes.Float32DType
    ]):
        estp_data = torch.load(os.path.join(data_dir, "data.pt"), weights_only=True)

    return tokenizer, estp_data
