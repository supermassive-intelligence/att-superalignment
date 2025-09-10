from infra.salign.util.default_config import Config

import os
import yaml


def get_config():
    loaded_config = {}
    return Config(**loaded_config).dict()
