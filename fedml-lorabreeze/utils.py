import types
import yaml

from fedml.utils.logging import logger


def load_endpoint(filepath: str) -> types.SimpleNamespace:
    with open(filepath, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            yaml_obj = types.SimpleNamespace(**yaml_dict)
            return yaml_obj
        except yaml.YAMLError as err:
            logger.error(err)
