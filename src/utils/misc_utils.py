from time import time
from typing import *


def stopwatch():
    started_at = time()
    return lambda: time() - started_at


def dt_str(dt: float) -> str:
    """
    :param dt: time in seconds
    """
    if dt >= 0.1:
        return f"{dt:.1f} s"
    ms = dt * 1000
    if ms >= 0.1:
        return f"{ms:.1f} ms"
    ns = ms * 1000000
    return f"{int(ns)} ns"


def stopwatch_str():
    stopwatch_ = stopwatch()
    return lambda: dt_str(stopwatch_())


def get_value_with_path(dict_: Dict[str, Any], path: str):
    """ Gets the value from a dict given the path in the form of `path.to.value`. """
    steps = path.split(".")
    current = dict_
    for step in steps:
        current = current[step]
    return current


def parse_dict_with_vars(current_dict: Dict[str, Any], original_dict: Optional[Dict[str, Any]] = None):
    """ Replaces the variables in the form of "{path.to.value}" in the given dictionary. """
    if original_dict is None:
        original_dict = current_dict
    for k, v in current_dict.items():
        if isinstance(v, str):
            if len(v) > 2 and v[0] == '{' and v[-1] == '}':
                current_dict[k] = get_value_with_path(original_dict, v[1:-1])
        elif isinstance(v, dict):
            current_dict[k] = parse_dict_with_vars(current_dict[k], original_dict)
    return current_dict


def load_json_with_vars(json_filepath: str) -> Dict[str, Any]:
    """ Loads a json capable of having variables! """
    import json

    with open(json_filepath, "r") as f:
        return parse_dict_with_vars(json.load(f))
