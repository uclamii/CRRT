from argparse import ArgumentParser, Namespace, Action
from typing import Dict, List, Optional
from json import loads
import re


def string_list_to_list(
    list_string: str, choices: Optional[List[str]] = None, convert: type = str
) -> List:
    # strip any {<space>, ', " [, ]}" and then split by comma
    values = re.sub(r"[ '\"\[\]]", "", list_string).split(",")
    if choices:
        values = [convert(x) for x in values if x in choices]
    else:
        values = [convert(x) for x in values]
    return values


def string_dict_to_dict(dict_string: str, choices: Optional[List[str]] = None) -> List:
    dict_obj = loads(dict_string.replace("'", '"'))
    if choices:  # filter down
        dict_obj = {key: value for key, value in dict_obj.items() if key in choices}
    return dict_obj


def YAMLStringListToList(convert: type = str, choices: Optional[List[str]] = None):
    class ConvertToList(Action):
        """Takes a comma separated list (no spaces) from command line and parses into list of some type (Default str)."""

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: str,
            option_string: Optional[str] = None,
        ):
            setattr(namespace, self.dest, string_list_to_list(values, choices, convert))

    return ConvertToList


def YAMLStringDictToDict(choices: Optional[List[str]] = None):
    class ConvertToDict(Action):
        """Takes a comma separated list (no spaces) from command line and parses into list of some type (Default str)."""

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            dict_string: str,
            option_string: Optional[str] = None,
        ):
            setattr(namespace, self.dest, string_dict_to_dict(dict_string, choices))

    return ConvertToDict
