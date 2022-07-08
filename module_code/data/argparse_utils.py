from argparse import ArgumentParser, Namespace, Action
from typing import Dict, List, Optional
from json import loads
import re


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
            # strip any {<space>, ', " [, ]}" and then split by comma
            values = re.sub(r"[ '\"\[\]]", "", values).split(",")
            if choices:
                values = [convert(x) for x in values if x in choices]
            else:
                values = [convert(x) for x in values]
            setattr(namespace, self.dest, values)

    return ConvertToList


def dict_string_to_json(dict_string: str) -> Dict:
    return loads(dict_string.replace("'", '"'))


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
            if choices:
                dict_obj = {
                    key: value
                    for key, value in dict_string_to_json(dict_string).items()
                    if key in choices
                }
            else:
                dict_obj = dict_string_to_json(dict_string)
            setattr(namespace, self.dest, dict_obj)

    return ConvertToDict
