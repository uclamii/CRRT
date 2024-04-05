"""
Manual grid search
We have moved to optuna
"""

from random import sample
from itertools import product
import itertools
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def random_search(num_searches=40, **kwargs):
    """
    This implements random search for hyper-parameter search.
    Parameters
    ----------
    num_searches : int
        The number of hyper-parameter searches.
    kwargs : dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values
    Returns
    -------
    list
        A list of dicts in which each dict is a set of hyper-parameter choices for all the hyper-parameters
    Notes
    ------
    Lot of content taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists.
    """
    # hyper-parameter arg names
    hyps = kwargs.keys()
    # hyper-parameter options
    hyps_vals = kwargs.values()
    params_dts = []
    params_dts_json = []

    # if num_searches is greater than 80% cartesian product set size, then call grid search instead
    # NOTE: 80% is used so random search doesn't take forever to find a not selected set of hyper-parameter choices
    cs_prod_len = float(np.prod([len(remove_duplicates(val)) for val in hyps_vals]))
    if num_searches > 0.80 * cs_prod_len:
        return grid_search(**kwargs)
    for _ in range(num_searches):
        not_contained = False
        # check to make sure hyper-parameter options haven't previously been seen
        while not not_contained:
            # retrieve the arg-values tuples for all hyper-parameter args in random order
            hyps_and_hyps_vals = sample(list(zip(hyps, hyps_vals)), len(hyps))
            hyps, hyps_vals = zip(*hyps_and_hyps_vals)
            # randomize the hyper-parameter options for each hyper-parameter arg
            hyps_vals = [sample(val, len(val)) for val in hyps_vals]
            # get one set of possible hyper-parameters for a model
            instance = next(product(*hyps_vals))
            # convert the instance to a dictionary
            params_dt = dict(zip(hyps, instance))
            # get the json string from the dict
            params_dt_json = json.dumps(params_dt, sort_keys=True)
            # add the instance to the experiment list if json hasn't been seen
            if params_dt_json not in params_dts_json:
                params_dts.append(params_dt)
                params_dts_json.append(params_dt_json)
                not_contained = True
    return params_dts


def grid_search(**kwargs):
    """
    This implements grid search for hyper-parameter search.
    Parameters
    ----------
    kwargs : dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values
    Returns
    -------
    list
        A list of dicts in which each dict is a set of hyper-parameter choices for all the hyper-parameters
    Notes
    ------
    Lot of content taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists.
    """
    # hyper-parameter arg names
    keys = kwargs.keys()
    # hyper-parameter options
    vals = kwargs.values()
    searches = []
    # get cartesian product from all hyper-parameter options
    for instance in itertools.product(*vals):
        searches.append(dict(zip(keys, instance)))
    return searches


def remove_duplicates(lst):
    dup = set()
    new_lst = []
    for it in lst:
        json_it = json.dumps(it, sort_keys=True)
        if json_it not in dup:
            new_lst += [it]
            dup.add(json_it)
    return new_lst
