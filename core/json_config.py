import argparse
import itertools
import json
import random
import re

from config import default_config

_re_check_json = re.compile(r'{.+?}')
_parse_json = re.compile(r"({.*?})")


def name_rd(rd):
    return "_".join([f"{k}-{v}" for k, v in rd.items()])


def _try_parse_json(str):
    if re.fullmatch(_re_check_json, str) is not None:
        return json.loads(str.strip("'"))
    return False


def _listify_values(input_dict):
    new_dict = {}
    for key, value in input_dict.items():
        new_value = value
        if not isinstance(value, (list, tuple)):
            new_value = [value]
        new_dict[key] = new_value
    return new_dict


def _productify_values(input_dict):
    return (dict(zip(input_dict.keys(), arg_list)) for arg_list in
            itertools.product(*input_dict.values()))


def _create_run_args(base_args, replacement_dict):
    run_args = vars(base_args)
    run_args.update(replacement_dict)
    run_args = argparse.Namespace(**run_args)
    run_name = name_rd(replacement_dict)
    run_args.name = run_name
    return run_args


def _make_configs(base_args, gs_config):
    replacement_dicts = _productify_values(_listify_values(gs_config))

    run_arg_list = [_create_run_args(base_args, replacement_dict)
                    for replacement_dict in replacement_dicts]
    return run_arg_list


def _try_open_json(path):
    try:
        print("path", path)
        with open(path, 'rb') as fhandle:
            return json.load(fhandle)
    except FileNotFoundError:
        return False


def make_queries(args):
    queries = [json.loads(qs) for qs in re.findall(_parse_json, " ".join(args))]
    return itertools.chain.from_iterable(_productify_values(_listify_values(q)) for q in queries)


def parse_configs(config_strs_or_file_paths, shuffle=False):
    gs_configs = []
    for entry in config_strs_or_file_paths:
        config_f = _try_open_json(entry) or _try_parse_json(entry)
        if config_f:
            gs_configs.append(config_f)

    run_configs = list(itertools.chain.from_iterable(
        [_make_configs(default_config, gs_config) for gs_config in gs_configs]))

    if shuffle:
        random.shuffle(run_configs)

    return run_configs
