import argparse
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from numpy import mean

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from core.player import play_test_episode, play_train_episode
from core import json_config
from config import config_from_args


class GridsearchParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('n', type=int)
        self.add_argument('specs', nargs='+')
        self.add_argument('--n-process', type=int, default=0)


def main(config, verbose=False, exit_on_done=False):
    test_rewards = []
    ep_n = 0
    max_r = -float('inf')

    for ep_n in range(config.n_episodes + 1):
        if ep_n % config.test_frequency == 0:
            avg_test_reward = mean([play_test_episode(config.env, config.test_policy)
                                    for _ in range(config.n_test_episodes)])

            config.manager.tensorboard.add_scalar('test/avg_reward', avg_test_reward, ep_n)
            test_rewards.append(avg_test_reward)

            if verbose:
                print(f"{ep_n}, test reward {avg_test_reward:4.1f}")
        is_solved = config.env.is_solved(test_rewards)
        config.manager.tensorboard.add_scalar('test/is_solved', is_solved, ep_n)

        episode = play_train_episode(config.env, config.policy, config.debug)

        max_r = max(max_r, episode.reward)
        config.manager.tensorboard.add_scalar('training/max_reward', max_r, ep_n)

        config.dataset.append(episode)
        config.train_iter(episode, ep_n, config, verbose)

        if exit_on_done and is_solved:
            return mean(test_rewards), ep_n - 100

    return mean(test_rewards), ep_n - 100


def f(x):
    args, seed, n_threads = x
    set_seed(seed, n_threads)
    return main(config_from_args(args), verbose=False, exit_on_done=args.exit_on_done), args


def s2ms_str(sec):
    return f"{sec // 60:.0f}:{sec % 60:0>2.0f}"


def set_seed(seed, n_threads):
    torch.set_num_threads(n_threads)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed & 0x00000000FFFFFFFF)


def _gridsearch_output(n_done, n_total, t0, out_str_len):
    t = time.time() - t0
    t_per_iter = t / n_done
    sec_remaining = t_per_iter * (n_total - n_done)

    out_str = ''
    out_str += f"\rDone {n_done}/{n_total} in "
    out_str += f"{s2ms_str(t)}; {t_per_iter:.1f}s/iter; "
    out_str += f"ETA {s2ms_str(sec_remaining)}"
    out_str += " " * (out_str_len - len(out_str))

    return out_str


def run_gridsearch(run_arg_list, job_count, thread_count):
    base_seed = torch.LongTensor(1).random_().item()
    seeds = [base_seed + i for i in range(len(run_arg_list))]

    results, out_str, t0 = [], '', time.time()
    for r in mp.Pool(job_count).imap_unordered(f, zip(run_arg_list, seeds, len(seeds) * [thread_count])):
        (mean_reward, ep_n), args = r
        results.append((args.env_size, ep_n))
        with open(f'data_out/{args.tag}_out.pkl', 'wb') as fhandle:
            pickle.dump(results, fhandle)
        out_str = _gridsearch_output(len(results), len(run_arg_list), t0, len(out_str))
        print(out_str, end="")
    print()


if __name__ == '__main__':
    args = GridsearchParser().parse_args()
    if args.n_process:
        job_count = args.n_process
        thread_count = int(math.floor(os.cpu_count() / job_count))
    else:
        job_count = os.cpu_count() - 1
        thread_count = 1
    run_configs = json_config.parse_configs(args.specs) * args.n
    random.shuffle(run_configs)

    try:
        run_gridsearch(run_configs, job_count, thread_count)
    except KeyboardInterrupt:
        exit()
