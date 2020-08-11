import argparse
import re

from torch.optim import Adam

import core.replay
import environments
from core import logpath, policies as policies
from core.manager import TrainingManager
from models.linear_uncertainty import OnlineVariance, OnlineVarianceMulti
from models.nn.losses import QLoss, SuccessorLosses, UBELoss
from models.temporal_difference import QNetwork, SF, BootQNetwork
from run.inner_loops import q_train_iter, sf_train_iter, ube_train_iter


class RunArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('algorithm', choices=('sf', 'ube', 'bdqn', 'boot'))
        self.add_argument('--env', default='grid')
        self.add_argument('--env-size', type=int, default=5)

        self.add_argument('--n-episodes', default=100, type=int)

        self.add_argument('--policy-arg', type=float, default=1.0)
        self.add_argument('--n-grad-steps', default=10, type=int)

        self.add_argument('--lr', default=1e-3, type=float)
        self.add_argument('--target-update-factor', default=1.0, type=float)
        self.add_argument('--discount-factor', default=1.0, type=float)

        self.add_argument('--batch-size', default=100, type=int)
        self.add_argument('--buffer-size', default=10000, type=int)

        self.add_argument('--feature-size', default=10, type=int)
        self.add_argument('--use-network', action='store_true')

        self.add_argument('--prior', default=1.0, type=float)
        self.add_argument('--p-bootstrap', default=0.5, type=float)

        self.add_argument('--print-frequency', default=1, type=int)

        self.add_argument('--name', default='unnamed', type=str)
        self.add_argument('--exit-on-done', action='store_true')
        self.add_argument('--verbose', action='store_true')

        self.add_argument('--debug', action='store_true')
        self.add_argument('--export-debug', action='store_true')
        self.add_argument('--test_mode', action='store_true')


_re_parse_fcall = re.compile(r'([a-z\-_\d]+)(?:\((.+)\))?')


def config_from_args(args):
    _ns = {'n_episodes': args.n_episodes,
           'n_grad_steps': args.n_grad_steps,
           'target_update_factor': args.target_update_factor,
           'print_frequency': args.print_frequency,
           "buffer_size": args.buffer_size,
           'debug': args.debug,
           'discount_factor': args.discount_factor,
           'prior': args.prior}

    # set up manager
    manager = TrainingManager(logpath, args.name, args.test_mode).register_args(args)
    _ns['manager'] = manager

    try:
        # set up data set & env
        _ns.update(get_env(args.env, args.env_size, args.verbose))

        _ns.update(get_data(args.batch_size, _ns['buffer_size'], _ns['env'].state_size, args.algorithm, args.policy_arg,
                            args.p_bootstrap))

        # set up model
        _ns.update(
            get_model(args.algorithm, _ns['env'], args.feature_size, args.use_network, policy_arg=args.policy_arg,
                      prior=args.prior))
        manager.register_modules([_ns['model']])

        # set up policy & uncertainty
        _ns.update(get_uncertainty(args.algorithm, _ns['model'],
                                   _ns['env'].action_size, args.prior, args.policy_arg))
        _ns.update(get_policy(args.algorithm, args.policy_arg, _ns['env'].action_size,
                              _ns['model'], _ns['uncertainty']))

        # set up loss & optimiser
        _ns.update(get_loss(args.algorithm, _ns['model'], _ns['policy'], _ns['discount_factor']))
        _ns.update(get_optim(set(_ns['model'].parameters()), args.lr))

        # set up training loop
        _ns.update(get_train_iter(_ns['algorithm']))

    except:
        manager.delete_log()
        raise

    return argparse.Namespace(**_ns)


def get_loss(algorithm, model, policy, discount_factor):
    if algorithm in ('bdqn', 'boot'):
        loss = QLoss(model, discount_factor)
    elif algorithm == 'sf':
        loss = SuccessorLosses(model, policy, discount_factor)
    elif algorithm == 'ube':
        loss = UBELoss(model, policy, discount_factor)
    else:
        raise ValueError(f'get_loss: algorithm {algorithm} not recognised.')
    return {"loss": loss}


def get_optim(parameters, lr):
    return {"optimiser": Adam(parameters, lr=lr)}


def get_data(batch_size, buffer_size, state_size, algorithm, n_ensembles, p_bootstrap):
    if algorithm == 'boot':
        replay_dataset = core.replay.EnsembleDataset(max_size=buffer_size, n_ensembles=n_ensembles, p=p_bootstrap)
    else:
        replay_dataset = core.replay.UniformDatasetOneHot(state_size, max_size=buffer_size)
    return {'dataset': replay_dataset,
            'sample_fn': core.replay.make_sample_function(replay_dataset, batch_size, algorithm)}


def get_env(env_name, env_arg, verbose):
    if env_name == 'tree':
        env = environments.EnvironmentTree(env_arg)
    elif env_name == 'grid':
        env = environments.EnvironmentGrid(env_arg, verbose)
    else:
        raise ValueError(f'get_env: env {env_name} not recognised.')
    return {"env": env, **env.default_args()}


def get_model(algorithm, env, feature_size, use_network, policy_arg, prior):
    if algorithm in ('bdqn', 'ube'):
        model = QNetwork(env.state_size, env.action_size, feature_size)
    elif algorithm == 'sf':
        model = SF(env.state_size, env.action_size, feature_size, use_network)
    elif algorithm == 'boot':
        model = BootQNetwork(env.state_size, env.action_size, feature_size,
                             prior_weight=prior, n_heads=int(policy_arg))
    else:
        raise ValueError(f'get_model: algorithm {algorithm} not recognised.')
    return {"model": model, 'algorithm': algorithm}


def get_policy(algorithm, policy_arg, action_size, model, uncertainty):
    if algorithm in ('sf', 'ube', 'bdqn'):
        policy = policies.ThompsonPolicy(action_size, model.q_fn, model.compute_q_fn_external, model.get_weights,
                                         uncertainty, policy_arg, model.ube_fn if algorithm == 'ube' else None)
    elif algorithm == 'boot':
        policy = policies.BootstrapPolicy(action_size, model, model.next_head)
    return {"policy": policy, "test_policy": policies.GreedyPolicy(action_size, model)}


def get_uncertainty(algorithm, model, action_size, variance_prior, varuance_likelihood):
    if algorithm.startswith('sf'):
        uncertainty = OnlineVariance(
            model.feature_size, train_featuriser=model.local_features, test_featuriser=model.global_features,
            prior_variance=variance_prior, likelihood_variance=varuance_likelihood, bias=False)
        model.register_weights(uncertainty.mean_vector)
    elif algorithm in ('ube', 'bdqn'):
        uncertainty = OnlineVarianceMulti(
            model.feature_size, action_size, model.local_features, variance_prior, varuance_likelihood, bias=False)
    else:
        uncertainty = None
    return {"uncertainty": uncertainty}


def get_train_iter(algorithm):
    if algorithm in ('bdqn', 'boot'):
        train_iter = q_train_iter
    elif algorithm == 'sf':
        train_iter = sf_train_iter
    elif algorithm == 'ube':
        train_iter = ube_train_iter
    else:
        raise ValueError(f'get_train_iter: algorithm {algorithm} not recognised.')
    return {"train_iter": train_iter}


default_config, _ = RunArgParser().parse_known_args(['bdqn'])
