import json
import logging
import os
import shutil
import sys
import time
import uuid

import tensorboardX
import torch


def _get_rand_str(len=6):
    return uuid.uuid4().hex[-len:]


class TrainingManager:
    def __init__(self, logdir, tag, leave_no_trace=False):
        self.leave_no_trace = leave_no_trace
        if leave_no_trace:
            logdir = '/tmp'

        self.log_path = TrainingManager.make_log_path(logdir, tag)
        os.makedirs(self.log_path, exist_ok=True)

        self.checkpoint_base_path = os.path.join(self.log_path, 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.logger = self.setup_logging(self.log_path)
        self.tensorboard = tensorboardX.SummaryWriter(self.log_path)

        self._registered_modules = []

    def __del__(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        if self.leave_no_trace:
            self.delete_log()

    def register_args(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        arg_file_path = os.path.join(self.log_path, "run_args.json")
        with open(arg_file_path, 'w') as fhandle:
            json.dump(args, fhandle)
        return self

    def register_module(self, m):
        self._registered_modules.append(m)

    def register_modules(self, ms):
        for m in ms: self.register_module(m)

    def checkpoint(self, step):
        save_dict = {i: m.state_dict() for i, m in
                     enumerate(self._registered_modules)}
        torch.save(save_dict, self.checkpoint_path(step))

    def restore(self, step):
        save_dict = torch.load(self.checkpoint_path(step))
        for i, m in enumerate(self._registered_modules):
            m.load_state_dict(save_dict[i])

    def checkpoint_path(self, step):
        return os.path.join(self.checkpoint_base_path, str(step))

    def delete_log(self):
        shutil.rmtree(self.log_path)

    @staticmethod
    def make_log_path(logdir, tag):
        time_str = time.strftime("%d-%H_%M_%S", time.localtime())
        rand_str = _get_rand_str(6)
        fname = f"{tag}_{time_str}_{rand_str}" \
            if tag is not None else f"{time_str}_{rand_str}"
        return os.path.join(logdir, fname)

    @staticmethod
    def setup_logging(logpath):
        logger = logging.getLogger()
        logger.propagate = False
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s::%(message)s')

        hdlr = logging.FileHandler(os.path.join(logpath, 'out.log'))
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        hdlr = logging.StreamHandler(stream=sys.stdout)
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        return logger
