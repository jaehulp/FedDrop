import os
import sys
import copy
import time
import yaml
import shutil
import random
import logging
import argparse
import torch
import numpy as np

from dotmap import DotMap

from utils.fed.server import set_server
from torch.utils.tensorboard import SummaryWriter

def main(args):
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, 'client'), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"Configuration: {args}")
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='w'))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    ServerBase = set_server(args.server.base)
    FLServer = ServerBase(args, logger=logger, writer=writer)
    FLServer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, _ = parser.parse_known_args()
    config_path = args.config

    with open(config_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = DotMap(args)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(output_dir, 'config.yaml'))

    if args.seed != False:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)

