import os, sys

import torch

import numpy as np
np.set_printoptions(threshold=np.nan)

from config_parser import Config
from model import Model

def print_out_paramater(config):
    """Print out model's parameter value
    """
    mode_name = sys.argv[2]
    model_path = os.path.join(config.working_dir, mode_name)

    param_dict = torch.load(model_path, map_location='cpu')
    model = Model(config)
    model.load_state_dict(param_dict)

    out_file = open(os.path.join(config.working_dir, '{}_checker.txt'.format(mode_name)), 'w')

    for name, param_value in model.named_parameters():
        out_file.write(name + '\n')
        out_file.write(str(param_value.data.numpy()) + '\n')
        out_file.write('='*50)
        out_file.write('\n')


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = Config(config_file)
    print_out_paramater(config)
