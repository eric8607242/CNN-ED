import torch
import torch.nn as nn

from .model_ops import get_block


def get_model_config(n_features):
                    # block_type, kernel_size, out_channels
    model_config = {"main" : [["conv1d_pool", 3, 8],
                              ["conv1d_pool", 3, 8]],
                    "last" : [["linear", None, n_features]]
                    }
    return model_config


class Model(nn.Module):
    def __init__(self, model_config, alphabet_len, max_str_len):
        super(Model, self).__init__()

        pool_size = 1
        in_channels = 1
        main_stage = []
        for l_cfg in model_config["main"]:
            block_type, kernel_size, out_channels = l_cfg

            block = get_block(block_type=block_type,
                              kernel_size=kernel_size,
                              in_channels=in_channels,
                              out_channels=out_channels)
            main_stage.append(block)

            pool_size = 2*pool_size if block_type == "conv1d_pool" else pool_size
            in_channels = out_channels 

        self.main_stage = nn.ModuleList(main_stage)

        in_channels = max_str_len // pool_size * alphabet_len * in_channels
        self.flatten_in_channels = in_channels

        last_stage = []
        for l_cfg in model_config["last"]:
            block_type, kernel_size, out_channels = l_cfg

            block = get_block(block_type=block_type,
                              kernel_size=kernel_size,
                              in_channels=in_channels,
                              out_channels=out_channels)
            last_stage.append(block)
            in_channels = out_channels 
        self.last_stage = nn.ModuleList(last_stage)

        self.max_str_len = max_str_len


    def forward(self, x):
        N = x.shape[0]
        x = x.view(-1, 1, self.max_str_len)
        for l in self.main_stage:
            x = l(x)

        x = x.view(N, -1)
        for l in self.last_stage:
            x = l(x)

        return x
