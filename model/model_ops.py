import torch
import torch.nn as nn

class Conv1dPool(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Conv1dPool, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 1, kernel_size//2, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

def get_block(block_type,
              kernel_size,
              in_channels,
              out_channels):

    if block_type == "conv1d":
        block = Conv1dPool(kernel_size, in_channels, out_channels)

    elif block_type == "linear":
        block = nn.Linear(in_channels, out_channels)

    else:
        raise

    return block
