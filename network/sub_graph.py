import torch
from torch import nn
import torch.nn.functional as F
import config
from network.mlp import MLP


class SubGraph(nn.Module):
    def __init__(self, v_len, layers_number):
        super(SubGraph, self).__init__()
        self.layers = nn.Sequential()      
        for i in range(layers_number):
            self.layers.add_module("sub{}".format(i), SubGraphLayer(v_len * (2 ** i)))
        self.v_len = v_len
        self.layers_number = layers_number

    def forward(self, x):
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        x = self.layers(x).to(config.device) # [batch_size, v_number, p_len]
        x = x.permute(0, 2, 1)  # [batch size, p_len, v_number]
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.permute(0, 2, 1)  # [batch size, 1, p_len]
        x.squeeze_(1)
        assert x.shape == (batch_size, self.v_len * (2 ** self.layers_number))
        return x


class SubGraphLayer(nn.Module):

    def __init__(self, len):
        super(SubGraphLayer, self).__init__()
        self.g_enc = MLP(len, len).to(config.device)

    def forward(self, x):
        assert len(x.shape) == 3
        x=x.to(torch.float32)
        x = self.g_enc(x)
        batch_size, n, length = x.shape
        x2 = x.permute(0, 2, 1) # [batch_size, len, n]
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, len, 1]
        x2 = torch.cat([x2] * n, dim=2)  # [batch_size, len, n]
     
        y = torch.cat((x2.permute(0, 2, 1), x), dim=2)
        assert y.shape == (batch_size, n, length*2)
        return y
