
import random

import torch
import torch.nn as nn

from torch_utils import softor, weight_sum
from infer import ClauseFunction

class EvalInferModule(nn.Module):
    """
    A class of differentiable foward-chaining inference.
    """

    def __init__(self, I, m, infer_step, gamma=0.01, device=None, train=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(InferModule, self).__init__()
        self.I = I
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        # clause functions
        self.cs = [ClauseFunction(i, I, gamma=gamma)
                   for i in range(self.I.size(0))]

        assert m == self.C, "Invalid m and C: " + \
            str(m) + ' and ' + str(self.C)

    def init_identity_weights(self, device):
        ones = torch.ones((self.C, ), dtype=torch.float32) * 100
        return torch.diag(ones).to(device)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        R = x
        for t in range(self.infer_step):
            R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        return R

    def r(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        C = torch.stack([self.cs[i](x)
                        for i in range(self.I.size(0))], 0)
        # taking weighted sum using m weights and stack to a tensor H
        # taking soft or to compose a logic program with m clauses
        # C * B * G
        return C