
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import softor, weight_sum


def init_identity_weights(X, device):
    ones = torch.ones((X.size(0), ), dtype=torch.float32) * 100
    return torch.diag(ones).to(device)


class InferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk = None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(InferModule, self).__init__()
        self.I = I
        self.I_bk = I_bk
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        self.beta = 0.01 #softmax temperature
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.tensor(
                #np.random.normal(size=(m, I.size(0))), requires_grad=True, dtype=torch.float32).to(device))
                np.random.rand(m, I.size(0)), requires_grad=True, dtype=torch.float32).to(device))
        # clause functions
        self.cs = [ClauseFunction(I[i], gamma=gamma)
                   for i in range(self.I.size(0))]


        if not I_bk is None:
            self.cs_bk = [ClauseFunction(I_bk[i], gamma=gamma)
                       for i in range(self.I_bk.size(0))]
            self.W_bk = init_identity_weights(I_bk, device)

        #print("W: ", self.W.shape)
        #print("W_bk: ", self.W_bk)

        #assert m == self.C, "Invalid m and C: " + \
        #    str(m) + ' and ' + str(self.C)

    def get_params(self):
        assert self.train_, "Infer module is not in training mode."
        return [self.W]

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        R = x
        if self.I_bk is None:
            for t in range(self.infer_step):
                R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        else:
            for t in range(self.infer_step):
                #R = softor([R, self.r_bk(R)], dim=1, gamma=self.gamma)
                R = softor([R, self.r(R), self.r_bk(R)], dim=1, gamma=self.gamma)
        return R

    def r(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        C = torch.stack([self.cs[i](x)
                        for i in range(self.I.size(0))], 0)

        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        #W_star = torch.softmax(self.W * (1 / self.beta), 1)
        W_star = torch.softmax(self.W, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        H = torch.sum(W_tild * C_tild, dim=1)
        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        return R

    def r_bk(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        C = torch.stack([self.cs_bk[i](x)
                        for i in range(self.I_bk.size(0))], 0)
        # B * G
        return softor(C, dim=0, gamma=self.gamma)
        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        W_star = torch.softmax(self.W_bk, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        #H = torch.sum(W_tild * C_tild, dim=1)


        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        return R


class ClauseInferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk = None):
        """
        Infer module using each clause.
        The result is not amalgamated in terms of clauses.
        """
        super(ClauseInferModule, self).__init__()
        self.I = I
        self.I_bk = I_bk
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0)))).to(device))
        # clause functions
        self.cs = [ClauseFunction(I[i], gamma=gamma)
                   for i in range(self.I.size(0))]

        if not self.I_bk is None:
            self.cs_bk = [ClauseFunction(I_bk[i], gamma=gamma)
                   for i in range(self.I_bk.size(0))]

        if not I_bk is None:
            self.W_bk = init_identity_weights(I_bk, device)

        assert m == self.C, "Invalid m and C: " + \
            str(m) + ' and ' + str(self.C)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        B = x.size(0)
        # C * B * G
        R = x.unsqueeze(dim=0).expand(self.C, B, self.G)
        if self.I_bk is None:
            for t in range(self.infer_step):
                R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        else:
            for t in range(self.infer_step):
                # infer by background knowledge
                #r_bk = self.r_bk(R[0])
                #R_bk = self.r_bk(r_bk).unsqueeze(dim=0).expand(self.C, B, self.G)
                #R = R_bk
                #print("R: ", R.shape)
                #print("r(R): ", self.r(R).shape)
                #print("r_bk(R): ", self.r_bk(R).shape)
                # shape? dim?
                R = softor([R, self.r(R), self.r_bk(R).unsqueeze(dim=0).expand(self.C, B, self.G)], dim=2, gamma=self.gamma)
        return R

    def r(self, x):
        # x: C * B * G
        B = x.size(1)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # infer from i-th valuation tensor using i-th clause
        C = torch.stack([self.cs[i](x[i])
                        for i in range(self.I.size(0))], 0)
        return C

    def r_bk(self, x):
        x = x[0]
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # just use the first row
        C = torch.stack([self.cs_bk[i](x)
                        for i in range(self.I_bk.size(0))], 0)
        # B * G
        return softor(C, dim=0, gamma=self.gamma)
        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        W_star = torch.softmax(self.W_bk, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        H = torch.sum(W_tild * C_tild, dim=1)
        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        return R


class ClauseFunction(nn.Module):
    """
    A class of the clause function.
    """

    def __init__(self, I_i, gamma=0.01):
        super(ClauseFunction, self).__init__()
        #self.i = i  # clause index
        self.I_i = I_i  # index tensor C * S * G, S is the number of possible substituions
        self.L = I_i.size(-1)  # number of body atoms
        self.S = I_i.size(-2)  # max number of possible substitutions
        self.gamma = gamma

    def forward(self, x):
        batch_size = x.size(0)  # batch size
        # B * G
        V = x
        # G * S * b
        #I_i = self.I[self.i, :, :, :]

        # B * G -> B * G * S * L
        V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
        # G * S * L -> B * G * S * L
        I_i_tild = self.I_i.repeat(batch_size, 1, 1, 1)

        # B * G
        C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3),
                   dim=2, gamma=self.gamma)
        return C
