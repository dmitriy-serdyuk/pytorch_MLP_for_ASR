import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, N_hid, drop_rate, use_batchnorm, num_classes):
        super(MLP, self).__init__()
        self.N_hid = N_hid
        self.use_batchnorm = use_batchnorm

        # list initialization
        self.hidden = nn.ModuleList([])
        self.droplay = nn.ModuleList([])
        self.bnlay = nn.ModuleList([])
        self.criterion = nn.CrossEntropyLoss()

        curr_in_dim = input_dim
        for i in range(N_hid):
            fc = nn.Linear(curr_in_dim, hidden_dim)
            fc.weight = torch.nn.Parameter(
                torch.Tensor(hidden_dim, curr_in_dim).uniform_(-np.sqrt(0.01 / (curr_in_dim + hidden_dim)),
                                                               np.sqrt(0.01 / (curr_in_dim + hidden_dim))))
            fc.bias = torch.nn.Parameter(torch.zeros(hidden_dim))
            curr_in_dim = hidden_dim
            self.hidden.append(fc)
            self.droplay.append(nn.Dropout(p=drop_rate))
            if use_batchnorm:
                self.bnlay.append(nn.BatchNorm1d(hidden_dim, momentum=0.05))

        self.fco = nn.Linear(curr_in_dim, num_classes)
        self.fco.weight = torch.nn.Parameter(torch.zeros(num_classes, curr_in_dim))
        self.fco.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, lab):
        out = x
        for i in range(self.N_hid):
            fc = self.hidden[i]
            drop = self.droplay[i]

            if self.use_batchnorm:
                batchnorm = self.bnlay[i]
                out = drop(F.relu(batchnorm(fc(out))))
            else:
                out = drop(F.relu(fc(out)))

        out = self.fco(out)
        pout = F.log_softmax(out, dim=1)
        pred = pout.max(dim=1)[1]
        err = torch.sum((pred != lab.long()).float())
        loss = self.criterion(out, lab.long())  # note that softmax is included in nn.CrossEntropyLoss()
        return loss, err, pout, pred
