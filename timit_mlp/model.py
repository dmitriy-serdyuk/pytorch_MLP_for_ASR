import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, options, num_classes):
        super().__init__()
        self.use_batchnorm = options.use_batchnorm

        layers = []

        self.criterion = nn.CrossEntropyLoss()

        dim_pairs = zip([input_dim] + options.hidden_dims[:-1],
                        options.hidden_dims)
        for i, (input_dim, output_dim) in enumerate(dim_pairs):
            fc = nn.Linear(input_dim, output_dim)
            layers.append(fc)

            fc.weight = torch.nn.Parameter(
                torch.Tensor(output_dim, input_dim).uniform_(
                    -np.sqrt(0.01 / (input_dim + output_dim)),
                    np.sqrt(0.01 / (input_dim + output_dim))))
            fc.bias = torch.nn.Parameter(torch.zeros(output_dim))

            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(output_dim, momentum=0.05))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=options.drop_rate))

        fco = nn.Linear(output_dim, num_classes)

        fco.weight = torch.nn.Parameter(torch.zeros(num_classes, output_dim))
        fco.bias = torch.nn.Parameter(torch.zeros(num_classes))

        layers.append(fco)

        self.network = nn.Sequential(*layers)

    def forward(self, x, lab):
        out = self.network(x)

        pout = F.log_softmax(out, dim=1)
        pred = pout.max(dim=1)[1]
        err = torch.sum((pred != lab.long()).float())
        loss = self.criterion(out, lab.long())  # note that softmax is included in nn.CrossEntropyLoss()
        return loss, err, pout, pred
