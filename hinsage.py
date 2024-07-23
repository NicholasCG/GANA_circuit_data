import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

class HinSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_types, concat='partial', dropout=0.0):
        super().__init__(aggr="mean")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_types = edge_types
        self.concat = concat
        self.dropout = dropout

        self.lin_self = Linear(in_channels, out_channels, bias = False)
        self.lins_neigh = ModuleList([Linear(in_channels, out_channels, bias = False) for _ in edge_types])

        if concat == 'partial':
            self.lin_cat = Linear(out_channels * 2, out_channels, bias=True)
        elif concat == 'full':
            self.lin_cat = Linear(out_channels * (1 + len(edge_types)), out_channels, bias=True)
        elif concat == 'none':
            self.lin_cat = Linear(out_channels, out_channels, bias=True)
        else:
            raise ValueError("concat should be one of 'partial', 'full', or 'none'.")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        for lin in self.lins_neigh:
            lin.reset_parameters()
        self.lin_cat.reset_parameters()

    def forward(self, x, edge_index_dict):
        out = self.lin_self(x)

        outs = [out]

        for i, edge_type in enumerate(self.edge_types):
            edge_index = edge_index_dict[edge_type]
            out_neigh = self.propagate(edge_index, x=x, size=None)
            out_neigh = self.lins_neigh[i](out_neigh)
            outs.append(out_neigh)

        if self.concat == 'partial':
            out = torch.cat([outs[0], torch.mean(torch.stack(outs[1:]), dim=0)], dim=1)
        elif self.concat == 'full':
            out = torch.cat(outs, dim=1)
        elif self.concat == 'none':
            out = outs[0] + torch.mean(torch.stack(outs[1:]), dim=0)

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin_cat(out)

        return out

    def message(self, x_j):
        return x_j

    def update(self, inputs):
        return inputs