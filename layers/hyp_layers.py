import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):

    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()

        self.num_gcn_layers = num_layers - 1

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def res_gcn(self, inputs):
        x_tangent, adj = inputs
        x_tangent_0 = x_tangent
        output = []
        for i in range(int((self.num_gcn_layers + 1) / 2)):
            x_tangent_1 = torch.spmm(adj, x_tangent_0)
            x_tangent_2 = torch.spmm(adj, x_tangent_1)

            x_tangent_0 = (x_tangent_1 + x_tangent_2)
            output.append(x_tangent_0)
        return sum(output)

    def res_gcn_v1(self, inputs):
        x_tangent, adj = inputs
        x_tangent_0 = x_tangent
        output = []
        for i in range(int((self.num_gcn_layers + 1) / 2)):
            x_tangent_1 = torch.spmm(adj, x_tangent_0)
            x_tangent_2 = torch.spmm(adj, x_tangent_1)

            x_tangent_0 = (x_tangent_1 + x_tangent_2)
            output.append(x_tangent_1)
            output.append(x_tangent_2)
        return sum(output)


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        output = self.stackGCNs((x_tangent, adj))
        output = output - output.mean(dim=0)
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
