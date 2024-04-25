import numpy as np
import torch
import torch.nn as nn

import manifolds
import models.encoders as encoders
from hgcn_utils.math_utils import arcosh
from utils.helper import default_device
import torch.nn.functional as F
import scipy.sparse as sp
from layers.layers import FermiDiracDecoder


class HRCFModel(nn.Module):

    def __init__(self, users_items, args, data):
        super(HRCFModel, self).__init__()

        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, "HRCF")(self.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.args = args
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(default_device())

        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))

        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)

        self.alpha = args.alpha
        # 增加参数
        self.data = data
        self.loss_coe = args.loss_coe
        self.origin = torch.zeros(args.embedding_dim).to(default_device()).view(1, -1)
        self.origin[0, 0] = 1.
        # 增强图
        self.ratio = args.ratio
        self.ssl_reg = args.ssl_reg
        self.user_temperature = args.user_temperature
        self.item_temperature = args.item_temperature

    def encode(self, adj):
        x = self.embedding.weight
        if torch.cuda.is_available():
           adj = adj.to(default_device())
           x = x.to(default_device())
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return [sqdist, probs]

    def geometric_regularizer(self, embeddings):
        embeddings_tan = self.manifold.logmap0(embeddings, c=1.0)
        item_embeddings = embeddings_tan[self.num_users:]
        item_mean_norm = ((1e-6 + item_embeddings.pow(2).sum(dim=1)).mean()).sqrt()
        return 1.0 / item_mean_norm

    def hyperbolic_geometric_regularizer(self, embeddings):
        """tlc 双曲空间 距离中心散开"""
        item_embeddings = embeddings[self.num_users:, :]
        distance = self.manifold.sqdist(item_embeddings, self.origin, c=self.c)
        item_dist = torch.sqrt((1e-6 + distance).mean())
        return 1.0 / item_dist

    def ranking_loss(self, pos_sqdist, neg_sqdist, ):
        loss = pos_sqdist - neg_sqdist + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss

    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]

        # sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        sampled_false_edges_list = [triples[:, [0, 3 + i]] for i in range(self.args.num_two_order)]

        pos = self.decode(embeddings, train_edges)
        pos_sqdist, pos_probs = pos
        neg = self.decode(embeddings, sampled_false_edges_list[0])
        neg_sqdist, neg_probs = neg

        ranking_loss = self.ranking_loss(pos_sqdist, neg_sqdist)
        gr_loss = self.geometric_regularizer(embeddings)

        return ranking_loss * self.loss_coe + self.alpha * gr_loss

    def compute_loss_v1(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]

        pos = self.decode(embeddings, train_edges)
        pos_sqdist, pos_probs = pos
        neg = self.decode(embeddings, sampled_false_edges_list[0])
        neg_sqdist, neg_probs = neg

        ranking_loss = self.ranking_loss(pos_sqdist, neg_sqdist)
        # gr_loss = self.geometric_regularizer(embeddings)

        return ranking_loss  # + self.alpha * gr_loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix

    # =================== 增加的常用方法 ========================
    def two_order_sampling(self, train_edges):
        """二阶采样"""
        user_index = train_edges[:, 0]
        user_item_two_order_sample = np.zeros(len(user_index)).astype(int)
        for i in range(len(user_index)):
            user_item_two_order_sample[i] = self.data.user_item_two_order_pairs_set[user_index[i]][
                np.random.randint(0, len(self.data.user_item_two_order_pairs_set[user_index[i]]))]
        return user_item_two_order_sample

    def dist_sq(self, emb1, emb2):
        """距离计算"""
        minkowski_dot = torch.matmul(emb1, emb2.T) - \
                        2 * torch.matmul(emb1[:, 0].view(-1, 1), emb2[:, 0].view(1, -1))
        # theta = -minkowski_dot * self.c
        theta = torch.clamp(-minkowski_dot * self.c, min=1.0 + self.manifold.eps[torch.float32])
        return torch.clamp(arcosh(theta) ** 2 / self.c, max=50.0)

    def info_nce(self, emb1, emb2, triples):
        """对比学习"""
        pos_edge = triples[:, [0, 1]]
        neg_edge = [triples[:, [2 + i]] for i in range(self.args.num_neg)][0]

        item_emb1 = emb1[pos_edge[:, 1]]
        item_emb2 = emb2[pos_edge[:, 1]]

        # neg_item_emb1 = emb1[neg_edge[:, 0]]
        # neg_item_emb2 = emb2[neg_edge[:, 0]]

        user_emb1 = emb1[pos_edge[:, 0]]
        user_emb2 = emb2[pos_edge[:, 0]]

        item_sq_dist = self.dist_sq(item_emb1, item_emb2)
        # neg_item_dist = self.dist_sq(neg_item_emb1, neg_item_emb2)
        user_sq_dist = self.dist_sq(user_emb1, user_emb2)

        item_ssl_loss = torch.sum(torch.logsumexp(
            (torch.diag(item_sq_dist).view(-1, 1) - item_sq_dist) / self.item_temperature, dim=1))
        # neg_item_ssl_loss = torch.sum(torch.logsumexp(
        #     (torch.diag(neg_item_dist).view(-1, 1) - neg_item_dist) / self.item_temperature, dim=1))
        user_ssl_loss = torch.sum(torch.logsumexp(
            (torch.diag(user_sq_dist).view(-1, 1) - user_sq_dist) / self.user_temperature, dim=1))
        return self.ssl_reg * (item_ssl_loss + user_ssl_loss)
        # return self.ssl_reg * item_ssl_loss

    def pos_item_contrast(self, embeddings, triples):
        pos_edge = triples[:, [0, 1]]
        pos_emb = embeddings[pos_edge[:, 1]]
        sampled_false_edges_list = [triples[:, [4 + i]] for i in range(self.args.item_item_num)]

        sq_dist = [torch.zeros((len(pos_emb), 1)).to(default_device())]
        for sampled_false_edges in sampled_false_edges_list:
            neg_emb = embeddings[sampled_false_edges[:, 0]]
            sq_dist.append(self.manifold.sqdist(pos_emb, neg_emb, c=self.c))

        item_item_sq_dist = torch.cat(sq_dist, dim=1)
        item_ssl_loss = torch.sum(torch.logsumexp(-item_item_sq_dist / self.temperature, dim=1))
        return self.pos_item_reg * item_ssl_loss

    def create_adj_mat(self):
        """创建增强图"""
        indices = self.data.indices
        # torch.nonzero(self.data.user_item)

        # 打乱并删除部分边
        idx = torch.randperm(indices.shape[0])
        indices = indices[idx, :]
        indices = indices[int(indices.shape[0] * self.ratio):, :]

        # 创建增强图
        indices[:, 1] += self.num_users  # self.data.num_users  # self.data.user_item.shape[0]
        values = torch.ones(indices.shape[0])
        num_user_item = self.num_users + self.num_items  # sum(self.data.user_item.shape)
        temp_adj = sp.csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=(num_user_item, num_user_item))

        adj_mat = temp_adj + temp_adj.T

        # 归一化
        adj_mat += sp.eye(num_user_item)
        row_sum = np.array(adj_mat.sum(1))
        r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj_matrix = r_mat_inv.dot(adj_mat)

        # 转化为tensor sparse
        sparse_mx = adj_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.Tensor(sparse_mx.data)
        return torch.sparse.FloatTensor(indices, values, sparse_mx.shape)
