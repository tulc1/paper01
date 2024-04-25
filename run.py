import csv
import time
import traceback
from datetime import datetime

import numpy as np
import torch

from config import parser
from eval_metrics import recall_at_k, recall_at_k_head_tail
from models.base_models import HRCFModel
from rgd.rsgd import RiemannianSGD
from utils.data_generator import Data
from utils.head_tail_split import output_format, split_items
from utils.helper import default_device, set_seed
from utils.log import Logger
from utils.sampler import WarpSampler
import itertools, heapq
from default_config.path import default_path


def train(model):
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(num_batches)

    # print("没有二阶采样")

    # === Train model
    for epoch in range(1, args.epochs + 1):
        avg_loss = 0.
        # === batch training
        t = time.time()
        # 构造对比图
        adj_train1 = model.create_adj_mat()
        adj_train2 = model.create_adj_mat()
        for batch in range(num_batches):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data.adj_train_norm)
            embeddings1 = model.encode(adj_train1)  # 对比图1
            embeddings2 = model.encode(adj_train2)  # 对比图2
            train_loss = model.compute_loss(embeddings, triples)
            train_loss_info_nce = model.info_nce(embeddings1, embeddings2, triples)  # 对比损失
            train_loss += train_loss_info_nce
            if torch.isnan(train_loss):  # exit if nan
                log.write('nan')
                import os
                os._exit(0)
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches

        # === evaluate at the end of each batch
        avg_loss = avg_loss.detach().cpu().numpy()
        if args.log:
            log.write('Train:{:3d} {:.2f}\n'.format(epoch, avg_loss))
        else:
            print(" ".join(['Epoch: {:04d}'.format(epoch),
                            '{:.3f}'.format(avg_loss),
                            'time: {:.4f}s'.format(time.time() - t)]), end=' ')
            print("")
            # ===================添加到文件中=====================================
            with open(default_path() + args.result, 'a+', encoding='utf-8') as f:
                f.write(" ".join(['Epoch: {:04d}'.format(epoch),
                            'avg_loss: {:.3f}'.format(avg_loss),
                            'time: {:.4f}s'.format(time.time() - t)]) + "\n")
            # ==================================================================

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start = time.time()
            embeddings = model.encode(data.adj_train_norm)
            print(time.time() - start)
            pred_matrix = model.predict(embeddings, data)
            print(time.time() - start)
            # results = eval_rec(pred_matrix, data)
            # if args.log:
            #     log.write('Test:{:3d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epoch + 1, results[0][1], results[0][2],
            #                                                                     results[-1][1], results[-1][2]))
            # else:
            #     print('\t'.join([str(round(x, 4)) for x in results[0]]))
            #     print('\t'.join([str(round(x, 4)) for x in results[-1]]))

            results_list = eval_rec(pred_matrix, data)
            recall, ndcg = results_list
            for ix in range(len(recall)):
                result_print = output_format(epoch, ix, recall, ndcg)
                # ===================添加到文件中=====================================
                with open(default_path() + args.result, 'a+', encoding='utf-8') as f:
                    f.write(result_print + "\n")
                # ==================================================================
                print(result_print)
    with open(default_path() + args.dataset + "-3-dim.csv", 'a+', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        embedding_numpy = embeddings.detach().cpu().numpy()
        for i in range(embedding_numpy.shape[0]):
            writer.writerow(embedding_numpy[i])

    sampler.close()


def argmax_top_k(a, top_k=50):
    topk_score_items = []
    for i in range(len(a)):
        topk_score_item = heapq.nlargest(top_k, zip(a[i], itertools.count()))
        topk_score_items.append([x[1] for x in topk_score_item])
    return topk_score_items


def ndcg_func_v1(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def ndcg_func(ground_truths, ranks, top_bottom_items):
    result = 0
    result_top = 0
    result_bottom = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        dcg_top = np.cumsum([1.0/np.log2(idx+2) if (item in ground_truth) and (item not in top_bottom_items[1]) else 0.0 for idx, item in enumerate(rank)])
        dcg_bottom = np.cumsum([1.0/np.log2(idx+2) if (item in ground_truth) and (item not in top_bottom_items[0]) else 0.0 for idx, item in enumerate(rank)])

        result += dcg / idcg
        result_top += dcg_top / idcg
        result_bottom += dcg_bottom / idcg

    return [result / len(ranks), result_top/len(ranks), result_bottom/len(ranks)]


def eval_rec_v1(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = np.NINF
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20, 50]:
        recall.append(recall_at_k(data.test_dict, pred_list, k))

    all_ndcg = ndcg_func([*data.test_dict.values()], pred_list)
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20, 50]]

    return recall, ndcg


def eval_rec(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = np.NINF
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    recall = []
    ndcg = []
    for top_bottom_items in top_bottom_items_list:
        recall_sub = []
        ndcg_sub = []
        for k in [5, 10, 20, 50]:
            recall_sub.append(recall_at_k_head_tail(data.test_dict, pred_list, k, top_bottom_items))
        recall.append(recall_sub)
        all_ndcg_list = ndcg_func([*data.test_dict.values()], pred_list, top_bottom_items)
        for x in [5, 10, 20, 50]:
            ndcg_sub.append([all_ndcg[x-1] for all_ndcg in all_ndcg_list])
        ndcg.append(ndcg_sub)
    return recall, ndcg


if __name__ == '__main__':
    args = parser.parse_args()

    print('3dim')
    print('==' * 20)
    print(args)
    print('==' * 20)
    # ===================添加到文件中=====================================
    with open(default_path() + args.result, 'a+', encoding='utf-8') as f:
        f.write("\n")
        f.write("==" * 20 + "\n")
        f.write(str(args) + "\n")
        f.write("==" * 20 + "\n")
    # ==================================================================

    if args.log:
        now = datetime.now()
        now = now.strftime('%m-%d_%H-%M-%S')
        log = Logger(args.log, now, args.dataset, args.i)

        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    else:
        print(args.embedding_dim, args.lr, args.weight_decay, args.margin, args.batch_size)
        print(args.scale, args.num_layers, args.network)

    # === fix seed
    set_seed(args.train_seed)

    # === prepare data
    data = Data(args.num_two_order, args.dataset, args.norm_adj, args.seed, args.test_ratio)
    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_users + data.num_items
    args.feat_dim = args.embedding_dim

    top_bottom_items_list = split_items(data)
    # === negative sampler (iterator)
    sampler = WarpSampler(data, (data.num_users, data.num_items), data.adj_train, args.batch_size, args.num_neg)

    model = HRCFModel((data.num_users, data.num_items), args, data)
    model = model.to(default_device())

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)

    try:
        train(model)
    except Exception:
        sampler.close()
        traceback.print_exc()
