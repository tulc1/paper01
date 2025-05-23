from multiprocessing import Process, Queue

import numpy as np
from scipy.sparse import lil_matrix
import itertools


def sample_function(data, adj_train, num_nodes, batch_size, n_negative, result_queue):
    num_users, num_items = num_nodes
    adj_train = lil_matrix(adj_train)
    all_pairs = np.asarray(adj_train.nonzero()).T
    user_item_pairs = all_pairs[: adj_train.count_nonzero() // 2]
    item_user_pairs = all_pairs[adj_train.count_nonzero() // 2:]

    assert len(user_item_pairs) == len(item_user_pairs)
    np.random.shuffle(user_item_pairs)
    np.random.shuffle(item_user_pairs)

    all_pairs_set = {idx: set(row) for idx, row in enumerate(adj_train.rows)}

    user_item_pairs_set = dict(itertools.islice(all_pairs_set.items(), num_users))

    while True:
        for i in range(int(len(user_item_pairs) / batch_size)):
            samples_for_users = batch_size

            user_positive_items_pairs = user_item_pairs[i * samples_for_users: (i + 1) * samples_for_users, :]
            user_negative_samples = np.random.randint(num_users, sum(num_nodes), size=(samples_for_users, n_negative))
            # user二阶item采样
            user_item_two_order_sample = np.zeros((samples_for_users, data.num_two_order)).astype(int)
            # item_item_sample = np.zeros((samples_for_users, data.item_item_num)).astype(int)

            for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                   user_negative_samples,
                                                   range(len(user_negative_samples))):
                user = user_positive[0]
                for j, neg in enumerate(negatives):
                    while neg in user_item_pairs_set[user]:
                        user_negative_samples[i, j] = neg = np.random.randint(num_users, sum(num_nodes))
                # 增加程序
                # user_item_two_order_sample[i, 0] = data.user_item_two_order_pairs_set[user][
                #     np.random.randint(0, len(data.user_item_two_order_pairs_set[user]))]
                # 增加程序
                for k in range(data.num_two_order):
                    user_item_two_order_sample[i, k] = data.user_item_two_order_pairs_set[user][
                        np.random.randint(0, len(data.user_item_two_order_pairs_set[user]))]
                # for k in range(data.item_item_num):
                #     item_item_sample[i, k] = data.item_item_pairs_set[user_positive[1] - num_users][
                #         np.random.randint(0, len(data.item_item_pairs_set[user_positive[1] - num_users]))]

            user_triples = np.hstack((user_positive_items_pairs, user_negative_samples,
                                      user_item_two_order_sample))

            np.random.shuffle(user_triples)
            result_queue.put(user_triples)


class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: user-positive-item pairs, negative-items
    of the shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, data, num_nodes, user_item_matrix, batch_size=10000, n_negative=10, n_workers=5):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(data, user_item_matrix,
                                                      num_nodes,
                                                      batch_size,
                                                      n_negative,
                                                      self.result_queue)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:  # type: Process
            p.terminate()
            p.join()
