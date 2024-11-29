import random


def count_two_order_head_tail(data):

    all_ratio = 0
    number = 0
    for key, values in data.test_dict.items():
        count = 0
        number += 1
        for v in values:
            item_list = data.user_item_two_order_pairs_set[key] - data.num_users  # 二阶可达物品，item序号
            item_set = set(item_list)
            if v in item_set:
                count += 1
        # ratio = count * 1. / len(values)
        all_ratio += count * 1. / len(values)
    all_ratio /= number
    print(all_ratio)

