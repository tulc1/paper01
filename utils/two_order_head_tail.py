import random


def count_two_order_head_tail(data, top_bottom_items_list):

    print("top_bottom_items_list len:", len(top_bottom_items_list))

    ratio_1_items, ratio_2_items = top_bottom_items_list

    for i in range(10):
        user_id = (random.randint(0, data.num_users))

        item_list = data.user_item_two_order_pairs_set[user_id] - data.num_users  # 二阶可达物品，item序号
        # print("item_list shape:", item_list.shape)
        item_set = set(item_list)
        item_list_1 = data.user_item_pairs_set[user_id] - data.num_users  # user-item 中item序号
        item_set_1 = set(item_list_1)
        # print("item_set shape:", len(item_set), type(item_set))

        top_items, bottom_items = ratio_2_items  # ratio 0.2 序号
        # print("top_items len:", len(top_items), type(top_items))
        # print("bottom_items len:", len(bottom_items), type(bottom_items))

        user_top_items = item_set - bottom_items
        user_bottom_items = item_set - top_items

        user_top_items_1 = top_items - item_set_1
        user_bottom_items_1 = bottom_items - item_set_1

        user_top_items_2 = item_set_1 - bottom_items
        user_bottom_items_2 = item_set_1 - top_items

        print("=============top bottom===================")
        print("top:", len(user_top_items), "bottom:", len(user_bottom_items), "item_set:", len(item_set))
        print("two-order user item head hail ratio:", user_id, len(user_top_items) * 1.0 / len(item_set),
              len(user_bottom_items) * 1.0 / len(item_set))
        print()

        u_item_set = len(top_items) + len(bottom_items) - len(item_set_1)
        print("top:", len(user_top_items_1), "bottom:", len(user_bottom_items_1), "item_set:", u_item_set)
        print("un user-item head hail ratio:", user_id, len(user_top_items_1) * 1.0 / u_item_set,
              len(user_bottom_items_1) * 1.0 / u_item_set)
        print()

        print("top:", len(user_top_items_2), "bottom:", len(user_bottom_items_2), "item_set:", len(item_set_1))
        print("user-item head hail ratio:", user_id, len(user_top_items_2) * 1.0 / len(item_set_1),
              len(user_bottom_items_2) * 1.0 / len(item_set_1))
        print()
