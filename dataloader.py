from datautil import load_rating_file_to_matrix, load_rating_file_to_list, load_negative_file, \
    load_group_member_to_dict, build_hyper_graph, build_group_graph, build_light_gcn_graph
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class GroupDataset(object):
    def __init__(self, user_path, group_path, num_negatives, dataset="Mafengwo"):
        print(f"[{dataset.upper()}] loading...")
        self.num_negatives = num_negatives

        # User data
        if dataset == "MafengwoS":
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt", num_users=11026, num_items=1235)
        else:
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.user_test_negatives = load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} "
              f"interactions, sparsity: {(1-len(self.user_train_matrix.keys()) / self.num_users / self.num_items):.5f}")

        # Group data
        self.group_train_matrix = load_rating_file_to_matrix(group_path + "Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_test_negatives = load_negative_file(group_path + "Negative.txt")
        self.num_groups, self.num_group_net_items = self.group_train_matrix.shape
        self.group_member_dict = load_group_member_to_dict(f"./data/{dataset}/groupMember.txt")

        print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, spa"
              f"rsity: {(1-len(self.group_train_matrix.keys()) / self.num_groups / self.group_train_matrix.shape[1]):.5f}")

        # Member-level Hyper-graph
        self.user_hyper_graph, self.item_hyper_graph, self.full_hg, group_data = build_hyper_graph(
            self.group_member_dict, group_path + "Train.txt", self.num_users, self.num_items, self.num_groups)
        # Group-level graph
        self.overlap_graph = build_group_graph(group_data, self.num_groups)
        # Item-level graph
        self.light_gcn_graph = build_light_gcn_graph(self.group_train_matrix, self.num_groups, self.num_group_net_items)
        print(f"\033[0;30;43m{dataset.upper()} finish loading!\033[0m", end='')

    def get_train_instances(self, train):
        """Generate train samples (user, pos_item, neg_itm)"""
        users, pos_items, neg_items = [], [], []

        num_users, num_items = train.shape[0], train.shape[1]

        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                users.append(u)
                pos_items.append(i)

                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                neg_items.append(j)
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    def get_user_dataloader(self, batch_size):
        users, pos_neg_items = self.get_train_instances(self.user_train_matrix)
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_dataloader(self, batch_size):
        groups, pos_neg_items = self.get_train_instances(self.group_train_matrix)
        train_data = TensorDataset(torch.LongTensor(groups), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)
