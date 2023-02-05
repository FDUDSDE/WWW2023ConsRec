"""Helper functions for loading dataset"""
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import torch


def load_rating_file_to_list(filename):
    """Return **List** format user/group-item interactions"""
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        # Each line: user item
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list


def load_rating_file_to_matrix(filename, num_users=None, num_items=None):
    """Return **Matrix** format user/group-item interactions"""
    if num_users is None:
        num_users, num_items = 0, 0

    lines = open(filename, 'r').readlines()
    for line in lines:
        contents = line.split()
        u, i = int(contents[0]), int(contents[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    for line in lines:
        contents = line.split()
        if len(contents) > 2:
            u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
            if rating > 0:
                mat[u, i] = 1.0
        else:
            u, i = int(contents[0]), int(contents[1])
            mat[u, i] = 1.0
    return mat


def load_negative_file(filename):
    """Return **List** format negative files"""
    negative_list = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        negatives = line.split()[1:]
        negatives = [int(neg_item) for neg_item in negatives]
        negative_list.append(negatives)
    return negative_list


def load_group_member_to_dict(user_in_group_path):
    """Return **Dict** format group-to-member-list mapping"""
    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(','):
            group_member_dict[group].append(int(member))
    return group_member_dict


def build_group_graph(group_data, num_groups):
    """Return group-level graph (**a weighted graph** with weights defined as ratio of common members and items)"""
    matrix = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group_a = set(group_data[i])
        for j in range(i + 1, num_groups):
            group_b = set(group_data[j])
            overlap = group_a & group_b
            union = group_a | group_b
            # weight computation
            matrix[i][j] = float(len(overlap) / len(union))
            matrix[j][i] = matrix[i][j]

    matrix = matrix + np.diag([1.0] * num_groups)
    degree = np.sum(np.array(matrix), 1)
    # \mathbf{D}^{-1} \dot \mathbf{A}
    return np.dot(np.diag(1.0 / degree), matrix)


def build_hyper_graph(group_member_dict, group_train_path, num_users, num_items, num_groups, group_item_dict=None):
    """Return member-level hyper-graph"""
    # Construct group-to-item-list mapping
    if group_item_dict is None:
        group_item_dict = defaultdict(list)

        for line in open(group_train_path, 'r').readlines():
            contents = line.split()
            if len(contents) > 2:
                group, item, rating = int(contents[0]), int(contents[1]), int(contents[2])
                if rating > 0:
                    group_item_dict[group].append(item)
            else:
                group, item = int(contents[0]), int(contents[1])
                group_item_dict[group].append(item)

    def _prepare(group_dict, rows, axis=0):
        nodes, groups = [], []

        for group_id in range(num_groups):
            groups.extend([group_id] * len(group_dict[group_id]))
            nodes.extend(group_dict[group_id])

        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, groups)), shape=(rows, num_groups))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    # Two separate hypergraphs (user_hypergraph, item_hypergraph for hypergraph convolution computation)
    user_hg, user_hg_deg = _prepare(group_member_dict, num_users)
    item_hg, item_hg_deg = _prepare(group_item_dict, num_items)

    for group_id, items in group_item_dict.items():
        group_item_dict[group_id] = [item + num_users for item in items]
    group_data = [group_member_dict[group_id] + group_item_dict[group_id] for group_id in range(num_groups)]
    full_hg, hg_dg = _prepare(group_data, num_users + num_items, axis=1)

    user_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(user_hg_deg),
                                       convert_sp_mat_to_sp_tensor(user_hg).t())
    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())
    full_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(hg_dg), convert_sp_mat_to_sp_tensor(full_hg))
    print(
        f"User hyper-graph {user_hyper_graph.shape}, Item hyper-graph {item_hyper_graph.shape}, Full hyper-graph {full_hyper_graph.shape}")

    return user_hyper_graph, item_hyper_graph, full_hyper_graph, group_data


def convert_sp_mat_to_sp_tensor(x):
    """Convert `csr_matrix` into `torch.SparseTensor` format"""
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def build_light_gcn_graph(group_item_net, num_groups, num_items):
    """Return item-level graph (**a group-item bipartite graph**)"""
    adj_mat = sp.dok_matrix((num_groups + num_items, num_groups + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = group_item_net.tolil()
    adj_mat[:num_groups, num_groups:] = R
    adj_mat[num_groups:, :num_groups] = R.T
    adj_mat = adj_mat.todok()
    # print(adj_mat, adj_mat.shape)

    row_sum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # print(d_mat)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    graph = convert_sp_mat_to_sp_tensor(norm_adj)
    return graph.coalesce()


# Test code
# if __name__ == "__main__":
#     g_m_d = {0: [0, 1, 2], 1: [2, 3], 2: [4, 5, 6]}
#     g_i_d = {0: [0, 1], 1: [1, 2], 2: [3]}
#     user_g, item_g, hg, g_data = build_hyper_graph(g_m_d, "", 7, 4, 3, g_i_d)
#
#     print(user_g)
#     print(item_g)
#     print(hg)
#     print()
#     g = build_group_graph(g_data, 3)
#     print(g)
