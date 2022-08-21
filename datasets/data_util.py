#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:

import io
import itertools
import os
import os.path as osp
from collections import defaultdict, namedtuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg
import pickle as pkl


def batcher():
    def batcher_dev(batch):
        graphs = zip(*batch)
        graphs = [dgl.batch(g) for g in graphs]
        return graphs

    return batcher_dev

# def batcher():
#     def batcher_dev(batch):
#         graph_q, graph_k = zip(*batch)
#         graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
#         return graph_q, graph_k

#     return batcher_dev

def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev

def link_prediction_batcher():
    def batcher_dev(batch):
        graph_q, graph_k, label = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k, torch.LongTensor(label)
    
    return batcher_dev
        


Data = namedtuple("Data", ["x", "edge_index", "y"])

def create_graph_classification_dataset(dataset_name):
    name = {
        "imdb-binary": "IMDB-BINARY",
        "imdb-multi": "IMDB-MULTI",
        "rdt-b": "REDDIT-BINARY",
        "rdt-5k": "REDDIT-MULTI-5K",
        "collab": "COLLAB",
    }[dataset_name]
    dataset = TUDataset(name)
    dataset.num_labels = dataset.num_labels[0]
    dataset.graph_labels = dataset.graph_labels.squeeze()
    return dataset


class Edgelist(object):
    def __init__(self, root, name, node2id_path=None):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path, node2id_path)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path, node2id_path):
        node2id = dict()
        if node2id_path is not None:
            node2id = pkl.load(open(node2id_path, 'rb'))
        with open(edge_list_path) as f:
            edge_list = []
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    assert node2id_path is None, "something is wrong with node2id dict"
                    node2id[x] = len(node2id)
                if y not in node2id:
                    assert node2id_path is None, "something is wrong with node2id dict"
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id and label >= 0:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name or "h-index" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label] if label >= 0 else -1)
            if "hindex" in self.name or 'h-index' in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        # assert num_nodes == len(set(nodes))
        # 这里考虑了没有标签的结点，对应的one-hot label全为0，在相应做classification时会删除这部分数据
        y = torch.zeros(num_nodes, len(label2id))
        nodes, labels = np.array(nodes), np.array(labels)
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


class SSSingleDataset(object):
    def __init__(self, root, name):
        edge_index = self._preprocess(root, name)
        self.data = Data(x=None, edge_index=edge_index, y=None)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)

        return torch.LongTensor(edge_list).t()

class SSDataset(object):
    def __init__(self, root, name1, name2):
        edge_index_1, dict_1, self.node2id_1 = self._preprocess(root, name1)
        edge_index_2, dict_2, self.node2id_2 = self._preprocess(root, name2)
        self.data = [
            Data(x=None, edge_index=edge_index_1, y=dict_1),
            Data(x=None, edge_index=edge_index_2, y=dict_2),
        ]
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        dict_path = os.path.join(root, name + ".dict")
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        name_dict = dict()
        with open(dict_path) as f:
            for line in f:
                name, str_x = line.split("\t")
                x = int(str_x)
                if x not in node2id:
                    node2id[x] = len(node2id)
                name_dict[name] = node2id[x]

        num_nodes = len(node2id)

        return torch.LongTensor(edge_list).t(), name_dict, node2id

def create_node_classification_dataset(dataset_name):
    if "degree" in dataset_name:
        if "usa-airports" in dataset_name:
            return Edgelist(
                "data/degree_classification/usa-airports/",
                {
                    "usa-airports_mst_twin_domain1_degree": "usa-airports_mst_twin_domain1_degree",
                    "usa-airports_mst_twin_domain2_degree": "usa-airports_mst_twin_domain2_degree",
                }[dataset_name],
                "data/node2id_dicts/usa-airports_node2id.pkl"
            )
        elif 'h-index' in dataset_name:
            return Edgelist(
                "data/degree_classification/hindex",
                {
                    "h-index_mst_twin_domain1_degree": "h-index_mst_twin_domain1_degree",
                    "h-index_mst_twin_domain2_degree": "h-index_mst_twin_domain2_degree",
                }[dataset_name],
                "data/node2id_dicts/h-index_node2id.pkl"
            )
        elif "actor" in dataset_name:
            return Edgelist(
                "data/degree_classification/actor",
                {
                    "actor_mst_twin_domain1_degree": "actor_mst_twin_domain1_degree",
                    "actor_mst_twin_domain2_degree": "actor_mst_twin_domain2_degree",
                }[dataset_name],
                "data/node2id_dicts/actor_node2id.pkl",
            )
    
    if "usa-airports" in dataset_name:
        return Edgelist(
            "data/node_classification/usa-airports",
            {
                "usa-airports_mst_twin_domain1": "usa-airports_mst_twin_domain1",
                "usa-airports_mst_twin_domain2": "usa-airports_mst_twin_domain2",
            }[dataset_name],
            "data/node2id_dicts/usa-airports_node2id.pkl"
        )
    elif "actor" in dataset_name:
        return Edgelist(
            "data/node_classification/actor",
            {
                "actor_mst_twin_domain1": "actor_mst_twin_domain1",
                "actor_mst_twin_domain2": "actor_mst_twin_domain2",
            }[dataset_name],
            "data/node2id_dicts/actor_node2id.pkl"
        )
    elif "h-index" in dataset_name:
        return Edgelist(
            "data/node_classification/hindex/",
            {
                "h-index_twin_domain2": "hindex_twin_domain2",
                "h-index_mst_twin_domain1": "h-index_mst_twin_domain1",
                "h-index_mst_twin_domain2": "h-index_mst_twin_domain2",
            }[dataset_name],
            "data/node2id_dicts/h-index_node2id.pkl"
        )
    elif "DD242" in dataset_name:
        return Edgelist(
            "data/node_classification/DD242",
            {
                "DD242_mst_twin_domain2": "DD242_mst_twin_domain2",
                "DD242_mst_twin_domain1": "DD242_mst_twin_domain1",
            }[dataset_name],
            "data/node2id_dicts/DD242_node2id.pkl"
        )
    elif "chameleon" in dataset_name:
        return Edgelist(
            "data/node_classification/chameleon",
            {
                "chameleon_mst_twin_domain1": "chameleon_mst_twin_domain1",
                "chameleon_mst_twin_domain2": "chameleon_mst_twin_domain2",
            }[dataset_name],
            "data/node2id_dicts/chameleon_node2id.pkl"
        )
    elif dataset_name in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"]:
        return SSSingleDataset("data/panther/", dataset_name)
    else:
        raise NotImplementedError
    
class Linklist(object):
    def __init__(self, root, name, node2id_path=None):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        edge_label_path = os.path.join(root, name + ".edgelabel")
        
        edge_index, self.edges, self.edge_labels, self.node2id = self._preprocess(edge_list_path, edge_label_path, node2id_path)
        self.data = Data(x=None, edge_index=edge_index, y=None)
        self.transform = None
        
    def _preprocess(self, edge_list_path, edge_label_path, node2id_path):
        node2id = dict()
        if node2id_path is not None:
            node2id = pkl.load(open(node2id_path, 'rb'))
        with open(edge_list_path) as f:
            edge_list = []
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    assert node2id_path is None, "node2id dict is wrong"
                    node2id[x] = len(node2id)
                if y not in node2id:
                    assert node2id_path is None, "node2id dict is wrong"
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])
        edge_list = torch.tensor(edge_list, dtype=torch.int64).t()
        
        with open(edge_label_path) as f:
            edges = []
            labels = []
            for line in f:
                x, y, label = list(map(int, line.split()))
                src_node, target_node = node2id[x], node2id[y]
                assert label in (0, 1)
                edges.append([src_node, target_node])
                labels.append(label)
        edges = torch.tensor(edges, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return edge_list, edges, labels, node2id




       
def create_link_prediction_dataset(dataset_name):
    if "usa-airports" in dataset_name:
        return Linklist(
            "data/link_prediction/usa-airports",
            {
                "usa-airports_mst_twin_domain1": "usa-airports_mst_twin_domain1",
                "usa-airports_mst_twin_domain2": "usa-airports_mst_twin_domain2",
                "usa-airports_mst_twin_domain1_4_fold": "usa-airports_mst_twin_domain1_4_fold",
            }[dataset_name],
            "data/node2id_dicts/usa-airports_node2id.pkl"
        )
    elif "actor" in dataset_name:
        return Linklist(
            "data/link_prediction/actor",
            {
                "actor_mst_twin_domain1": "actor_mst_twin_domain1",
                "actor_mst_twin_domain2": "actor_mst_twin_domain2",
                "actor_mst_twin_domain1_4_fold": "actor_mst_twin_domain1_4_fold",
            }[dataset_name],
            "data/node2id_dicts/actor_node2id.pkl"
        )
    elif "h-index" in dataset_name:
        return Linklist(
            "data/link_prediction/hindex/",
            {
                "h-index_mst_twin_domain1": "h-index_mst_twin_domain1",
                "h-index_mst_twin_domain2": "h-index_mst_twin_domain2",
                "h-index_mst_twin_domain1_4_fold": "h-index_mst_twin_domain1_4_fold"
            }[dataset_name],
            "data/node2id_dicts/h-index_node2id.pkl"
        )
    elif 'chameleon' in dataset_name:
        return Linklist(
            "data/link_prediction/chameleon/",
            {
                "chameleon_mst_twin_domain1": "chameleon_mst_twin_domain1",
                "chameleon_mst_twin_domain2": "chameleon_mst_twin_domain2",
                "chameleon_mst_twin_domain1_4_fold": "chameleon_mst_twin_domain1_4_fold"
            }[dataset_name],
            {
                "chameleon_mst_twin_domain1": "data/node2id_dicts/chameleon_node2id.pkl",
                "chameleon_mst_twin_domain1_4_fold": "data/node2id_dicts/chameleon_node2id.pkl",
                "chameleon_mst_twin_domain2": "data/node2id_dicts/chameleon_node2id.pkl"
            }[dataset_name]
        )
    elif 'DD242' in dataset_name:
        return Linklist(
            "data/link_prediction/DD242/",
            {
                "DD242_mst_twin_domain1": "DD242_mst_twin_domain1",
                "DD242_mst_twin_domain2": "DD242_mst_twin_domain2",
                "DD242_mst_twin_domain1_4_fold": "DD242_mst_twin_domain1_4_fold"
            }[dataset_name],
            "data/node2id_dicts/DD242_node2id.pkl"
        )
    elif "squirrel" in dataset_name:
        return Linklist(
            "data/link_prediction/squirrel/",
            {
                "squirrel_mst_twin_domain1": "squirrel_mst_twin_domain1",
                "squirrel_mst_twin_domain2": "squirrel_mst_twin_domain2",
                "squirrel_mst_twin_domain1_4_fold": "squirrel_mst_twin_domain1_4_fold"
            }[dataset_name],
            "data/node2id_dicts/squirrel_node2id.pkl"
        )
    else:
        print(dataset_name)
        raise NotImplementedError           


def _rwr_trace_to_dgl_graph(
    g, seed, trace, positional_embedding_size, entire_graph=False, use_positional_embedding=True
):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)

    if use_positional_embedding:
        subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g
