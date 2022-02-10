import dgl


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from networkx.generators.random_graphs import dense_gnm_random_graph, erdos_renyi_graph
from networkx.generators.classic import barbell_graph
from networkx.generators.community import connected_caveman_graph
from networkx.generators.community import planted_partition_graph
from networkx.algorithms.community.asyn_fluid import asyn_fluidc

import numpy as np


import os
import sys
import pickle as pkl
import scipy.sparse as sp


import networkx as nx
from graphgallery.datasets import NPZDataset
import random

from torch import nn
import torch

from networkx.generators.random_graphs import dense_gnm_random_graph

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class Classification(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = nn.Linear(emb_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def load_graphgallery_data(dataset):
    # set `verbose=False` to avoid additional outputs
    data = NPZDataset(dataset, verbose=False)
    graph = data.graph
    nx_g = nx.from_scipy_sparse_matrix(graph.adj_matrix)

    for node_id, node_data in nx_g.nodes(data=True):
        node_data["features"] = graph.node_attr[node_id].astype(np.float32)
        if dataset in ['blogcatalog', 'flickr']:
            node_data["labels"] = graph.node_label[node_id].astype(np.long) - 1
        else:
            node_data["labels"] = graph.node_label[node_id].astype(np.long)

    dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    return dgl_graph, len(np.unique(graph.node_label))

# def load_data(dataset_name):
#     """
#     https://docs.dgl.ai/api/python/dgl.data.html#node-prediction-datasets

#     We can select dataset from:
#         Citation datasets: Cora, Citeseer, Pubmed
#     """
#     if dataset_name == "Cora":
#         data = dgl.data.CoraGraphDataset()
#     elif dataset_name == "Citeseer":
#         data = dgl.data.CiteseerGraphDataset()
#     elif dataset_name == "Pubmed":
#         data = dgl.data.PubmedGraphDataset()
#     elif dataset_name == "Amazon":
#         data = dgl.data.AmazonCoBuyComputerDataset()
#     elif dataset_name == 'Photo': # not working
#         data = dgl.data.AmazonCoBuyPhotoDataset(force_reload=True)
#     elif dataset_name == 'Physics':
#         data = dgl.data.CoauthorPhysicsDataset()
#     elif dataset_name == "Coauthor":
#         data = dgl.data.CoauthorCSDataset()
#     elif dataset_name == "Reddit": # not working
#         data = dgl.data.RedditDataset()
#     elif dataset_name == 'AIFB': # not working
#         data = dgl.data.AIFBDataset()
#     elif dataset_name == 'CoraFull':
#         data = dgl.data.CoraFullDataset()

#     g = data[0]
#     g = dgl.transform.add_self_loop(g)

#     g.ndata['features'] = g.ndata['feat']
#     g.ndata['labels'] = g.ndata['label']
#     if dataset_name in ['Photo', 'Amazon']: # workaround of incorrect value returned by DGL
#         return g, len(np.unique(g.ndata['labels']))
#     return g, data.num_classes


def get_dataset_feature_label(dataset_name):
    """
    generate feature and label for each dataset
    """
    if dataset_name == "Cora":
        data = dgl.data.CoraGraphDataset()
    elif dataset_name == "Citeseer":
        data = dgl.data.CiteseerGraphDataset()
    elif dataset_name == "Pubmed":
        data = dgl.data.PubmedGraphDataset()
    elif dataset_name == "Amazon":
        data = dgl.data.AmazonCoBuyComputerDataset()
    elif dataset_name == 'Photo':  # not working
        data = dgl.data.AmazonCoBuyPhotoDataset(force_reload=True)
    elif dataset_name == 'Physics':
        data = dgl.data.CoauthorPhysicsDataset()
    elif dataset_name == "Coauthor":
        data = dgl.data.CoauthorCSDataset()
    elif dataset_name == "Reddit":  # not working
        data = dgl.data.RedditDataset()
    elif dataset_name == 'AIFB':  # not working
        data = dgl.data.AIFBDataset()
    elif dataset_name == 'CoraFull':
        data = dgl.data.CoraFullDataset()

    g = data[0]
    g = dgl.transform.add_self_loop(g)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    feature = g.ndata['feat'].numpy()
    label = g.ndata['label'].numpy()
    return feature, label


def compute_fidelity(pred_surrogate, pred_target):
    _surrogate = th.argmax(pred_surrogate, dim=1)
    _target = th.argmax(pred_target, dim=1)
    _fidelity = (_surrogate == _target).float().sum() / len(pred_surrogate)
    return _fidelity.clone().detach().cpu().item()


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g


def projection(X, y, transform_name='TSNE', show_figure=True, gnn='Graphsage', dataset='Cora'):

    if transform_name == 'TSNE':
        transform = TSNE
        trans = transform(n_components=2, n_iter=3000, n_jobs=-1)
        emb_transformed = pd.DataFrame(trans.fit_transform(X))
    elif transform_name == 'PCA':
        transform = PCA
        trans = transform(n_components=2)
        emb_transformed = pd.DataFrame(trans.fit_transform(X))

    if show_figure:
        emb_transformed["label"] = y
        alpha = 0.7

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            emb_transformed[0],
            emb_transformed[1],
            c=emb_transformed["label"].astype("category"),
            cmap="jet",
            alpha=alpha,
        )
        ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
        plt.axis('off')
        plt.title(
            "{} visualization of {} embeddings for {} dataset".format(transform.__name__,
                                                                      gnn,
                                                                      dataset)
        )
        plt.show()

    return emb_transformed.iloc[:, [0, 1]]


# def generate_random_graph(nodes_per_class, num_classes, num_feats):

#     g_query = planted_partition_graph(num_classes, nodes_per_class, 0.5, 0.02)
#     node_features_query = np.random.choice([0.0, 1.0], size=(len(g_query), num_feats), p=[9./10, 1./10])

#     node_labels_query = []
#     for _class in range(0, num_classes):
#         node_labels_query.extend([_class]*nodes_per_class)

#     for node_id, node_data in g_query.nodes(data=True):
#         node_data["features"] = node_features_query[node_id].astype(np.float32)
#         node_data["labels"] = node_labels_query[node_id]

#     G_QUERY = dgl.from_networkx(g_query, node_attrs=['features', 'labels'])
#     return G_QUERY


def generate_random_graph(n, m, num_classes, num_feats):
    #g_query = dense_gnm_random_graph(n, m)
    g_query = erdos_renyi_graph(n, 0.01)
    node_features_query = np.random.choice(
        [0, 1], size=(len(g_query), num_feats), p=[9.5/10, 0.5/10])

    labels = list(range(0, num_classes))
    for node_id, node_data in g_query.nodes(data=True):
        node_data["features"] = node_features_query[node_id].astype(np.float32)
        node_data["labels"] = random.choice(labels)

    G_QUERY = dgl.from_networkx(g_query, node_attrs=['features', 'labels'])
    G_QUERY = dgl.add_self_loop(G_QUERY)
    return G_QUERY


def load_planetoid(dataset, data_path, _log):
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    if _log is not None:
        _log.info('Loading dataset %s.' % dataset)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(data_path, "ind.{}.test.index".format(dataset))
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # cast!!!
    adj = adj.astype(np.int)
    features = features.tocsr()
    features = features.astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    g = nx.from_scipy_sparse_matrix(adj)

    return g, adj, features.toarray(), labels, idx_train, idx_val, idx_test


def split_graph_into_communities(g, n_community):

    GCCs = sorted(nx.connected_components(g), key=len, reverse=True)
    g_GCC = g.subgraph(GCCs[0])
    communities = asyn_fluidc(g_GCC, n_community)

    splitted_graphs = []
    for community in communities:
        splitted_graphs.append(g_GCC.subgraph(community))

    splitted_graphs.sort(key=len, reverse=True)

    return splitted_graphs


def generate_DGL_graph(g, features, labels):

    mapping = dict([(nid, i) for i, nid in enumerate(g.nodes())])

    for node_id, node_data in g.nodes(data=True):
        node_data["features"] = features[node_id]
        node_data["labels"] = np.argmax(labels[node_id])

    relabel_g = nx.relabel_nodes(g, mapping)

    return relabel_g


def split_graph(g, frac_list=[0.6, 0.2, 0.2]):
    train_subset, val_subset, test_subset = dgl.data.utils.split_dataset(
        g, frac_list=frac_list, shuffle=True)
    train_g = g.subgraph(train_subset.indices)
    val_g = g.subgraph(val_subset.indices)
    test_g = g.subgraph(test_subset.indices)

    if not 'features' in train_g.ndata:
        train_g.ndata['features'] = train_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        train_g.ndata['labels'] = train_g.ndata['label']

    if not 'features' in val_g.ndata:
        val_g.ndata['features'] = val_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        val_g.ndata['labels'] = val_g.ndata['label']

    if not 'features' in test_g.ndata:
        test_g.ndata['features'] = test_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        test_g.ndata['labels'] = test_g.ndata['label']
    return train_g, val_g, test_g


def split_graph_different_ratio(g, frac_list=[0.6, 0.2, 0.2], ratio=0.5):
    print(g)
    train_subset, val_subset, test_subset = dgl.data.utils.split_dataset(
        g, frac_list=frac_list, shuffle=True)
    train_index = train_subset.indices[:int(len(train_subset.indices) * ratio)]
    train_g = g.subgraph(train_index)
    val_g = g.subgraph(val_subset.indices)
    test_g = g.subgraph(test_subset.indices)

    if not 'features' in train_g.ndata:
        train_g.ndata['features'] = train_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        train_g.ndata['labels'] = train_g.ndata['label']

    if not 'features' in val_g.ndata:
        val_g.ndata['features'] = val_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        val_g.ndata['labels'] = val_g.ndata['label']

    if not 'features' in test_g.ndata:
        test_g.ndata['features'] = test_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        test_g.ndata['labels'] = test_g.ndata['label']
    return train_g, val_g, test_g


def delete_dgl_graph_edge(train_g):
    temp_g = nx.Graph()
    for nid in range(train_g.number_of_nodes()):
        temp_g.add_node(nid)
    dgl_g = dgl.from_networkx(temp_g)
    dgl_g = dgl.add_self_loop(dgl_g)
    dgl_g.ndata['features'] = train_g.ndata['features']
    dgl_g.ndata['labels'] = train_g.ndata['labels']
    return dgl_g


def train_detached_classifier(test_g, embds_surrogate):

    X, y = make_classification(n_samples=100, random_state=1)

    X = embds_surrogate.clone().detach().cpu().numpy()
    y = test_g.ndata['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=1)

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])

    clf.predict(X_test[:5, :])

    print(clf.score(X_test, y_test))
    return clf
