from .utils import *
import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
from dgl.nn.pytorch.conv import SAGEConv
import torch


import dgl
import torch as th
import networkx as nx
import numpy as np
import time
from tqdm import tqdm


from networkx.generators.random_graphs import dense_gnm_random_graph
from networkx.generators.classic import barbell_graph
from networkx.generators.community import connected_caveman_graph
from networkx.generators.community import planted_partition_graph
from networkx.algorithms.community.asyn_fluid import asyn_fluidc

import torch as th
th.manual_seed(0)


"""
surrogate model (graphsage)
"""


class SAGEEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))

        self.layers.append(dglnn.SAGEConv(n_hidden, n_output_dim, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x

        for i in range(0, self.n_layers):
            h = self.layers[i](blocks[i], h)
            h = self.activation(h)
            if i != self.n_layers - 1:
                h = self.dropout(h)

        return h

    def inference(self, g, x, batch_size, device):
        for l, layer in enumerate(self.layers):

            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_output_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)

                h = self.activation(h)
                if l != self.n_layers - 1:
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y

        return y.to(device)


def evaluate_sage_surrogate(model, clf, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    # move labels to gpu
    labels = labels.to(device)
    model.eval()
    clf.eval()
    with th.no_grad():
        embs = model.inference(g, inputs, batch_size, device)
        logists = clf(embs)

    model.train()
    clf.train()
    return compute_acc(logists, labels), logists, embs


def run_sage_surrogate(args, device, data, model_filename):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g, target_response = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = target_response.shape[1]

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model_surrogate = SAGEEMB(in_feats, args.num_hidden, n_output_dim, n_classes,
                              args.num_layers, F.relu, args.batch_size, args.num_workers, args.dropout)
    model_surrogate = model_surrogate.to(device)
    loss_fcn = nn.MSELoss()
    loss_fcn = loss_fcn.to(device)

    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    optimizer = optim.Adam(model_surrogate.parameters(), lr=args.lr)

    clf = Classification(n_output_dim, n_classes)
    clf = clf.to(device)
    optimizer_classification = optim.SGD(clf.parameters(),
                                         lr=0.01)

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            batch_output_nid = blocks[-1].dstdata['_ID']

            # Compute loss and prediction
            embs = model_surrogate(blocks, batch_inputs)
            loss = torch.sqrt(
                loss_fcn(embs, target_response[batch_output_nid]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_classification.zero_grad()
            logists = clf(embs.detach())
            loss_sup = loss_clf(logists, batch_labels)
            loss_sup.backward()
            optimizer_classification.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_sage_surrogate(
                model_surrogate, clf, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
#             if eval_acc > best_val_score:
#                 print("save best surrogate model to {}".format(model_filename))
#                 torch.save(model_surrogate, model_filename)
#                 best_val_score = eval_acc
            test_acc, test_preds, test_embs = evaluate_sage_surrogate(
                model_surrogate, clf, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.batch_size, device)
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    eval_acc, eval_preds, eval_embs = evaluate_sage_surrogate(
        model_surrogate, clf, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, args.batch_size, device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    return model_surrogate, clf, detached_classifier
