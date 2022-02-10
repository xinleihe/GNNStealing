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
import traceback

from .utils import compute_acc


class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        linear = nn.Linear(in_feats, n_hidden)
        self.layers.append(dglnn.GINConv(linear, 'sum'))
        for i in range(1, n_layers-1):
            linear = nn.Linear(n_hidden, n_hidden)
            self.layers.append(dglnn.GINConv(linear, 'sum'))
        linear = nn.Linear(n_hidden, n_classes)
        self.layers.append(dglnn.GINConv(linear, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_classes)
            if l == len(self.layers) - 2:
                embs = th.zeros(g.number_of_nodes(), self.n_hidden)
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
                if l < len(self.layers) - 2:
                    h = self.activation(h)
                    h = self.dropout(h)
                elif l == len(self.layers) - 2:
                    emb = self.activation(h)
                    embs[output_nodes] = emb.cpu()
                    h = self.dropout(emb)

                y[output_nodes] = h.cpu()

            x = y
        return y, embs


# def compute_acc(pred, labels):
#     """
#     Compute the accuracy of prediction given the labels.
#     """
#     return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate_gin_target(model,
                        g,
                        inputs,
                        labels,
                        val_nid,
                        batch_size,
                        device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata['features']
        pred, embds = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), pred, embds


def run_gin_target(args, device, data):
    # Unpack data
    train_g, val_g, test_g, in_feats, labels, n_classes = data

    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

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
    model = GIN(in_feats,
                args.num_hidden,
                n_classes,
                args.num_layers,
                F.relu,
                args.batch_size,
                args.num_workers,
                args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.to(device) for blk in blocks]

            # Load the input features as well as output labels
            # batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, pred, embds = evaluate_gin_target(
                model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, args.val_batch_size, device)
            test_acc, pred, embds = evaluate_gin_target(
                model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.val_batch_size, device)
#             if args.save_pred:
#                 np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            print('Eval Acc {:.4f}'.format(eval_acc))
            print('Test Acc {:.4f}'.format(test_acc))
#             if eval_acc > best_eval_acc:
#                 best_eval_acc = eval_acc
#                 best_test_acc = test_acc
#             print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    return model
