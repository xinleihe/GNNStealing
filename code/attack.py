from src.utils import delete_dgl_graph_edge
from src.gin import *
from src.gat import *
from src.sage import *
from src.sagesurrogate import *
from src.ginsurrogate import *
from src.gatsurrogate import *
from src.utils import *
from src.constants import *
from scipy import sparse
import os
import pickle
import argparse
import torch
import numpy as np
from core.model_handler import ModelHandler

torch.set_num_threads(1)
torch.manual_seed(0)


def idgl(config):
    model = ModelHandler(config)
    model.train()
    test_metrics, adj = model.test()
    return adj


DATASETS = ['citeseer_full']
RESPONSES = ['projection', 'prediction', 'embedding']


argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=int, default=1,
                       help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--dataset', type=str, default='citeseer_full')
argparser.add_argument('--num-epochs', type=int, default=200)
argparser.add_argument('--transform', type=str, default='TSNE')
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--fan-out', type=str, default='10,50')
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--log-every', type=int, default=20)
argparser.add_argument('--eval-every', type=int, default=50)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=4,
                       help="Number of sampling processes. Use 0 for no extra process.")
argparser.add_argument('--inductive', action='store_true',
                       help="Inductive learning setting")
argparser.add_argument('--head', type=int, default=4)
argparser.add_argument('--wd', type=float, default=0)
argparser.add_argument('--target-model', type=str, default='sage')
argparser.add_argument('--target-model-dim', type=int, default=256)
argparser.add_argument('--surrogate-model', type=str, default='sage')
argparser.add_argument('--num-hidden', type=int, default=256)
argparser.add_argument('--recovery-from', type=str, default='embedding')
argparser.add_argument('--round_index', type=int, default=1)
argparser.add_argument('--query_ratio', type=float, default=1.0,
                       help="1.0 means we use 30% target dataset as the G_QUERY. 0.5 means 15%...")
argparser.add_argument('--structure', type=str, default='original')
argparser.add_argument('--delete_edges', type=str, default='no')

args, _ = argparser.parse_known_args()

args.inductive = True


surrogate_projection_accuracy = []
surrogate_prediction_accuracy = []
surrogate_embedding_accuracy = []

surrogate_projection_fidelity = []
surrogate_prediction_fidelity = []
surrogate_embedding_fidelity = []

target_accuracy = []


if args.gpu >= 0:
    device = th.device('cuda:%d' % args.gpu)
else:
    device = th.device('cpu')


g, n_classes = load_graphgallery_data(args.dataset)


model_args = pickle.load(open('./target_model_' + args.target_model +
                         '_' + str(args.target_model_dim) + '/model_args', 'rb'))
model_args = model_args.__dict__
model_args["gpu"] = args.gpu

if args.target_model == 'gin':
    print('target model: ', args.target_model)
    target_model = GIN(g.ndata['features'].shape[1],
                       model_args['num_hidden'],
                       n_classes,
                       model_args['num_layers'],
                       F.relu,
                       model_args['batch_size'],
                       model_args['num_workers'],
                       model_args['dropout'])
    target_model.load_state_dict(torch.load('./target_model_' + args.target_model + '_' + str(args.target_model_dim) + '/' + './target_model_gin_' + args.dataset,
                                            map_location=torch.device('cpu')))
else:
    target_model = torch.load('./target_model_' + args.target_model + '_' + str(args.target_model_dim) + '/' + './target_model_' + args.target_model + '_' + args.dataset,
                              map_location=torch.device('cpu'))
target_model = target_model.to(device)


train_g, val_g, test_g = split_graph_different_ratio(g, frac_list=[0.3, 0.2, 0.5], ratio=args.query_ratio)

if args.structure == 'original':
    G_QUERY = train_g
    # only use node to query
    if args.delete_edges == "yes":
        G_QUERY = delete_dgl_graph_edge(train_g)

elif args.structure == 'idgl':
    config['dgl_graph'] = train_g
    config['cuda_id'] = args.gpu
    adj = idgl(config)
    adj = adj.clone().detach().cpu().numpy()
    if args.dataset in ['acm', 'amazon_cs']:
        adj = (adj > 0.9).astype(np.int)
    elif args.dataset in ['coauthor_phy']:
        adj = (adj >= 0.999).astype(np.int)
    else:
        adj = (adj > 0.999).astype(np.int)

    sparse_adj = sparse_csr_mat = sparse.csr_matrix(adj)
    G_QUERY = dgl.from_scipy(sparse_adj)
    G_QUERY.ndata['features'] = train_g.ndata['features']
    G_QUERY.ndata['labels'] = train_g.ndata['labels']
    G_QUERY = dgl.add_self_loop(G_QUERY)

else:
    print("wrong structure param... stop!")
    sys.exit()

print(train_g.number_of_nodes(), val_g.number_of_nodes(), test_g.number_of_nodes())


# query target model with G_QUERY
if args.target_model == 'sage':
    query_acc, query_preds, query_embs = evaluate_sage_target(target_model,
                                                              G_QUERY,
                                                              G_QUERY.ndata['features'],
                                                              G_QUERY.ndata['labels'],
                                                              G_QUERY.nodes(),
                                                              args.batch_size,
                                                              device)
elif args.target_model == 'gin':
    query_acc, query_preds, query_embs = evaluate_gin_target(target_model,
                                                             G_QUERY,
                                                             G_QUERY.ndata['features'],
                                                             G_QUERY.ndata['labels'],
                                                             G_QUERY.nodes(),
                                                             args.batch_size,
                                                             device)
elif args.target_model == 'gat':
    query_acc, query_preds, query_embs = evaluate_gat_target(target_model,
                                                             G_QUERY,
                                                             G_QUERY.ndata['features'],
                                                             G_QUERY.ndata['labels'],
                                                             G_QUERY.nodes(),
                                                             model_args['val_batch_size'],
                                                             model_args['head'],
                                                             device)


query_embs = query_embs.to(device)
query_preds = query_preds.to(device)


if args.structure != 'original':
    print("using idgl reconstructed graph")
    train_g = G_QUERY


train_g.create_formats_()
val_g.create_formats_()
test_g.create_formats_()

surrogate_model_filename = './surrogate_model'


# preprocess query response
if args.recovery_from == 'prediction':
    print(args.dataset, args.recovery_from,
          'round {}'.format(str(args.round_index)))
    data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_preds
elif args.recovery_from == 'embedding':
    print(args.dataset, args.recovery_from,
          'round {}'.format(str(args.round_index)))
    data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_embs
elif args.recovery_from == 'projection':
    print(args.dataset, args.recovery_from,
          'round {}'.format(str(args.round_index)))
    tsne_embs = projection(query_embs.clone().detach().cpu().numpy(
    ), G_QUERY.ndata['labels'], transform_name=args.transform, gnn=args.target_model, dataset=args.dataset)
    tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
    data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, tsne_embs
else:
    print("wrong recovery-from value")
    sys.exit()

# which surrogate model to build
if args.surrogate_model == 'gin':
    print('surrogate model: ', args.surrogate_model)
    model_s, classifier, detached_classifier = run_gin_surrogate(
        args, device, data, surrogate_model_filename)
    acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s,
                                                                             classifier,
                                                                             test_g, test_g.ndata['features'],
                                                                             test_g.ndata['labels'],
                                                                             test_g.nodes(),
                                                                             args.batch_size, device)


elif args.surrogate_model == 'gat':
    print('surrogate model: ', args.surrogate_model)
    model_s, classifier, detached_classifier = run_gat_surrogate(
        args, device, data, surrogate_model_filename)
    acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s,
                                                                             classifier,
                                                                             test_g,
                                                                             test_g.ndata['features'],
                                                                             test_g.ndata['labels'],
                                                                             test_g.nodes(),
                                                                             args.batch_size,
                                                                             args.head,
                                                                             device)


elif args.surrogate_model == 'sage':
    print('surrogate model: ', args.surrogate_model)
    model_s, classifier, detached_classifier = run_sage_surrogate(
        args, device, data, surrogate_model_filename)
    acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s,
                                                                              classifier,
                                                                              test_g,
                                                                              test_g.ndata['features'],
                                                                              test_g.ndata['labels'],
                                                                              test_g.nodes(),
                                                                              args.batch_size,
                                                                              device)
else:
    print("wrong recovery-from value")
    sys.exit()

_acc = detached_classifier.score(embds_surrogate.clone().detach().cpu().numpy(),
                                 test_g.ndata['labels'])
_predicts = detached_classifier.predict_proba(
    embds_surrogate.clone().detach().cpu().numpy())


if args.target_model == 'sage':
    test_acc, pred, embs = evaluate_sage_target(target_model,
                                                test_g,
                                                test_g.ndata['features'],
                                                test_g.ndata['labels'],
                                                test_g.nodes(),
                                                args.batch_size,
                                                device)
elif args.target_model == 'gat':
    test_acc, pred, embs = evaluate_gat_target(target_model,
                                               test_g,
                                               test_g.ndata['features'],
                                               test_g.ndata['labels'],
                                               test_g.nodes(),
                                               model_args['val_batch_size'],
                                               model_args['head'],
                                               device)

elif args.target_model == 'gin':
    test_acc, pred, embs = evaluate_gin_target(target_model,
                                               test_g,
                                               test_g.ndata['features'],
                                               test_g.ndata['labels'],
                                               test_g.nodes(),
                                               args.batch_size,
                                               device)

target_accuracy.append(test_acc)

_fidelity = compute_fidelity(torch.from_numpy(
    _predicts).to(device), pred.to(device))

# which output to save
if args.recovery_from == 'prediction':
    surrogate_prediction_fidelity.append(_fidelity)
    surrogate_prediction_accuracy.append(_acc)
elif args.recovery_from == 'embedding':
    surrogate_embedding_fidelity.append(_fidelity)
    surrogate_embedding_accuracy.append(_acc)
elif args.recovery_from == 'projection':
    surrogate_projection_fidelity.append(_fidelity)
    surrogate_projection_accuracy.append(_acc)
else:
    print("wrong recovery-from value")
    sys.exit()


OUTPUT_FOLDER = './results_acc_fidelity/results_%s_%d_%s_%d' % (
    args.target_model, args.target_model_dim, args.surrogate_model, args.num_hidden)
if not os.path.isdir(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if args.structure == 'original':
    _filename = OUTPUT_FOLDER + '/' + args.dataset + '_original.txt'
elif args.structure == 'idgl':
    _filename = OUTPUT_FOLDER + '/' + args.dataset + '_idgl.txt'
else:
    print("wrong structure params... stop!")
    sys.exit()

with open(_filename, 'a') as wf:
    wf.write("%s,%d,%s,%d,%s,%d,%f,%f,%f,%f\n" % (args.target_model,
                                                  args.target_model_dim,
                                                  args.surrogate_model,
                                                  args.num_hidden,
                                                  args.recovery_from,
                                                  args.round_index,
                                                  args.query_ratio,
                                                  test_acc,
                                                  _acc,
                                                  _fidelity))
