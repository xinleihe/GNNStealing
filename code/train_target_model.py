import argparse
import torch as th
from src.utils import load_graphgallery_data, split_graph
from graphgallery.datasets import NPZDataset
from src.gat import run_gat_target
from src.gin import run_gin_target
from src.sage import run_sage_target, evaluate_sage_target
import pickle
import os

argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=int, default=1,
                       help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--target-model', type=str, default='gat')
argparser.add_argument('--dataset', type=str, default='dblp',
                       help="['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm' 'amazon_photo']")
argparser.add_argument('--num-epochs', type=int, default=200)
argparser.add_argument('--num-hidden', type=int, default=256)
argparser.add_argument('--num-layers', type=int, default=3)
argparser.add_argument('--fan-out', type=str, default='10,10,10')
argparser.add_argument('--batch-size', type=int, default=512)
argparser.add_argument('--val-batch-size', type=int, default=512)
argparser.add_argument('--log-every', type=int, default=20)
argparser.add_argument('--eval-every', type=int, default=100)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=8,
                       help="Number of sampling processes. Use 0 for no extra process.")
argparser.add_argument('--inductive', action='store_true',
                       help="Inductive learning setting")
argparser.add_argument('--graphgallery', action='store_true', default=False)
argparser.add_argument('--save-pred', type=str, default='')
argparser.add_argument('--head', type=int, default=4)
argparser.add_argument('--wd', type=float, default=0)
args, _ = argparser.parse_known_args()
# params for sage
if args.target_model == "sage":
    args.inductive = True
    args.fan_out = '10,25'
# datasets = ['dblp', 'pubmed', 'citeseer', 'coauthor_phy', 'acm' 'amazon_photo']
# print(datasets)


if args.gpu >= 0:
    device = th.device('cuda:%d' % args.gpu)
else:
    device = th.device('cpu')


# Load data and preprocessing
g, n_classes = load_graphgallery_data(args.dataset)
in_feats = g.ndata['features'].shape[1]
labels = g.ndata['labels']

train_g, val_g, test_g = split_graph(g, frac_list=[0.6, 0.2, 0.2])
print(train_g.number_of_nodes(), val_g.number_of_nodes(), test_g.number_of_nodes())

train_g.create_formats_()
val_g.create_formats_()
test_g.create_formats_()

# Train the target model
# print(args)
# exit()
SAVE_PATH = './target_model_%s_%s/' % (args.target_model, args.num_hidden)
SAVE_NAME = 'target_model_%s_%s' % (args.target_model, args.dataset)
os.makedirs(SAVE_PATH, exist_ok=True)

if args.target_model == "gat":
    data = train_g, val_g, test_g, in_feats, labels, n_classes, g, args.head
    target_model = run_gat_target(args, device, data)
    th.save(target_model, SAVE_PATH + SAVE_NAME)

elif args.target_model == "gin":
    data = train_g, val_g, test_g, in_feats, labels, n_classes
    target_model = run_gin_target(args, device, data)
    th.save(target_model.state_dict(), SAVE_PATH + SAVE_NAME)

elif args.target_model == "sage":
    # print("##########")
    data = in_feats, n_classes, train_g, val_g, test_g
    target_model = run_sage_target(args, device, data)
    th.save(target_model, SAVE_PATH + SAVE_NAME)

else:
    raise ValueError("target-model should be gat, gin, or sage")


# Save model args
pickle.dump(args, open(SAVE_PATH + 'model_args', 'wb'))
print("Finish")
