# Model Stealing Attacks Against Inductive Graph Neural Networks

This is a PyTorch implementation of Model Stealing Attacks Against Inductive Graph Neural Networks, as described in our paper:

Yun Shen, Xinlei He, Yufei Han, Yang Zhang, [Model Stealing Attacks Against Inductive Graph Neural Networks](https://arxiv.org/abs/2112.08331) (IEEE S&P 2022)

## Step 0: Setup the environments

```
conda env create --file environment.yaml &&
conda activate gnn_model_stealing && 
# Install GraphGallery
wget https://github.com/EdisonLeeeee/GraphGallery/archive/refs/tags/1.0.0.tar.gz && 
tar -zxvf 1.0.0.tar.gz && 
cd GraphGallery-1.0.0/ &&
pip install -e . --verbose &&
cd ..
```

## Step 1: Train the target models:

```
cd code;
python train_target_model.py --dataset citeseer_full --target-model gat  --num-hidden 256

# You can also run it with a specified gpu (e.g., gpu02):
python train_target_model.py --dataset citeseer_full --target-model gat  --num-hidden 256 --gpu 2
```
Note that we use the following datasets, target model architectures, and numbers of hidden neurons in our paper:
```
--dataset:      ['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']
--target-model: ['gat', 'gin', 'sage']
--num-hidden:   [64, 128, 256]
```

## Step 2: Conduct the model stealing attacks

```
# Type I attack:
python3 attack.py --dataset citeseer_full --target-model-dim 256 --num-hidden 256 --target-model gat --surrogate-model gin --recovery-from prediction --query_ratio 1.0 --structure original

# Type II attack:
python3 attack.py --dataset citeseer_full --target-model-dim 256 --num-hidden 256 --target-model gat --surrogate-model gin --recovery-from prediction --query_ratio 1.0 --structure idgl
```

Explainations:
```
--dataset:          ['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']  # Datasets used to train the surrogate model
--target-model-dim: [64, 128, 256]                                                              # Numbers of hidden neurons for the target model
--num-hidden:       [64, 128, 256]                                                              # Numbers of hidden neurons for the surrogate model
--target-model:     ['gat', 'gin', 'sage']                                                      # Target model's architecuture
--surrogate-model:  ['gat', 'gin', 'sage']                                                      # Surrogate model's architecuture
--recovery-from:    ['prediction', 'embedding', 'projection']                                   # Target model's response
--query_ratio:      [0.1, 0.2, ..., 1.0]                                                        # Ratio of query graph used to train the surrogate model, e.g., 1.0 means we use the whole query graph (30% of the whole dataset); 0.5 means we use half of the query graph (15% of the whole dataset);
--structure:        ['original', 'idgl']                                                        # Type I/II attacks, 'original' means we use the original graph structure and 'idgl' means we use idgl to reconstruct the graph structure.
```


## Notes

1. To train the target model, we randomly sample 60% of the nodes to construct the training graph;
2. To train the surrogate model, for each dataset, we split them into three parts.
    - The first part consists of 20\% randomly sampled nodes that are left;
    - The second part consists of 30\% randomly sampled nodes, forming our query graph $\mathbf{G}_Q$.
    - The third part consists of the rest 50\% of the nodes, functioning as the testing data for both $\mathcal{M}_T$ and $\mathcal{M}_S$.
3. We follow the official IDGL implementation from [IDGL](https://github.com/hugochan/IDGL).


## Cite

If you use this code, please consider citing the following papers:

```
@inproceedings{SHHZ22,
author = {Yun Shen and Xinlei He and Yufei Han and Yang Zhang},
title = {{Model Stealing Attacks Against Inductive Graph Neural Networks}},
booktitle = {{IEEE Symposium on Security and Privacy (S\&P)}},
publisher = {IEEE},
year = {2022}
}

@inproceedings{CWZ20,
author = {Yu Chen and Lingfei Wu and Mohammed J. Zaki},
title = {{Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings}},
booktitle = {{Annual Conference on Neural Information Processing Systems (NeurIPS)}},
publisher = {NeurIPS},
year = {2020}
}
```