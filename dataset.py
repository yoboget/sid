import os
import json
import torch
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from utils.utils import batch_to_networkx

from transforms import Qm9Transform, Qm9ConditionalTransform, Qm9DiscreteGuidanceTransform, Qm9HTransform
from data.datasets import ZINC, QM9_, KekulizedMolDataset, SpectreGraphDataset

def get_dataset(config):

    transforms = None
    if config.dataset == 'qm9H':
        transforms = Qm9HTransform()
        data = QM9_(f'./data/qm9H', pre_transform=transforms)
        idx = torch.randperm(len(data))
        train_idx, test_idx = idx[10000:], idx[:10000]
        train_idx, val_idx = train_idx[config.training.val_size:], train_idx[:config.training.val_size]
        train, val, test = data[train_idx], data[test_idx], data[val_idx]
        test_size = 10_000

    elif config.dataset == 'qm9':
        transforms = Qm9Transform()
        data = KekulizedMolDataset(f'./data/qm9/', dataset='qm9', pre_transform=transforms)
        train_idx, test_idx = get_indices(config, 'qm9', len(data))
        perm = torch.randperm(len(train_idx))
        train_idx = train_idx[perm]
        train_idx, val_idx = train_idx[config.training.val_size:], train_idx[:config.training.val_size]
        train, val, test = data[train_idx], data[test_idx], data[val_idx]
        test_size = 10_000

    elif config.dataset == 'qm9_cc':
        transforms = Qm9ConditionalTransform()
        data = QM9_(f'./data/qm9_cc', pre_transform=transforms)
        SEED = 42
        train_idx, test_idx = idx[10000:], idx[:10000]
        train_idx, val_idx = train_idx[config.log.n_val_samples:], train_idx[:config.log.n_val_samples]
        train, val, test = data[train_idx], data[test_idx], data[val_idx]
        test_size = 10_000

    elif config.dataset == 'qm9_dg':
        transforms = Qm9DiscreteGuidanceTransform()
        data = QM9_(f'./data/qm9_dg', pre_transform=transforms)
        SEED = 42
        torch.manual_seed(SEED)
        idx = torch.randperm(len(data))
        train_idx, test_idx = idx[10000:], idx[:10000]
        train_idx, val_idx = train_idx[config.training.val_size:], train_idx[:config.training.val_size]
        train, val, test = data[train_idx], data[test_idx], data[val_idx]
        test_size = 10_000

    elif config.dataset == 'zinc':
        train = ZINC(f'./data/zinc', split= 'train', pre_transform=transforms)
        val = ZINC(f'./data/zinc', split= 'val', pre_transform=transforms)
        test = ZINC(f'./data/zinc', split= 'test', pre_transform=transforms)
        test_size = 10_000

    elif config.dataset in ['planar', 'sbm']:
        train = SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split= 'train',
                                    pre_transform=transforms)
        val = SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split= 'val',
                                    pre_transform=transforms)
        test = SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split= 'test',
                                    pre_transform=transforms)
        test_size = len(test)

    train_loader = DataLoader(train, batch_size=config.training.batch_size,
                                  shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=config.log.n_val_samples)
    test_loader = DataLoader(test, shuffle= True, batch_size=test_size, drop_last=False)

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    save_test_set(test_loader, config.dataset)
    return loaders

def get_indices(config, dataset, n_instances):
    with open(os.path.join('./data/qm9/raw/', f'test_idx_qm9.json')) as f:
        test_idx = json.load(f)
        if dataset == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # Create a boolean mask for the training indices
    train_idx = torch.ones(n_instances).bool()
    train_idx[test_idx] = False
    train_idx = train_idx[train_idx]

    return train_idx, test_idx

def save_test_set(test_loader, dataset):
    filepath = f'./data/{dataset}/test_graph.pkl'
    if not os.path.exists(filepath):
        test_batch = next(iter(test_loader))
        if dataset in ['zinc', 'qm9']:
            test_batch.x = test_batch.x.argmax(-1, keepdims=True)
        test_graphs = batch_to_networkx(test_batch)
        with open(f'./data/{dataset}/test_graph.pkl', 'wb') as f:
            pickle.dump(test_graphs, f)