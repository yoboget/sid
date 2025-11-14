import time

import torch

import wandb
import yaml
from dataset import get_dataset
from easydict import EasyDict as edict
from trainer import Trainer
import numpy as np

from utils.parser import parse_args

def get_dense_batch(batch, dataset='qm9'):
    from torch_geometric.utils import to_dense_batch, to_dense_adj
    x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
    x, mask = to_dense_batch(x, batch_idx)
    if dataset == 'qm9':
        x = x[..., :4].argmax(-1)
    elif dataset in ['qm9H', 'qm9_cc']:
        x = x[..., :5].argmax(-1)
    else:
        x = x[..., :9].argmax(-1)
    adj = to_dense_adj(edge_index, batch_idx, edge_attr)
    adj_ = adj.argmax(-1) + 1
    adj_[adj.sum(-1) == 0] = 0
    mask_adj = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    return x, adj_, mask, mask_adj

def main() -> None:
    # Parse command line arguments
    args = parse_args()
    work_type = args.work_type
    dataset = args.dataset

    if args.wandb == 'no':
        args.wandb = 'disabled'
    elif args.wandb == 'on':
        args.wandb = 'online'
    elif args.wandb == 'off':
        args.wandb = 'offline'

    config_path = f'./config/{dataset}.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config = edict(config)
    config = update_config(config, dataset, work_type, args)

    if work_type == 'train':
        wandb.init(project=f'sid_{dataset}', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        trainer.train()

    elif work_type == 'sample':
        runs = []
        N_RUNS = 5
        config.denoiser_dir = args.denoiser_dir
        wandb.init(project=f'sid_{dataset}_sample', config=config, mode=args.wandb)
        times = []

        for r in range(N_RUNS):
            dataloader = get_dataset(config)
            trainer = Trainer(dataloader, config)
            with torch.no_grad():
                X, A, mask, mask_adj = get_dense_batch(next(iter(dataloader['train'])), dataset='zinc')
                start_time = time.time()
                # X, A, mask, mask_adj= trainer.sampler(config.log.n_samples_generation, trainer.denoiser,
                #                                       critic=trainer.critic, iter_denoising=config.sampling.id)
                sampling_time = time.time() - start_time
                wandb.log({'sampling_time': sampling_time})
                times.append(sampling_time)
                run = trainer.eval_samples(X, A, mask, mask_adj, ema=True)
                runs.append(run)

        keys = runs[0].keys()
        latex_format = {}
        for key in keys:
            val = []
            for run in runs:
                val.append(run[key])
            mean = np.asarray(val).mean()
            std = np.asarray(val).std()
            print(f'mean {key}: {mean}')
            print(f'std {key}: {std}')
            if key in ['valid', 'unique', 'novel']:
                mean *= 100
                std *= 100
                latex_format[key] = f'${mean:.2f} \pm {std:.2f}$'
            elif key in ['nspdk', 'degree', 'cluster', 'orbit', 'spectral']:
                mean *= 1000
                std *= 1000
                latex_format[key] = f'${mean:.3f} \pm {std:.3f}$'
            else:
                latex_format[key] = f'${mean:.3f} \pm {std:.3f}$'

        if dataset in ['qm9', 'qm9_cc', 'qm9H', 'zinc']:
            print(
                f'{latex_format["valid"]} &  {latex_format["fcd"]} & {latex_format["nspdk"]} & {latex_format["unique"]} & {latex_format["novel"]} \\')
        else:
            latex_format["valid"], latex_format['novel'] = 0, 0
            print(
                f'{latex_format["valid"]} &  {latex_format["degree"]} & {latex_format["cluster"]} & {latex_format["orbit"]}& {latex_format["spectral"]} & {latex_format["novel"]} \\')
        times = np.asarray(times)
        print(f'time: {times.mean()} \pm {times.std()}')
        wandb.finish()

def update_config(config, dataset, work_type, args):
    config.dataset = dataset
    config.work_type = work_type
    config.train_model = args.train_model
    config.train_critic = args.train_critic
    config.denoiser_dir = args.denoiser_dir
    config.critic_dir = args.critic_dir
    return config


if __name__ == "__main__":
    main()

