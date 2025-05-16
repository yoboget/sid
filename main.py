import torch

import wandb
import yaml
from dataset import get_dataset
from easydict import EasyDict as edict
from trainer import Trainer

from utils.parser import parse_args


def main() -> None:
    # Parse command line arguments
    args = parse_args()
    work_type = args.work_type
    dataset = args.dataset

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
        config.denoiser_dir = args.denoiser_dir
        wandb.init(project=f'sid_{dataset}_sample', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        with torch.no_grad():
            X, A, mask, mask_adj= trainer.sampler(config.log.n_samples_generation, trainer.denoiser,
                                                  critic=trainer.critic, iter_denoising=config.sampling.id,
                                                  lambda_guidance=50., condition_off=False)

            trainer.eval_samples(X, A, mask, mask_adj, ema=True, conditional_values=trainer.sampler.cond_batch)


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

