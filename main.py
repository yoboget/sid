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

    if work_type == 'train' or  work_type == 'all':
        wandb.init(project=f'unmask_diffusion_{dataset}', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        trainer.train()

    elif work_type == 'sample':
        config.denoiser_dir = args.denoiser_dir
        wandb.init(project=f'unmask_diffusion_{dataset}_sample', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        # Ts = [4, 8]
        # for T in Ts:
        #     config.model.T = T
        trainer = Trainer(dataloader, config)
        with torch.no_grad():
            critical = False if config.critic_dir is None else True
            N_RUNS = 1
            for _ in range(N_RUNS):

                X, A, mask, mask_adj= trainer.sampler(config.log.n_samples_generation, trainer.denoiser,
                                                      critic=trainer.critic, iter_denoising=config.sampling.id,
                                                      lambda_guidance=50., condition_off=False)

                # trainer.sampler.cond= torch.tensor([1, -1, -1], device='cpu')
                trainer.eval_samples(X, A, mask, mask_adj, ema=True, conditional_values=trainer.sampler.cond_batch)

                # X, A, mask, mask_adj = trainer.sampler(config.log.n_samples_generation, trainer.denoiser,
                #                                        critic=trainer.critic,
                #                                        iter_denoising=config.sampling.id, condition_off=True)
                # trainer.eval_samples(X, A, mask, mask_adj, ema=True, conditional_values=trainer.sampler.cond)


def update_config(config, dataset, work_type, args):
    config.dataset = dataset
    config.work_type = work_type
    config.train_model = args.train_model
    config.train_critic = args.train_critic
    config.train_regressor = args.train_regressor
    config.train_node_predictor = args.train_node_predictor
    config.denoiser_dir = args.denoiser_dir
    config.critic_dir = args.critic_dir
    config.regressor_dir = args.regressor_dir
    config.node_predictor_dir = args.node_predictor_dir
    return config


if __name__ == "__main__":
    main()

