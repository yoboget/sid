import os
import torch
from torch import optim
from models.nn import Mlp
from models.models import DenseGNN, DenseGNN3,NodePredictor



def load_denoiser(config, loader, extra_features, device, prior,
                  denoiser_dir=None):
    """
    Loads the denoiser and critic models, along with their optimizers and schedulers.

    Args:
        config: Configuration object.
        loader: Data loader containing training data.
        extra_features: Object containing extra features information.
        device: Device to load models onto (e.g., 'cuda' or 'cpu').
        prior: Type of diffusion prior ('masking', 'absorbing', 'marginal').
        denoiser_dir: Directory to load pre-trained denoiser model from (optional).
        critic_dir: Directory to load pre-trained critic model from (optional).

    Returns:
        Tuple: ((denoiser, optimizer, scheduler), (critic, critic_optimizer, critic_scheduler))
    """
    sizes  = get_input_sizes(config, loader, extra_features, prior)
    input_sizes, output_sizes, hidden_size, n_node_attr =  sizes

    # --- Denoiser ---
    print("Loading Denoiser...")
    print(input_sizes)
    denoiser = DenseGNN(config, *input_sizes, *output_sizes, hidden_size,
                        norm_out=True).to(device)
    params = list(denoiser.parameters())
    betas = config.training.betas.beta1, config.training.betas.beta2
    opt = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, config.training.lr_decay)
    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f'Number of parameters in the encoder: {n_params}')

    if denoiser_dir is not None:
        if config.dataset in ['qm9', 'qm9_cc', 'zinc', 'qm9H']:
            filename = 'best_run_fcd_ema.pt'
        else:
            filename = 'best_run_avg_ema.pt'
        # filename = 'best_run_nspdk_ema.pt'
        denoiser = load_trained_model(denoiser, 'denoiser', denoiser_dir, filename, device)

    return denoiser, opt, scheduler


def load_trained_model(model, model_name, model_dir, filename, device):
    model_path = os.path.join(model_dir, filename)
    saved_model = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_model[model_name])
    return model

def get_input_sizes(config, loader, extra_features, prior, denoiser=True):
    n_extra_feat = extra_features.n_features
    batch = next(iter(loader['train']))
    n_node_attr = batch.x.size(-1)
    n_edge_attr = batch.edge_attr.size(1)
    nhf = config.model.nhf

    print(f"Node features: {n_node_attr}, Extra features: {n_extra_feat}")
    if denoiser:
        nnf_in = n_node_attr + n_extra_feat
        nef_in = n_edge_attr + 1

        if prior == 'masking':
            if n_node_attr > 1:
                nnf_in += 1
            nef_in += 1
        else:
            nnf_in += 1
            nnf_in += 1

        if extra_features.rrwp:
            nef_in += 10

        nnf_out = n_node_attr
        nef_out = n_edge_attr
        if n_node_attr > 1:
            nef_out += 1
        return (nnf_in, nef_in), (nnf_out, nef_out), nhf, n_node_attr
    else:
        nnf_in = n_node_attr + n_extra_feat
        if n_node_attr == 1:
            nnf_in += 1
        nef_in = n_edge_attr + 1
        return nnf_in, nef_in, nhf

def load_critic(config, loader, extra_features, device, prior, model_dir=None) :
    if config.train_critic or config.sampling.critical:
        nnf_in, nef_in, nhf = get_input_sizes(config, loader, extra_features, prior, denoiser=False)
        critic = DenseGNN3(config, nnf_in+2, nef_in, 1, 1, nhf, norm_out=True).to(device)
        params = list(critic.parameters())
        betas = config.training.betas.beta1, config.training.betas.beta2
        opt_critic = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(opt_critic, config.training.lr_decay)
        if model_dir is not None:
            #filename = 'best_run_nspdk.pt'
            # filename = 'best_run_avg.pt'
            filename = 'best_run_loss.pt'
            critic = load_trained_model(critic, 'denoiser', model_dir, filename, device)
        return critic, opt_critic, scheduler_critic
    else:
        return None, None, None

def load_regressor(config, loader, extra_features, device, prior, model_dir=None, cf_guidance=False, c_idx=False):
    nnf_in, nef_in, nhf = get_input_sizes(config, loader, extra_features, prior, cf_guidance, denoiser=False)

    if c_idx is None:
        dy = 3
    else:
        dy = len(c_idx)

    if config.regressor.denoiser:
        nnf_in += 2

    regressor = DenseGNN3(config, nnf_in, nef_in, dy, 1, nhf, norm_out=True, graph_predictor=True).to(device)
    params = list(regressor.parameters())
    betas = config.training.betas.beta1, config.training.betas.beta2
    opt_critic = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
    scheduler_critic = optim.lr_scheduler.ExponentialLR(opt_critic, config.training.lr_decay)
    if model_dir is not None:
        filename = 'best_run_loss.pt'
        regressor = load_trained_model(regressor, 'denoiser', model_dir, filename, device)
    return regressor, opt_critic, scheduler_critic

def load_node_predictor(config, nhf, device, model_dir):
    NF_IN = 3
    nf_out = config.data.max_num_nodes
    node_predictor = Mlp(NF_IN, nf_out, [nhf, nhf]).to(device)
    # node_predictor = NodePredictor(NF_IN, nf_out, [nhf, nhf], 5).to(device)
    params = list(node_predictor.parameters())
    betas = config.training.betas.beta1, config.training.betas.beta2
    opt_node_predictor = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
    scheduler_node_predictor = optim.lr_scheduler.ExponentialLR(opt_node_predictor, config.training.lr_decay)
    if model_dir is not None:
        filename = 'best_run_loss.pt'
        node_predictor = load_trained_model(node_predictor, 'denoiser', model_dir, filename, device)
    return node_predictor, opt_node_predictor, scheduler_node_predictor