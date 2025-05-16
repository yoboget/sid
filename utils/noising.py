import os.path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class MarginalUniformTransition:
    def __init__(self, marginals):
        if marginals.dim() == 2:
            self.n_classes = marginals.shape[-1]
            self.u = marginals.expand(self.n_classes, -1).unsqueeze(0)
        else:
            self.n_classes = marginals.shape[-1]
            self.u = marginals.unsqueeze(0).expand(self.n_classes, -1).unsqueeze(0)
        self.marginals = marginals
        self.base_distrib = marginals


    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """

        if self.marginals.dim() == 2:
            beta_t = beta_t.view(-1, 1, 1, 1)
        else:
            beta_t = beta_t.view(-1, 1, 1)
        beta_t = beta_t.to(device)
        self.u = self.u.to(device)
        q_x = beta_t * self.u + (1 - beta_t) * torch.eye(self.n_classes, device=device).unsqueeze(0)
        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        if self.marginals.dim() == 2:
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        else:
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u = self.u.to(device)
        if self.marginals.dim() == 2:
            q_x = alpha_bar_t * torch.eye(self.n_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u
        else:
            q_x = alpha_bar_t * torch.eye(self.n_classes, device=device) + (1 - alpha_bar_t) * self.u
        return q_x

class NoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(NoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = custom_beta_schedule_discrete(timesteps)

        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas.type_as(t_int)[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)

def noise(indices, batch, T, diffusion, noise_schedule, quantizer, alpha_bar=None):
    n = batch.bincount()
    device = indices.device
    if alpha_bar is None:
        t = torch.randint(low=1, high=T+1, size=(n.size(0),)).to(device).int()
        t = torch.repeat_interleave(t, n, dim=0)
        alpha_bar = noise_schedule.get_alpha_bar(t_int=t)
        t = t / T
    else:
        t = None
    # Artificially annotates virtual node to avoid issue with probs, but it is masked later anyway
    Qt_bar_x = diffusion.get_Qt_bar(alpha_bar, device)
    Qt_bar_x.squeeze_()
    z = F.one_hot(indices, diffusion.n_classes).float()
    probs = z @ Qt_bar_x.to(device)
    noisy_z = Categorical(probs=probs.squeeze()).sample()
    #noisy_z = F.one_hot(noisy_z, diffusion.n_classes).float()
    noisy_z = quantizer.indices_to_zq(noisy_z)
    return noisy_z, t

class Noiser:
    def __init__(self, timesteps, noise_schedule = 'cosine'):
        self.noise_schedule = NoiseScheduleDiscrete(noise_schedule, timesteps)
        self.T = timesteps

    def __call__(self, batch, t=None):
        device = batch.edge_index.device
        if t is None:
            t = torch.randint(self.T, (batch.batch.max()+1,)).to(device)
        alpha_bar = self.noise_schedule.get_alpha_bar(t_int=t)
        batch.x = self.noise_x(batch.x, batch.batch, alpha_bar)
        batch.edge_index, batch.edge_attr = self.noise_edge(batch.edge_index, batch.edge_attr, batch.batch, alpha_bar)
        return batch

    def noise_x(self, x, batch, alpha_bar):
        n = batch.bincount()
        alpha_bar = alpha_bar.repeat_interleave(n, dim=0)
        mask = torch.bernoulli(alpha_bar).unsqueeze(-1)
        x = x * mask
        x = torch.cat((x, 1-mask), dim=-1)
        return x

    def noise_edge(self, edge_index, edge_attr, batch, alpha_bar):
        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        edge_attr = edge_attr[edge_index[0]]
        batch_edge = batch[edge_index[0]]
        m = batch_edge.bincount()
        alpha_bar = alpha_bar.repeat_interleave(m, dim=0)
        mask = torch.bernoulli(alpha_bar).unsqueeze(-1)
        edge_attr = edge_attr * mask
        edge_attr = torch.cat((edge_attr, 1 - mask), dim=-1)
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0)
        return edge_index, edge_attr


class NoiserDense:
    def __init__(self, timesteps, prior, noise_schedule = 'cosine', distributions=None,
                 mask_prior=False):

        self.noise_schedule = NoiseScheduleDiscrete(noise_schedule, timesteps)
        self.schedule = noise_schedule
        self.T = timesteps
        self.prior = prior
        if distributions is not None:
            self.marginal_x = distributions[0]
            self.marginal_a = distributions[1]

        self.mask_prior = mask_prior

        if self.schedule == 'cosine':
            self.get_alpha_bar = self.cosine
        elif self.schedule == 'linear':
            self.get_alpha_bar = self.identity
        elif self.schedule == 'cocosine':
            self.get_alpha_bar = self.cocosine

    def __call__(self, x, a, mask, alpha_bars=None, t=None):
        self.device = a.device

        if alpha_bars is None:
            if t is None:
                t = torch.rand((x.size(0),)).to(self.device)
            alpha_bar = self.get_alpha_bar(t)
            alpha_bar_x = alpha_bar.view(-1, 1) * torch.ones(x.size(0), x.size(1)).to(self.device)
            alpha_bar_a = alpha_bar.view(-1, 1, 1) * torch.ones(a.size(0), a.size(1), a.size(2)).to(self.device)

        else:
            alpha_bar_x, alpha_bar_a = alpha_bars[0], alpha_bars[1]
        if x.size(-1) > 1:
            x, noise_mask_x = self.noise_x(x, mask, alpha_bar_x)
        else:
            noise_mask_x = None
            if self.prior in ['absorbing', 'marginal']:
                x = torch.cat((x, alpha_bar_x.unsqueeze(-1), 1-alpha_bar_x.unsqueeze(-1)), dim=-1)

        a, noise_mask_a = self.noise_adj(a, mask, alpha_bar_a)
        masks = noise_mask_x, noise_mask_a
        return x, a, masks, (alpha_bar_x, alpha_bar_a)


    def noise_x(self, x, mask, alpha_bar):
        absorb = torch.bernoulli(alpha_bar).unsqueeze(-1)

        x = x * absorb
        noise_mask = (1 - absorb).bool().squeeze()
        if self.prior == 'masking':
            x = torch.cat((x, 1 - absorb), dim=-1)

        elif self.prior == 'absorbing':
            x[noise_mask] = torch.eye(x.size(-1))[0].to(self.device)
            x = torch.cat((x, alpha_bar.unsqueeze(-1), 1-alpha_bar.unsqueeze(-1)), dim=-1)

        elif self.prior == 'marginal':
            if noise_mask.sum() > 0:
                noise = Categorical(probs=self.marginal_x).sample(x[noise_mask].shape[:-1])
                x[noise_mask] = F.one_hot(noise, num_classes=x.size(-1)).float().to(self.device)
            x = torch.cat((x, alpha_bar.unsqueeze(-1), 1-alpha_bar.unsqueeze(-1)), dim=-1)

        x = x * mask.unsqueeze(-1)
        return x, noise_mask

    def noise_adj(self, a, mask, alpha_bar):
        mask = mask.unsqueeze(1) * mask.unsqueeze(2) * (1-torch.eye(a.size(1)).to(a.device).unsqueeze(0))
        absorb = torch.bernoulli(alpha_bar)
        absorb = absorb.tril(-1) + absorb.tril(-1).transpose(1, 2)

        a = a * absorb.unsqueeze(-1)
        noise_mask = ~absorb.bool()
        if self.prior == 'masking':
            a = torch.cat((a, 1-absorb.unsqueeze(-1)), dim=-1)
            noise_mask.squeeze_(-1)

        elif self.prior == 'absorbing':
            a[noise_mask] = torch.eye(a.size(-1))[0].to(self.device)
            noise_mask.squeeze_(-1)

        elif self.prior == 'marginal':
            if noise_mask.sum() > 0:
                noise = Categorical(probs=self.marginal_a).sample(a[noise_mask].shape[:-1]).to(self.device)
                noise = F.one_hot(noise, num_classes=a.size(-1)).float()
                a[noise_mask] = noise
                bs, n, _, d = a.shape
                a = a.permute(0, 3, 1, 2).flatten(0, 1)
                a = a.tril(-1) + a.tril(-1).transpose(1, 2)
                a = a.reshape(bs, d, n, n).permute(0, 2, 3, 1)

        a = a * mask.unsqueeze(-1)
        return a, noise_mask

    def cosine(self, t):
        return torch.cos(0.5 * np.pi * (t))

    def cocosine(self, t):
        return torch.tensor((2 / np.pi) * torch.arccos(1-t))

    def identity(self, t):
        return t

def get_distribution(loader, dataset):
    if os.path.isfile(f'./data/{dataset}/distributions.pt'):
        node_distrib, edge_distrib = torch.load(f'./data/{dataset}/distributions.pt')
    else:
        print('Recompute distributions')
        node_sum, edge_sum, n_node_pairs = 0, 0, 0
        for batch in loader:
            x = batch.x
            edge_attr = batch.edge_attr
            node_sum += x.sum(0)
            edge_sum += edge_attr.sum(0)
            n = batch.batch.bincount()
            n_node_pairs += (n * (n-1)).sum()
        edge_sum = torch.cat((n_node_pairs.unsqueeze(-1)-edge_sum.sum(-1), edge_sum), -1)
        node_distrib, edge_distrib = node_sum/node_sum.sum(), edge_sum/n_node_pairs
        torch.save((node_distrib, edge_distrib), f'./data/{dataset}/distributions.pt')
    return node_distrib, edge_distrib
