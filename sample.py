
import torch
import torch.nn.functional as F
import time
import numpy as np

from utils.noising import NoiseScheduleDiscrete
from utils.func import  get_edge_mask, mask_adj_batch
from utils.conditional_sampling import get_cfg_conditional_distribution, get_gradiant
from torch.distributions.categorical import Categorical


class Sampler:
    def __init__(self, models, prior, noiser, data_infos, extra_features, sampling_batch_size,
                 diff, device):

        self.denoiser, self.critic = models
        self.device = device
        self.max_num_nodes, self.n_node_attr, self.n_edge_attr, self.num_node_distribution = data_infos
        self.noiser = noiser
        self.extra_features = extra_features
        self.T = noiser.T
        self.diff_x, self.diff_e = diff
        self.prior = prior
        self.noise_sched = NoiseScheduleDiscrete('cosine', self.T)
        self.sampling_batch_size = sampling_batch_size


    def __call__(self, n_samples, models, iter_denoising=True, critic=None):
        self.denoiser.eval()
        self.critical = False if critic is None else True
        if self.critic is not None: self.critic.eval()

        batch_size = self.sampling_batch_size if self.sampling_batch_size < n_samples else n_samples
        x, a, mask, mask_adj = self.sample_batch(batch_size,iter_denoising=iter_denoising)
        remaining_samples = n_samples - batch_size

        while remaining_samples > 0:
            x_, a_, mask_, mask_adj_ = self.sample_batch(batch_size, iter_denoising=iter_denoising)
            x, a, mask, mask_adj = (torch.cat((x , x_), dim=0), torch.cat((a , a_), dim=0),
                                torch.cat((mask , mask_), dim=0), torch.cat((mask_adj, mask_adj_), dim=0))
            remaining_samples -= batch_size
        X, A, mask, mask_adj = x[:n_samples], a[:n_samples], mask[:n_samples], mask_adj[:n_samples]
        return X, A, mask, mask_adj


    def sample_batch(self, n_samples, iter_denoising=True):
        print('Sampling starts... ')

        n = Categorical(probs=self.num_node_distribution).sample((n_samples,)).to(self.device).squeeze()
        mask = torch.tril(torch.ones(self.max_num_nodes, self.max_num_nodes)).to(self.device)[n-1].bool()
        x_t, a_t, mask, mask_adj = self.sample_noise(n_samples, mask, iter_denoising)

        if iter_denoising == 'dfm':
            x_t = x_t[..., :-2]

        elaps_list = []
        start_time = time.time()
        for i, t in enumerate(reversed(range(0, self.T))):
            if iter_denoising == 'sid':
                x_t, a_t = self.iterative_denoising_step(t, x_t, a_t, mask, mask_adj)
            elif iter_denoising == 'dif':
                x_t, a_t = self.backward_step_digress(t, x_t, a_t, mask)
            elif iter_denoising == 'dfm':
                delta_t = 1 / self.T
                t = 1-(t / self.T)- delta_t
                x_t, a_t = self.flow_matching_step(x_t, a_t, t, delta_t, (mask, mask_adj))
            else:
                raise NotImplementedError ('Sampling method not implemented.')
            if (i + 1) % 100 == 0:
                print(f'{i + 1} timesteps done. Sampling resumes...')
            if i % 100 == 0:
                elaps = time.time() - start_time
                print(elaps)
                elaps_list.append(elaps)
                start_time = time.time()
        time_array = np.asarray([elaps_list[1:6]])
        print('time mean and std : ', time_array.mean(), time_array.std())
        print('Sampling finished.')
        return x_t, a_t, mask.bool(), mask_adj.bool().squeeze()

    def flow_matching_step(self, x_t, a_t, t, delta_t, masks):
        """
        Compute a backward step in standard discrete flow matching.

        Given:
          - x_t: the current state at time t (e.g., a NumPy array or tensor),
          - t: the current time,
          - delta_t: the time increment (so that s = t + delta_t),
          - velocity_fn: a callable that computes the velocity field v(x, t).

        The backward step is computed using a backward Euler update:
          x_s = x_t - delta_t * v(x_t, t)

        Returns:
          - x_s: the state at time s = t + delta_t.
        """
        # Compute x1
        mask, mask_adj = masks
        x_t_time = self.add_time_features(x_t, t, mask)
        x_t_extra, a_t_extra = self.extra_features(x_t_time, a_t, mask)
        x1_hat, a1_hat = self.denoiser(x_t_extra, a_t_extra, mask.bool())
        if a1_hat.size(-1) == 1:
            a1_hat = torch.cat((a1_hat, 1 - a1_hat), dim=-1)
        x1_hat, a1_hat = x1_hat.softmax(-1) * mask.unsqueeze(-1), a1_hat.softmax(-1) * mask_adj.unsqueeze(-1)
        vx, va = self.compute_velocity(x1_hat, x_t, a1_hat, a_t, t, delta_t)

        px_s = x_t + delta_t * vx
        pa_s = a_t + delta_t * va

        px_s = px_s / px_s.sum(-1, keepdim=True)
        pa_s = pa_s / pa_s.sum(-1, keepdim=True)
        x_s, a_s = self.sample_ps(px_s, pa_s, masks)

        if t + delta_t == 1:
            x_s, a_s = x_s.argmax(-1) * mask, a_s.argmax(-1) * mask_adj
            x_s, a_s = x_s.int(), a_s.int()
        return x_s, a_s

    def sample_ps(self, px, pa, masks):
        mask, mask_adj = masks
        x_s = torch.zeros_like(px)
        x = Categorical(probs=px[mask.bool()]).sample().to(self.device)
        x_s[mask.bool()] = F.one_hot(x, num_classes=px.shape[-1]).float()
        x_s = x_s * mask.unsqueeze(-1)

        a_s =  torch.zeros_like(pa)
        a = Categorical(probs=pa[mask_adj.bool()]).sample().to(self.device)
        a_s[mask_adj.bool()] = F.one_hot(a, num_classes=pa.shape[-1]).float()
        a_s = a_s.permute(0, 3, 1, 2)
        a_s = a_s.tril(-1) + a_s.tril(-1).transpose(2, 3)
        a_s = a_s.permute(0, 2, 3, 1)
        a_s = a_s * mask_adj.unsqueeze(-1)
        return x_s, a_s

    def compute_velocity(self, x1_hat, xt, a1_hat, at, t, delta_t):
        """
        Compute the velocity field v(x, t) for flow matching using independent coupling
        and a convex interpolant.


        The velocity field is given by the time derivative of x_t:
          v(x_t, t) = d/dt x_t = lambda_dot(t) * (x1 - x0)

        Args:
            x_t: The current state at time t (typically computed as the convex combination above).
            t: The current time.
            delta_t: The time increment.
            lambda_dot_fn: A callable that computes the derivative of lambda(t) with respect to t.

        Returns:
            The velocity field v(x_t, t) = alpha_dot(t)/(1-alpha) * (x1_hat - xt).
        """

        # Compute the scalor alpha_dot(t)/(1-alpha_t)
        s = t + delta_t
        alpha_s = self.noiser.noise_schedule.get_alpha_bar(t_normalized=torch.tensor([s], device=self.device).float())
        alpha_t = self.noiser.noise_schedule.get_alpha_bar(t_normalized=torch.tensor([t], device=self.device).float())
        alpha_dot = (alpha_t-alpha_s) / delta_t
        scalor = alpha_dot/(alpha_t)
        return scalor * (x1_hat -xt), scalor * (a1_hat - at)

    def add_time_features(self, x, t, mask):
        t = torch.tensor([t], device=self.device).float()
        alpha_t = self.noiser.noise_schedule.get_alpha_bar(t_normalized=1-t)
        alpha_t = alpha_t * torch.ones(*x.shape[:-1], 1).to(x.device)
        x = torch.cat((x, alpha_t, 1-alpha_t), dim=-1)
        return x * mask.unsqueeze(-1)

    def backward_step_digress(self, t, x_t, e_t, mask):
        """
            Inspired from Digress diffusion_model_discrete.py
            Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well
           """
        bs = mask.shape[0]
        device = e_t.device
        t = t + 1
        s = t - 1

        beta_t = self.noiser.noise_schedule(t_int=torch.tensor([t]).float())
        alpha_s_bar = self.noiser.noise_schedule.get_alpha_bar(t_int=torch.tensor([s], device=self.device).float())
        alpha_t_bar = self.noiser.noise_schedule.get_alpha_bar(t_int=torch.tensor([t], device=self.device).float())

        # Retrieve transitions matrix
        if x_t is not None:
            Qtbx = self.diff_x.get_Qt_bar(alpha_t_bar, device)
            Qsbx = self.diff_x.get_Qt_bar(alpha_s_bar, device)
            Qtx = self.diff_x.get_Qt(beta_t, device)

        Qtbe = self.diff_e.get_Qt_bar(alpha_t_bar, device)
        Qsbe = self.diff_e.get_Qt_bar(alpha_s_bar, device)
        Qte = self.diff_e.get_Qt(beta_t, device)

        x_t_extra, e_t_extra = self.extra_features(x_t, e_t, mask)
        x_hat, e_hat = self.denoiser(x_t_extra, e_t_extra, mask.bool())

        if e_hat.shape[-1] == 1:
            e_hat = e_hat.sigmoid()
            p_e = torch.cat([e_hat, 1 - e_hat], dim=-1)
        else:
            p_e = e_hat.softmax(dim=-1)

        if self.n_node_attr > 1:
            p_x = x_hat.softmax(dim=-1) * mask.unsqueeze(-1)
            if self.prior in ['absorbing', 'marginal']:
                x_t = x_t[..., :x_hat.size(-1)]

            ps_x = self.compute_batched_over0_posterior_distribution_digress(x_t, Qtx, Qsbx, Qtbx)

            if self.prior in ['masking']:
                p_x = torch.cat([p_x, torch.zeros(p_x.size(0), p_x.size(1), 1, device=self.device)], dim=-1)

            weighted_x = p_x.unsqueeze(-1) * ps_x.squeeze()  # N, d0, d_t-1
            unnormalized_prob_x = weighted_x.sum(dim=-2)  # N, d_t-1
            unnormalized_prob_x[torch.sum(unnormalized_prob_x, dim=-1) == 0] = 1e-5
            p_x = unnormalized_prob_x / torch.sum(unnormalized_prob_x, dim=-1, keepdim=True)  # N, d_t-1
            if self.prior in ['masking'] and s ==0 :
                p_x = p_x[..., :-1]/p_x[..., :-1].sum(-1, keepdim=True)

            assert ((p_x.sum(dim=-1) - 1).abs() < 1e-4).all()
            x_s = Categorical(probs=p_x).sample().to(device)  # *mask
            x_s = F.one_hot(x_s, num_classes=x_t.shape[-1]).float() * mask.unsqueeze(-1)
            if self.prior in ['marginal', 'absorbing']:
                alpha_bar = alpha_s_bar.repeat(x_s.size(0), self.max_num_nodes, 1)
                x_s = torch.cat((x_t, alpha_bar, 1 - alpha_bar), dim=-1)

        else:
            x_s = x_t

        p_e = mask_adj_batch(p_e, mask)
        ps_e = self.compute_batched_over0_posterior_distribution_digress(e_t, Qte, Qsbe, Qtbe)

        p_e = p_e.reshape(bs, -1, p_e.shape[-1])
        if self.prior in ['masking']:
            p_e = torch.cat([p_e, torch.zeros(p_e.size(0), p_e.size(1), 1, device=self.device)], dim=-1)


        weighted_e = p_e.unsqueeze(-1) * ps_e  # N, nc, d0, d_t-1
        unnormalized_p_e = weighted_e.sum(dim=2).squeeze()
        unnormalized_p_e[torch.sum(unnormalized_p_e, dim=-1) == 0] = 1e-5

        if self.prior in ['masking']:
            unnormalized_p_e = unnormalized_p_e * (1-torch.eye(unnormalized_p_e.size(-1))[-1].unsqueeze(0).unsqueeze(0).to(self.device))

        p_e = unnormalized_p_e / torch.sum(unnormalized_p_e, dim=-1, keepdim=True)
        p_e = p_e.reshape(bs, self.max_num_nodes, self.max_num_nodes, e_t.shape[-1])
        # print(p_e[..., 0].mean())
        assert ((p_e.sum(dim=-1) - 1).abs() < 1e-4).all()

        # print(p_e[..., 0].mean())
        e_s = Categorical(probs=p_e).sample().to(device)
        #print(e_s.sum())
        e_st = torch.triu(e_s, diagonal=1)
        e_s = e_st + e_st.transpose(-1, -2)
        edge_mask = get_edge_mask(mask)
        e_s[~edge_mask] = 0
        e_s = F.one_hot(e_s, num_classes=e_t.shape[-1]).float() * edge_mask.unsqueeze(-1)
        if s == 0:
            if self.n_node_attr > 1:
                x_0 = x_s[..., :p_x.size(-1)].argmax(-1) * mask
            else:
                x_0 = x_s[..., :1]
            a_0 = e_s.argmax(-1) * edge_mask
            return x_0.int(), a_0.int()
        else:
            return x_s, e_s

    def compute_batched_over0_posterior_distribution_digress(self, X_t, Qt, Qsb, Qtb):
        """ M: X or E
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
            X_t: bs, n, dt          or bs, n, n, dt
            Qt: bs, d_t-1, dt
            Qsb: bs, d0, d_t-1
            Qtb: bs, d0, dt.
        """
        # Flatten feature tensors
        # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
        # bs x (n ** 2) x d
        X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # bs x N x dt

        Qt_T = Qt.transpose(-1, -2)  # bs, dt, d_t-1
        left_term = X_t @ Qt_T.squeeze()  # bs, N, d_t-1 SQUEEZE ADDED 26.4
        left_term = left_term.unsqueeze(dim=2)  # bs, N, 1, d_t-1

        right_term = Qsb.unsqueeze(1)  # bs, 1, d0, d_t-1
        numerator = left_term * right_term  # bs, N, d0, d_t-1

        X_t_transposed = X_t.transpose(-1, -2)  # bs, dt, N

        prod = Qtb @ X_t_transposed  # bs, d0, N
        prod = prod.transpose(-1, -2)  # bs, N, d0
        denominator = prod.unsqueeze(-1)  # bs, N, d0, 1
        denominator[denominator == 0] = 1e-6

        out = numerator / denominator
        return out


    def iterative_denoising_step(self, t, x_t, a_t, mask, mask_adj):
        """
            Inspired from Digress diffusion_model_discrete.py
            Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well
           """
        self.t = t + 1

        #### DENOISING ###
        x_0, a_0 = self.denoising_step(x_t, a_t, mask, mask_adj)

        if self.n_node_attr == 1:
            x_0 = x_t[..., :-2]
        if self.prior in ['masking']:
            if self.n_node_attr > 1:
                x_0[~self.noise_masks[0]] = x_t[~self.noise_masks[0]][..., :-1]
            a_0[~self.noise_masks[1]] = a_t[~self.noise_masks[1]][..., :-1]

        assert (a_0 != a_0.transpose(1, 2)).sum() == 0, (a_0 != a_0.transpose(1, 2)).sum()

        #### RE-NOISING ###
        if self.critical:
            x_s, a_s = self.noising_with_critic(x_0, a_0, mask, t)
        else:
            alpha_bar = self.noiser.get_alpha_bar(torch.tensor(t-1)/self.T)
            alpha_bar_x = alpha_bar * torch.ones(*x_0.shape[:-1]).to(self.device)
            alpha_bar_a = alpha_bar * torch.ones(*a_0.shape[:-1]).to(self.device)
            x_s, a_s, self.noise_masks, _ = self.noiser(x_0, a_0, mask, alpha_bars=(alpha_bar_x, alpha_bar_a))
        if self.t == 1:
            if self.n_node_attr == 1:
                x_0 = torch.cat((torch.ones(*x_t.shape[:-1], 1).to(self.device), x_0), dim=-1)
            x_0 = x_0.argmax(-1) * mask

            a_0 = a_0.argmax(-1) * mask_adj
            return x_0.int(), a_0.int()
        else:

            return x_s * mask.unsqueeze(-1), a_s * mask_adj.unsqueeze(-1)


    def denoising_step(self, x_t, a_t, mask, mask_adj):
        if self.n_node_attr == 1:
            x_t = torch.cat((torch.ones(*x_t.shape[:-1], 1).to(self.device), x_t), dim=-1)
        x_t, a_t = self.extra_features(x_t, a_t, mask)

        p_x, p_a = self.denoiser(x_t, a_t, mask.bool())

        p_x = torch.softmax(p_x, dim=-1)
        if self.n_edge_attr > 1:
            p_a = torch.softmax(p_a, dim=-1)
        else:
            p_a = torch.sigmoid(p_a)
            p_a = torch.cat((p_a, 1 - p_a), dim=-1)


        x = Categorical(probs=p_x).sample().to(self.device)
        x_0 = F.one_hot(x, num_classes=p_x.shape[-1]).float()
        x_0 = x_0 * mask.unsqueeze(-1)

        a = Categorical(probs=p_a).sample().to(self.device)
        a = F.one_hot(a, num_classes=p_a.shape[-1]).float()
        a = a.permute(0, 3, 1, 2)
        a = a.tril(-1) + a.tril(-1).transpose(2, 3)
        a_0 = a.permute(0, 2, 3, 1)
        a_0 = a_0 * mask_adj.unsqueeze(-1)
        return x_0, a_0


    def sample_noise(self, n_samples, mask, iter_denoising):
        if self.n_node_attr > 1:
            if self.prior == 'masking':
                self.q0x = torch.eye(self.n_node_attr+1)[-1].to(self.device)
                x_t = self.q0x.view(1, 1, -1).repeat(n_samples, self.max_num_nodes, 1)
            elif self.prior == 'absorbing':
                self.q0x = torch.eye(self.n_node_attr)[0].to(self.device)
                x_t = self.q0x.view(1, 1, -1).repeat(n_samples, self.max_num_nodes, 1)
                x_t = torch.cat((x_t, torch.ones(*x_t.shape[:-1], 1).to(self.device)), dim=-1)

                alpha_bar = self.noiser.get_alpha_bar(torch.ones(1, device=x_t.device))
                alpha_bar = alpha_bar.repeat(n_samples, self.max_num_nodes, 1)
                x_t = torch.cat((x_t, alpha_bar, 1-alpha_bar), dim=-1)
            elif self.prior == 'marginal':
                self.q0x = self.diff_x.base_distrib.to(self.device)
                x_t = Categorical(probs=self.q0x).sample((n_samples, self.max_num_nodes)).to(self.device).squeeze()
                x_t = F.one_hot(x_t, num_classes=self.q0x.shape[-1]).float() * mask.unsqueeze(-1)
                alpha_bar = self.noiser.get_alpha_bar(torch.ones(1, device=x_t.device))
                alpha_bar = alpha_bar.repeat(n_samples, self.max_num_nodes, 1)
                x_t = torch.cat((x_t, alpha_bar, 1-alpha_bar), dim=-1)
            x_t = x_t * mask.unsqueeze(-1)
        else:

            if iter_denoising == 'sid':
                x_t = torch.ones(n_samples, self.max_num_nodes, 0).to(self.device) * mask.unsqueeze(-1)
            else: # discrete diffusion
                x_t = torch.ones(n_samples, self.max_num_nodes, 1).to(self.device) * mask.unsqueeze(-1)
            if self.prior in ['absorbing', 'marginal']:
                alpha_bar = self.noiser.get_alpha_bar(torch.ones(1, device=x_t.device))
                alpha_bar = alpha_bar.repeat(n_samples, self.max_num_nodes, 1)
                x_t = torch.cat((x_t, alpha_bar, 1-alpha_bar), dim=-1)

        if self.prior == 'masking':
            self.q0a = torch.eye(self.n_edge_attr + 2)[-1].to(self.device)
            a_t = self.q0a.view(1, 1, 1, -1).repeat(n_samples, self.max_num_nodes, self.max_num_nodes, 1)
        elif self.prior == 'absorbing':
            self.q0a = torch.eye(self.n_edge_attr+1)[0].to(self.device)
            a_t = self.q0a.view(1, 1, 1, -1).repeat(n_samples, self.max_num_nodes, self.max_num_nodes, 1)
        elif self.prior == 'marginal':
            self.q0a = self.diff_e.base_distrib
            a_t = Categorical(probs=self.q0a).sample((n_samples, self.max_num_nodes,
                                                     self.max_num_nodes)).to(self.device).squeeze()
            a_t = F.one_hot(a_t, num_classes=self.q0a.shape[-1]).float()
            a_t = a_t.permute(0, 3, 1, 2).flatten(0, 1)
            a_t = a_t.tril(-1) + a_t.tril(-1).transpose(1, 2)
            a_t = a_t.reshape(n_samples, self.q0a.shape[-1], self.max_num_nodes, self.max_num_nodes).permute(0, 2, 3, 1)

        mask_adj =  mask.unsqueeze(1) * mask.unsqueeze(2) * (1-torch.eye(a_t.size(1)).to(self.device).unsqueeze(0))
        a_t = a_t * mask_adj.unsqueeze(-1)
        self.noise_masks = mask.bool().squeeze(), mask_adj.bool().squeeze()
        return x_t, a_t, mask, mask_adj

    def get_classifier_gradiant(self, x, a, mask, t, noising_guidance=False):
        dx, da =  5, 4#x.size(-1), a.size(-1)
        if noising_guidance:
            alpha = torch.tensor([1, 0]).view(1, 1, -1).repeat(*x.shape[:2], 1).to(self.device)
            x = torch.cat((x, alpha), dim=-1)
        else:
            alpha = self.noiser.get_alpha_bar(torch.tensor((t - 1) / self.T, device=self.device))
            alpha =  torch.tensor([alpha, 1-alpha]).view(1, 1, -1).repeat(*x.shape[:2], 1).to(self.device)
            x = torch.cat((x, alpha), dim=-1)
        x, a = self.extra_features(x, a, mask)
        grad_x, grad_a = get_gradiant(self.regressor, x, a, mask.bool(), self.cond)
        grad_x, grad_a = grad_x[..., :dx], grad_a[..., :da]
        LAMBDA = -2000
        p_x = (grad_x * LAMBDA).softmax(-1) # (grad_x*LAMBDA).exp() / (grad_x*LAMBDA).exp().sum(-1) but stable
        p_a = (grad_a * LAMBDA).softmax(-1)
        p_x = torch.clamp(p_x, min=0., max=1.)
        p_a = torch.clamp(p_a, min=0., max=1.)
        p_x[p_x.sum(-1) == 0] = torch.eye(dx)[0].to(self.device)
        p_a[p_a.sum(-1) == 0] = torch.eye(da)[0].to(self.device)
        return p_x, p_a

    def noising_with_critic(self, x_0, a_0, mask, t):
        alpha_bar = self.noiser.get_alpha_bar(torch.tensor((t - 1) / self.T, device=self.device))
        if self.n_node_attr == 1:
            x = torch.cat((torch.ones(*x_0.shape[:-1], 1).to(self.device), x_0), dim=-1)
            x, a = self.extra_features(x, a_0, mask)
        else:
            x, a = self.extra_features(x_0, a_0, mask)
        if self.prior in ['masking']:
            x = torch.cat((x, alpha_bar.repeat(*x.shape[:-1], 1)), dim=-1)
        else:
            alpha_bar_ = alpha_bar.repeat(*x.shape[:-1], 1)
            x = torch.cat((x, alpha_bar_, 1 - alpha_bar_), dim=-1)
        if self.guidance:
            if self.condition_off:
                dy = self.cond_batch.size(-1)
                cond = torch.eye(dy + 1)[-1].view(1, 1, -1).repeat(*x_0.shape[:2], 1).to(self.device)
                x = torch.cat((x, cond), dim=-1)
            else:
                cond = self.cond.repeat(self.max_num_nodes, 1).reshape(*x_0.shape[:2], -1) * mask.unsqueeze(-1)
                cond = cond[..., self.conditional_idx]
                cond_in = torch.zeros(*x_0.shape[:2], 1, device=self.device)
                x = torch.cat((x, cond, cond_in), dim=-1)

        logit_x, logit_a = self.critic(x, a, mask.bool())
        alpha_logit = torch.log(alpha_bar / (1 - alpha_bar))
        alpha_bar_x = (logit_x + alpha_logit.view(-1, 1, 1)).sigmoid().squeeze()
        alpha_bar_a = (logit_a + alpha_logit.view(-1, 1, 1, 1)).sigmoid().squeeze()
        x_s, a_s, self.noise_masks, _ = self.noiser(x_0, a_0, mask, alpha_bars=(alpha_bar_x, alpha_bar_a))
        return x_s, a_s