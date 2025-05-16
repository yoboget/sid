
import time
import copy
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from torch_geometric.data import Batch

from utils.extra_feat import ExtraFeatures
from utils.noising import NoiserDense, MarginalUniformTransition, get_distribution
from utils.func import batch_to_dense
from utils.node_distribtuion import get_num_nodes_distribution
from eval.metrics import SamplingMetrics
from models.loaders import load_denoiser, load_critic, load_regressor, load_node_predictor
from logger import RunningMetric, save_model
from sample import Sampler


class Trainer:
    def __init__(self, loaders, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'
        print(f'Run on {self.device}')

        self.train_model = config.train_model
        self.train_critic = config.train_critic
        self.train_regressor = config.train_regressor
        self.train_node_predictor = config.train_node_predictor

        self.sampling = config.work_type == 'sample'
        if self.train_model and not self.sampling:
            config.denoiser_dir = None

        self.loaders = loaders
        self.extra_features = ExtraFeatures(config)
        self.prior = config.model.prior

        if config.dataset == 'qm9_cc' or 'qm9_dg':
            self.conditional = config.model.conditional
            self.cf_guidance = config.sampling.classifier_free_guidance
            self.classifier_guidance = config.sampling.classifier_guidance
            self.c_idx = None
        else:
            self.cf_guidance, self.classifier_guidance = False, False

        self.sample_ema_model = True
        ### LOAD MODELS ###
        ### DENOISER ###
        denoiser = load_denoiser(config, loaders, self.extra_features, self.device, prior=config.model.prior,
                                 denoiser_dir=config.denoiser_dir, conditional=self.conditional,
                                 conditional_idx=self.c_idx)
        self.denoiser, self.opt, self.sched = denoiser
        self.denoiser_ema = copy.deepcopy(self.denoiser)

        ### CRITIC ###
        critic = load_critic(config, loaders, self.extra_features, self.device, prior=config.model.prior,
                                 model_dir=config.critic_dir, cf_guidance=self.cf_guidance)
        self.critic, self.opt_critic, self.scheduler_critic = critic # if critic not needed, critic is set to 3*(None)
        self.critic_ema = copy.deepcopy(self.critic)

        ### REGRESSOR ###
        regressor = load_regressor(config, loaders, self.extra_features, self.device, prior=config.model.prior,
                                   model_dir=config.regressor_dir, cf_guidance=self.cf_guidance, c_idx=self.c_idx)
        self.regressor, self.opt_regressor, self.scheduler_regressor = regressor
        self.regressor_ema = copy.deepcopy(self.regressor)

        ### NODE PREDICTOR ###
        node_predictor = load_node_predictor(config, config.model.nhf, self.device, model_dir=config.node_predictor_dir)
        self.node_predictor, self.opt_node_predictor, self.scheduler_node_predictor = node_predictor
        self.node_predictor_ema = copy.deepcopy(self.node_predictor)


        # For comparison
        batch = next(iter(self.loaders['train']))
        nnf = batch.x.size(-1)
        nef = batch.edge_attr.size(-1)

        if self.prior == 'masking':
            self.edge_marginal = torch.eye(nef + 2, device=self.device)[-1]
            self.node_marginal = torch.eye(nnf + 1, device=self.device)[-1]
        elif self.prior == 'absorbing':
            self.edge_marginal = torch.eye(nef + 1, device=self.device)[0]
            self.node_marginal = torch.eye(nnf, device=self.device)[0]
            self.id = config.sampling.id
        elif self.prior == 'marginal':
            self.node_marginal, self.edge_marginal = get_distribution(self.loaders['train'], config.dataset)
        self.id = config.sampling.id
        node, edge = get_distribution(self.loaders['train'], config.dataset)


        self.diff_x = MarginalUniformTransition(self.node_marginal)
        self.diff_e = MarginalUniformTransition(self.edge_marginal)
        self.diff = self.diff_x, self.diff_e

        # Extract configuration variables
        self.epochs = config.training.epochs
        self.decay_iteration = config.training.decay_iteration
        self.max_num_nodes = config.data.max_num_nodes

        # Define Logger
        self.metrics = RunningMetric(['loss', 'node', 'edge'])
        self.metrics_critic = RunningMetric(['loss_critic', 'node_critic', 'edge_critic'])
        self.metrics_regressor = RunningMetric(['loss_regressor'])
        self.metrics_node_predictor = RunningMetric(['loss_node_predictor'])

        self.n_logging_steps = config.log.n_loggin_steps
        self.n_logging_epochs = config.log.n_loggin_epochs
        self.best_run = {}

        self.noiser = NoiserDense(config.model.T, self.prior, distributions=(node, edge))

        num_node_distrib = get_num_nodes_distribution(self.loaders, self.max_num_nodes, config.dataset)
        data_infos = self.max_num_nodes, nnf, nef, num_node_distrib
        diff = self.diff_x, self.diff_e

        val_loader = self.loaders['val'] if self.cf_guidance or self.classifier_guidance else None
        val_loader = self.loaders['val']
        self.sampling_batch_size = config.log.sampling_batch_size
        models = self.denoiser, self.critic, self.regressor, self.node_predictor
        self.sampler = Sampler(models, self.prior, self.noiser, data_infos, self.extra_features,
                               self.sampling_batch_size, diff, self.device, conditional_loader=val_loader,
                               cf_guidance=self.cf_guidance, classifier_guidance=self.classifier_guidance)

        ref_loader = self.loaders['test'] if self.sampling else self.loaders['val']
        self.eval_samples = SamplingMetrics(config.dataset, self.max_num_nodes, self.sampling, ref_loader=ref_loader)
        self.val_size = config.log.n_val_samples

        # batch = next(iter(self.loaders['test'])).to(self.device)
        # X, A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)
        # diag_mask = (1 - torch.eye(A.size(1)).to(A.device).unsqueeze(0)).bool()
        # adj_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & diag_mask
        # self.eval_samples(X[..., :5].argmax(-1), A.argmax(-1), mask, adj_mask, ema=True, conditional_values=self.sampler.cond)

        self.dataset = config.dataset
        self.save_graphs = config.log.save_graphs


    def train(self) -> None:
        print(f'The training set contains {len(self.loaders["train"])} batches with size {len(self.loaders["train"])}')
        starting_time = time.time(), time.process_time()
        self.step = 0
        print('Training starts...')
        for self.epoch in range(1, self.epochs + 1):
            # TRAIN
            for batch in self.loaders['train']:
                self.step += 1

                # TRAIN MODEL
                if self.train_model:
                    critic_input = self.fit(batch.to(self.device), train=True)
                    if self.step % self.n_logging_steps == 0:
                        self.metrics.log(self.step, key='iter', times=starting_time)
                elif self.train_critic:
                    with torch.no_grad():
                        critic_input = self.fit(batch.to(self.device), train=False)

                # TRAIN CRITIC
                if self.train_critic:
                    # self.fit_critic(*critic_input)
                    self.fit_critic2(batch.to(self.device))
                    if self.step % self.n_logging_steps == 0:
                        self.metrics_critic.log(self.step, key='iter', times=starting_time)

                # TRAIN REGRESSOR
                if self.train_regressor:
                    self.fit_regressor(batch.to(self.device))

                # TRAIN NODE PREDICTOR
                if self.train_node_predictor:
                    self.fit_node_predictor(batch.to(self.device))

                # TRAIN NODE PREDICTOR

            if self.train_model:
                self.metrics.log(self.step, key='train', times=starting_time)
            if self.train_critic:
                self.metrics_critic.log(self.step, key='train', times=starting_time)
            if self.train_regressor:
                self.metrics_regressor.log(self.step, key='train', times=starting_time)
            if self.train_node_predictor:
                self.metrics_node_predictor.log(self.step, key='train', times=starting_time)

            # VAL
            with (torch.no_grad()):
                for batch in self.loaders['val']:
                    if self.train_model:
                        critic_input = self.fit(batch.to(self.device), train=False)
                    if self.train_critic:
                        # self.fit_critic(*critic_input, train=False)
                        self.fit_critic2(batch.to(self.device), train=False)
                    if self.train_regressor:
                        self.fit_regressor(batch.to(self.device), train=False)
                    if self.train_node_predictor:
                        self.fit_node_predictor(batch.to(self.device), train=False)

                if self.train_model:
                    val_metrics = self.metrics.log(self.step, key='val', times=starting_time)
                if self.train_critic:
                    val_metrics_critic = self.metrics_critic.log(self.step, key='val', times=starting_time)
                if self.train_regressor:
                    val_metrics_regressor = self.metrics_regressor.log(self.step, key='val', times=starting_time)
                if self.train_node_predictor:
                    val_metrics_node_predictor = self.metrics_node_predictor.log(self.step, key='val',
                                                                                 times=starting_time)

                if self.epoch % self.n_logging_epochs == 0:
                    if self.train_model:
                        ### SAMPLE EMA MODEL ###
                        if self.sample_ema_model:
                            cond_val = self.sampler.cond_batch if self.cf_guidance or self.classifier_guidance else None
                            X, A, mask, mask_adj = self.sampler(self.val_size, self.denoiser_ema, iter_denoising=self.id,
                                                                critic=self.critic)
                            sampling_metrics = self.eval_samples(X, A, mask, mask_adj, ema=True,
                                                                 conditional_values=cond_val)
                            to_save = self.denoiser_ema, self.opt, self.sched
                            self.save_model(to_save, sampling_metrics, val_metrics['loss'], ema=True)
                        else:
                            ### SAMPLE RUNNING MODEL ###
                            X, A, mask, mask_adj = self.sampler(self.val_size, self.denoiser, iter_denoising=self.id,
                                                                critic=self.critic)
                            sampling_metrics = self.eval_samples(X, A, mask, mask_adj, ema=False)
                            to_save = self.denoiser, self.opt, self.sched
                            self.save_model(to_save, sampling_metrics, val_metrics['loss'], ema=False)

                        if self.cf_guidance or self.classifier_guidance:
                            X, A, mask, mask_adj = self.sampler(self.val_size, self.denoiser_ema,
                                                                iter_denoising=self.id,
                                                                critic=self.critic, condition_off=True)
                            sampling_metrics = self.eval_samples(X, A, mask, mask_adj, ema=True,
                                                                 conditional_values=cond_val,
                                                                 condition_off=True)
                            to_save = self.denoiser_ema, self.opt, self.sched
                            self.save_model(to_save, sampling_metrics, val_metrics['loss'], ema=True)
                    else:
                        sampling_metrics = None

                    if self.train_critic:
                        to_save = self.critic, self.opt_critic, self.scheduler_critic
                        self.save_model(to_save, sampling_metrics, val_metrics_critic['loss_critic'])
                    if self.train_regressor:
                        to_save = self.regressor, self.opt_regressor, self.scheduler_regressor
                        self.save_model(to_save, sampling_metrics, val_metrics_regressor['loss_regressor'])
                    if self.train_node_predictor:
                        to_save = self.node_predictor, self.opt_node_predictor, self.scheduler_node_predictor
                        self.save_model(to_save, sampling_metrics, val_metrics_node_predictor['loss_node_predictor'])

            if self.step % self.decay_iteration == 0:
                if self.train_model:
                    self.sched.step()
                if self.train_critic:
                    self.scheduler_critic.step()
                if self.train_regressor:
                    self.scheduler_regressor.step()
                if self.train_node_predictor:
                    self.scheduler_node_predictor.step()

    def fit(self, batch: Batch, train: bool = True):
        # Set model mode (train/eval) based on the "train" parameter
        if train:
            self.opt.zero_grad()
            self.denoiser.train()
        else:
            self.denoiser.eval()
        # PREP
        self.X, self.A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)
        diag_mask = (1-torch.eye(self.A.size(1)).to(self.A.device).unsqueeze(0)).bool()
        adj_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & diag_mask

        X_noisy, A_noisy, noise_masks, alphas = self.noiser(self.X, self.A, mask)
        X_noisy, A_noisy = self.extra_features(X_noisy, A_noisy, mask)

        if self.cf_guidance:
            cond = batch.c.repeat(self.max_num_nodes, 1).reshape(*X_noisy.shape[:2], -1) * mask.unsqueeze(-1)
            if self.c_idx is not None:
                cond = cond[..., self.c_idx]
            is_cond = torch.bernoulli(0.25*torch.ones(X_noisy.shape[0], 1))
            self.is_cond = is_cond.repeat(1, self.max_num_nodes).unsqueeze(-1).to(self.device)
            X_noisy = torch.cat((X_noisy, self.cond, self.is_cond), dim=-1)

        else: self.cond = None

        if not train and self.sample_ema_model:
            X_pred_, A_pred_ = self.denoiser_ema(X_noisy, A_noisy, mask)

        else:
            X_pred_, A_pred_ = self.denoiser(X_noisy, A_noisy, mask)

        # if self.cf_guidance:
            #loss_cond = self.compute_conditional_loss(X_pred_, A_pred_, mask, self.cond)


        X_pred = X_pred_[mask]
        X_targ = self.X[mask]


        A_pred = A_pred_[adj_mask]
        A_targ = self.A[adj_mask]

        if self.prior == 'masking':
            A_pred = A_pred[noise_masks[1][adj_mask]]
            A_targ = A_targ[noise_masks[1][adj_mask]]
            if noise_masks[0] is not None:
                X_pred = X_pred[noise_masks[0][mask]]
                X_targ = X_targ[noise_masks[0][mask]]

        if X_targ.size(-1) > 1:
            loss_x = F.cross_entropy(X_pred, X_targ)
            n, m = X_targ.size(0), A_targ.size(0)

            loss_a = F.cross_entropy(A_pred, A_targ)
            loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_a

        else:
            A_targ = A_targ[..., :1]
            loss_a = F.binary_cross_entropy(A_pred.sigmoid(), A_targ)
            loss = loss_a
            loss_x = torch.zeros(1, dtype=X_targ.dtype, device=X_targ.device)

        if train:
            loss.backward()
            self.opt.step()
            update_ema(self.denoiser, self.denoiser_ema)

        to_log = [loss.item(), loss_x.item(), loss_a.item()]
        self.metrics.step(to_log, train)

        if self.train_critic:
            if self.prior == 'masking':
                if X_targ.size(-1) > 1:
                    X_pred_[~noise_masks[0]] = self.X[~noise_masks[0]]
                    X_pred_[~mask] = X_pred_[~mask].softmax(-1)
                    X_pred_[noise_masks[0]] = X_pred_[noise_masks[0]].softmax(-1)
                else:
                    X_pred_ = self.X

                if A_pred_.size(-1) > 1:
                    A_pred_[~noise_masks[1]] = self.A[~noise_masks[1]]
                    A_pred_[~mask] = A_pred_[~mask].softmax(-1)
                    A_pred_[noise_masks[1]] = A_pred_[noise_masks[1]].softmax(-1)
                else:
                    A = self.A[..., :1]
                    A_pred_[~noise_masks[1]] = A[~noise_masks[1]]
                    A_pred_[~mask] = A_pred_[~mask].sigmoid()
                    A_pred_[noise_masks[1]] = A_pred_[noise_masks[1]].sigmoid()
                    A_pred_ = torch.cat((A_pred_, 1-A_pred_), dim=-1)
            else:
                if X_targ.size(-1) > 1:
                    X_pred_[~noise_masks[0]] = self.X[~noise_masks[0]]
                    X_pred_[~mask] = X_pred_[~mask].softmax(-1)
                    X_pred_[noise_masks[0]] = X_pred_[noise_masks[0]].softmax(-1)
                else:
                    X_pred_ = self.X

                if A_pred_.size(-1) > 1:
                    A_pred_[~noise_masks[1]] = self.A[~noise_masks[1]]
                    A_pred_[~mask] = A_pred_[~mask].softmax(-1)
                    A_pred_[noise_masks[1]] = A_pred_[noise_masks[1]].softmax(-1)
                else:
                    A = self.A[..., :1]
                    A_pred_[~noise_masks[1]] = A[~noise_masks[1]]
                    A_pred_[~mask] = A_pred_[~mask].sigmoid()
                    A_pred_[noise_masks[1]] = A_pred_[noise_masks[1]].sigmoid()
                    A_pred_ = torch.cat((A_pred_, 1-A_pred_), dim=-1)


        return X_pred_, A_pred_, (mask, adj_mask), noise_masks, alphas

    def fit_critic(self, X, A, masks, noise_masks, alphas, train=True):
        if train:
            self.opt_critic.zero_grad()
            self.critic.train()
        else:
            self.critic.eval()

        mask, adj_mask = masks

        if X.size(-1) > 1:
            X_hat = Categorical(probs=X).sample()
            x_0 = F.one_hot(X_hat, num_classes=X.shape[-1]).float() * mask.unsqueeze(-1)
        else:
            x_0 = X.long()
        A_hat = Categorical(probs=A).sample()
        # We noise the uppper triangular and copy the value to the lower to ensure symetry
        A_hat = torch.triu(A_hat, diagonal=1)
        A_hat = A_hat + A_hat.transpose(-1, -2)
        a_0 = F.one_hot(A_hat, num_classes=A.shape[-1]).float() * adj_mask.unsqueeze(-1)

        if self.prior == 'masking':
            targ_x = 1 - noise_masks[0][mask].unsqueeze(1).float()
            targ_a = 1 - noise_masks[1][adj_mask].unsqueeze(1).float()
        else:
            targ_x = (x_0 == self.X).all(-1)[mask].unsqueeze(1).float()
            targ_a = (a_0 == self.A).all(-1)[adj_mask].unsqueeze(1).float()

        x_0, a_0 = self.extra_features(x_0, a_0, mask)
        if self.prior == 'masking':
            x_0 = torch.cat((x_0, alphas[0].unsqueeze(-1)), dim=-1)
        else:
            x_0 = torch.cat((x_0, alphas[0].unsqueeze(-1), 1-alphas[0].unsqueeze(-1)), dim=-1)
        if self.cf_guidance:
            x_0 = torch.cat((x_0, self.cond, self.is_cond), dim=-1)

        if not train and self.sample_ema_model:
            x_pred, a_pred = self.critic_ema(x_0, a_0, mask)
        else:
            x_pred, a_pred = self.critic(x_0, a_0, mask)


        alpha_logit_x = torch.log(alphas[0]/(1-alphas[0]))
        alpha_logit_a = torch.log(alphas[1]/(1-alphas[1]))

        x_pred, a_pred = x_pred + alpha_logit_x.unsqueeze(-1), a_pred + alpha_logit_a.unsqueeze(-1)
        loss_a = F.binary_cross_entropy(a_pred[adj_mask].sigmoid(), targ_a)

        if X.size(-1) > 1:
            loss_x = F.binary_cross_entropy(x_pred[mask].sigmoid(), targ_x)
            n, m = x_pred[mask].size(0), a_pred[adj_mask].size(0)
            loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_a
        else:
            loss_x = torch.zeros(1, dtype=x_pred.dtype, device=x_pred.device)
            loss = loss_a

        if train:
            loss.backward()
            self.opt_critic.step()
            update_ema(self.critic, self.critic_ema)

        to_log = [loss.item(), loss_x.item(), loss_a.item()]
        self.metrics_critic.step(to_log, train)


    def fit_critic2(self, batch, train=True):
        if train:
            self.opt_critic.zero_grad()
            self.critic.train()
        else:
            self.critic.eval()

        X, A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)
        adj_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & (1 - torch.eye(A.size(1)).to(A.device).unsqueeze(0)).bool()

        X_noisy, A_noisy, noise_masks, alphas = self.noiser(X, A, mask)
        X_noisy, A_noisy = self.extra_features(X_noisy, A_noisy, mask)

        if not train and self.sample_ema_model:
            x_pred, a_pred = self.critic_ema(X_noisy, A_noisy, mask)
        else:
            x_pred, a_pred = self.critic(X_noisy, A_noisy, mask)

        alpha_logit_x = torch.log(alphas[0]/(1-alphas[0]))
        alpha_logit_a = torch.log(alphas[1]/(1-alphas[1]))

        x_pred, a_pred = x_pred.squeeze() + alpha_logit_x, a_pred + alpha_logit_a.unsqueeze(-1)

        loss_a = F.binary_cross_entropy(a_pred[adj_mask].sigmoid(), 1 - noise_masks[1][adj_mask].unsqueeze(1).float())
        if X.size(-1) > 1:
            loss_x = F.binary_cross_entropy(x_pred[mask].sigmoid(), 1-noise_masks[0][mask].float())
            n, m = x_pred[mask].size(0), a_pred[adj_mask].size(0)
            loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_a
        else:
            loss_x = torch.zeros(1, dtype=x_pred.dtype, device=x_pred.device)
            loss = loss_a

        if train:
            print(loss)
            loss.backward()
            self.opt_critic.step()
            update_ema(self.critic, self.critic_ema)

        to_log = [loss.item(), loss_x.item(), loss_a.item()]
        self.metrics_critic.step(to_log, train)


    def fit_regressor(self, batch, train=True):
        if train:
            self.opt_regressor.zero_grad()
            self.regressor.train()
        else:
            self.regressor.eval()
            self.regressor_ema.eval()

        X, A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)

        ### WE TRAIN THE REGRESSOR ON HALF NOISY - HALF CLEAN DATA ###
        noise_indicator = torch.bernoulli(0.5 * torch.ones(X.shape[0], device=self.device)).bool()
        X = torch.cat((X, torch.zeros_like(X[..., :2])), dim=-1)
        X[noise_indicator], A[noise_indicator], _, _ = self.noiser(X[noise_indicator, :, :-2],
                                                                   A[noise_indicator], mask[noise_indicator])
        X[~noise_indicator][..., -2:] = torch.tensor([1, 0], device=self.device)

        X, A = self.extra_features(X, A, mask)
        if self.dataset == 'qm9_cc':
            cond = batch.c[..., self.c_idx]
        else:
            cond = batch.logP

        if not train and self.sample_ema_model:
            x_pred, a_pred = self.regressor_ema(X, A, mask)
        else:
            x_pred, a_pred = self.regressor(X, A, mask)

        if self.dataset == 'qm9_cc':
            loss = F.mse_loss(x_pred, cond.squeeze())
            print(loss)
        else:
            loss = F.cross_entropy(x_pred, cond.squeeze())

        if train:
            loss.backward()
            self.opt_regressor.step()
            update_ema(self.regressor, self.regressor_ema)

        to_log = [loss.item()]
        self.metrics_regressor.step(to_log, train)

    def fit_node_predictor(self, batch, train=True):
        if train:
            self.opt_node_predictor.zero_grad()
            self.node_predictor.train()
        else:
            self.node_predictor.eval()

        X, A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)
        cond = batch.c

        if not train and self.sample_ema_model:
            n_pred = self.node_predictor_ema(cond)
        else:
            n_pred = self.node_predictor(cond)

        loss = F.nll_loss(n_pred.log_softmax(-1), mask.sum(-1) - 1)

        if train:
            loss.backward()
            self.opt_node_predictor.step()
            update_ema(self.node_predictor, self.node_predictor_ema)

        to_log = [loss.item()]
        self.metrics_node_predictor.step(to_log, train)

    def get_reconstruction_loss(self, x_target, edge_target, x_pred, edge_attr_pred, masks, n, m):
        mask, edge_mask = masks
        x_pred = x_pred * mask.unsqueeze(-1)
        loss_x = F.cross_entropy(x_pred.permute(0, 2, 1), x_target, reduction='none')
        loss_x = (loss_x * mask).mean()
        loss_edge = F.cross_entropy(edge_attr_pred.permute(0, 3, 1, 2), edge_target, reduction='none')
        loss_edge = (loss_edge * edge_mask).mean()
        rec_loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_edge
        return rec_loss, loss_x, loss_edge

    # def sample(self, n_samples, iter_denoising=True, critical=False, ema=False,
    #            conditional_values=None):
    #
    #     self.denoiser.eval()
    #     self.critical = critical
    #     if self.critical:
    #         self.critic.eval()
    #     batch_size = self.sampling_batch_size if self.sampling_batch_size < n_samples else n_samples
    #     if ema:
    #         x, a, mask, mask_adj = self.sample_batch_ema(batch_size, iter_denoising=iter_denoising, critical=critical,
    #                                                      conditional_values=conditional_values)
    #     else:
    #         x, a, mask, mask_adj = self.sample_batch(batch_size, iter_denoising=iter_denoising, critical=critical,
    #                                                  conditional_values=conditional_values)
    #     remaining_samples = n_samples - batch_size
    #     while remaining_samples > 0:
    #         if ema:
    #             x_, a_, mask_, mask_adj_ = self.sample_batch_ema(batch_size, iter_denoising=iter_denoising,
    #                                                          critical=critical, conditional_values=conditional_values)
    #         else:
    #             x_, a_, mask_, mask_adj_ = self.sample_batch(batch_size, iter_denoising=iter_denoising,
    #                                                          critical=critical, conditional_values=conditional_values)
    #         remaining_samples -= batch_size
    #         x = torch.cat((x , x_), dim=0)
    #         a = torch.cat((a , a_), dim=0)
    #         mask = torch.cat((mask , mask_), dim=0)
    #         mask_adj = torch.cat((mask_adj, mask_adj_), dim=0)
    #     X, A, mask, mask_adj = x[:n_samples], a[:n_samples], mask[:n_samples], mask_adj[:n_samples]
    #     return X, A, mask, mask_adj
    #
    #
    # def eval_samples(self, X, A, mask, mask_adj, ema=False, n_run=0, conditional_values=None):
    #     ema_string = '_ema' if ema else ''
    #     distrib_metric = eval_general_stats(X, A, mask, mask_adj, self.dataset)
    #     if self.dataset in ['qm9', 'zinc', 'qm9_conditional']:
    #         metrics = self.get_metrics((X, A, mask), self.dataset, conditional_values)
    #     else:
    #         gen_graphs = get_networkx_from_dense(X, A, mask)
    #         metrics = self.get_metrics(gen_graphs, self.dataset, conditional_values)
    #         if self.save_graphs:
    #             import pickle
    #             fname = f'./dump/{self.dataset}_{self.transition}_{self.prior}_{self.id}{n_run + 1}{ema_string}'
    #             pickle.dump(gen_graphs, open(fname, 'wb'))
    #     if self.sampling:
    #         self.step = None
    #     else:
    #         metrics['epoch'] = self.epoch
    #     wandb.log({f'sampling{ema_string}/': metrics}, step=self.step)
    #     wandb.log({f'distributions{ema_string}/': distrib_metric}, step=self.step)
    #     print(f'sampling{ema_string}: ', metrics)
    #     print(f'distributions{ema_string}: ', distrib_metric)
    #     return metrics
    #
    # def eval_general_stats(self, X, A, mask, mask_adj):
    #     if self.dataset == 'zinc':
    #         node_names = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br','I']
    #         edge_names = ['no_bond', 'single', 'double', 'triple']
    #     elif self.dataset == 'qm9':
    #         node_names = ['C', 'N', 'O', 'F']
    #         edge_names = ['no_bond', 'single', 'double', 'triple']
    #     elif self.dataset == 'qm9_conditional':
    #         node_names = ['H', 'C', 'N', 'O', 'F']
    #         edge_names = ['no_bond', 'single', 'double', 'triple']
    #     elif self.dataset in ['planar', 'sbm']:
    #         node_names = []
    #         edge_names = ['no_edge', 'edge']
    #
    #     if len(node_names) > 0:
    #         node_distrib = X[mask].bincount(minlength=X.shape[-1])
    #         node_distrib = node_distrib / node_distrib.sum()
    #         node_dict = {node_name:node_ratio for node_name, node_ratio in zip(node_names, node_distrib)}
    #     else: node_dict = {}
    #
    #     edge_distrib = A[mask_adj].bincount(minlength=A.shape[-1])
    #     edge_distrib = edge_distrib / edge_distrib.sum()
    #     edge_dict = {edge_name: edge_ratio for edge_name, edge_ratio in zip(edge_names, edge_distrib)}
    #
    #     return node_dict | edge_dict

    def save_model(self,  to_save, metrics, loss, ema=False):
        if metrics is not None:
            if self.dataset in ['zinc', 'qm9', 'qm9_cc', 'qm9_dg']:
                ref_metric_name = 'nspdk'
                ref_metric_name_save = 'nspdk'
                if ema: ref_metric_name_save += '_ema'
                self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                           to_save=to_save, step=self.step, save_name=ref_metric_name_save)
                ref_metric_name = 'fcd'
                ref_metric_name_save = 'fcd'
                if ema: ref_metric_name_save += '_ema'
                self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                           to_save=to_save, step=self.step, save_name=ref_metric_name_save)
            else:
                ref_metric_name = 'avg'
                ref_metric_name_save = 'avg'
                if ema: ref_metric_name_save += '_ema'
                self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                           to_save=to_save, step=self.step, save_name=ref_metric_name_save)
                ref_metric_name = 'valid'
                ref_metric_name_save = 'valid'
                if ema: ref_metric_name_save += '_ema'
                if 'valid' in metrics.keys():
                    self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                               to_save=to_save, step=self.step, save_name=ref_metric_name_save)
        self.best_run = save_model(loss, best_run=self.best_run, to_save=to_save,
                                   step=self.step, save_name='loss')

    def compute_conditional_loss(self, X, A, mask, cond):
        # X = F.gumbel_softmax(X, hard=True, tau=0.1)
        # A = F.gumbel_softmax(A, hard=True, tau=0.1)
        X_ = Categorical(X.softmax(-1)).sample().unsqueeze(-1)
        X_ = torch.zeros_like(X).scatter_(-1, X_, 1)
        X = X + X_ - X.detach()
        # X = X.softmax(-1)

        A_ = Categorical(A.softmax(-1)).sample().unsqueeze(-1)
        A_ = torch.zeros_like(A).scatter_(-1, A_, 1)
        A = A + A_ - A.detach()
        # A = A.softmax(-1)

        A = 0.5 * (A + A.transpose(1, 2))
        X, A = self.extra_features(X, A, mask)
        y_pred, _ = self.regressor(X, A, mask)
        return F.mse_loss(y_pred, self.cond[:, 0])


@torch.no_grad()
def update_ema(model, ema_model, ema_decay=0.999):
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_decay * ema_param.data + (1.0 - ema_decay) * model_param.data

    # Buffer copy for Batch_Norm modules
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data = model_buffer.data.clone()

