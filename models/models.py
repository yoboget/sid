import torch.nn
import torch.nn as nn
from torch_geometric.nn import GraphNorm

from models.nn import DenseLayer2, Mlp, DenseGraphNorm



class DenseGNN(nn.Module):
    def __init__(self, config, nnf_in, nef_in, nnf_out, nef_out, nhf, norm_out=False):
        super().__init__()
        self.sizes = (nnf_in, nef_in, nnf_out, nef_out, nhf)

        self.normalization = config.model.normalization
        mlp_n_layers = config.model.mlp_n_layers
        mlp_hidden_size = nhf
        fact = config.model.edge_node_ratio

        self.linear_in_x = nn.Sequential(nn.Linear(nnf_in, nhf), nn.ReLU(), nn.Linear(nhf, nhf))
        # self.linear_in_e = nn.Sequential(nn.Linear(nef_in, int(nhf / fact)), nn.ReLU(), nn.Linear(int(nhf / fact), int(nhf / fact)))
        self.linear_out_x = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nnf_out))
        # self.linear_out_e = nn.Sequential(nn.Linear(int(nhf / fact), int(nhf / fact)), nn.ReLU(), nn.Linear(int(nhf / fact), nef_out))

        n_layers = config.model.n_layers
        layers = []
        attention_layers = []

        for layer in range(n_layers):
            layers.append(DenseLayer2(nhf, nhf, nhf, nhf, mlp_n_layers, mlp_hidden_size, fact,
                                   normalization=self.normalization))
            attention_layers.append(torch.nn.MultiheadAttention(nhf, 16, batch_first=True))

        self.linear_in_e = nn.Linear(nef_in, int(nhf / fact))
        self.linear_out_e = nn.Linear(int(nhf / fact), nef_out)
        self.layers = nn.Sequential(*layers)
        self.attention_layers = nn.Sequential(*attention_layers)

        # self.linear_in_y = nn.Linear(3, 3*nhf)

        self.norm_out = norm_out
        if norm_out:
            if self.normalization == 'batch_norm':
                self.edge_norm = nn.BatchNorm1d(nef_out)
                self.node_norm = nn.BatchNorm1d(nhf)

            elif self.normalization == 'layer_norm':
                # self.edge_norm = nn.LayerNorm(int(nhf / fact))
                self.edge_norm = nn.LayerNorm(nef_out)
                self.node_norm = nn.LayerNorm(nhf)

            elif self.normalization == 'graph_norm':
                self.edge_norm = DenseGraphNorm(nef_out)
                self.node_norm = DenseGraphNorm(nhf)

    def forward(self, X, A, mask, y=None):
        X = self.linear_in_x(X)
        A = self.linear_in_e(A)
        # if y is not None:
        #     y = y[:, 0]
        #     dy = y.size(-1)
        #     Y = self.linear_in_y(y)
        #     Y = Y.reshape(y.size(0), dy, -1)

        for layer, attention_layer in zip(self.layers, self.attention_layers):
            X_new, A_new = layer(X, A, mask, skip_connection=True)
            X = X + X_new
            A = A + A_new
            # if y is not None:
            #     X = torch.cat((Y, X), dim=1)
            #     mask_y = torch.ones(X.size(0), dy).to(X.device)
            #     mask_ = torch.cat((mask_y, mask), dim=1).bool()
            X_new, _ = attention_layer(X, X, X, key_padding_mask=~mask)
            # if y is not None:
            #     X_new = X_new[:, dy:]
            #     X = X[:, dy:]
            X = X + X_new
            if self.normalization == 'batch_norm':
                X = self.node_norm(X.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                X = self.node_norm(X)
                # A = self.edge_norm(A)

        X = self.linear_out_x(X)
        A = self.linear_out_e(A)
        A = (A + A.transpose(1, 2)) * 0.5
        return X, A

    def conditional_num_nodes(self, conditional_features):
        return self.n_nodes_pred(conditional_features)


class DenseGNN2(nn.Module):
    def __init__(self, config, nnf_in, nef_in, nnf_out, nef_out, nhf, norm_out=False,
                 conditional=False):
        super().__init__()
        self.sizes = (nnf_in, nef_in, nnf_out, nef_out, nhf)

        normalization = config.model.normalization
        mlp_n_layers = config.model.mlp_n_layers
        mlp_hidden_size = nhf
        fact = config.model.edge_node_ratio

        self.linear_in_x = nn.Sequential(nn.Linear(nnf_in, nhf), nn.ReLU(), nn.Linear(nhf, nhf))
        self.linear_in_e = nn.Sequential(nn.Linear(nef_in, int(nhf / fact)), nn.ReLU(),
                                         nn.Linear(int(nhf / fact),int(nhf / fact)))
        self.linear_out_x = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nnf_out))
        self.linear_out_e = nn.Sequential(nn.Linear(int(nhf / fact), int(nhf / fact)),
                                          nn.ReLU(), nn.Linear(int(nhf / fact), nef_out))

        n_layers = config.model.n_layers
        layers = []
        ylayers = []
        attention_layers = []

        for layer in range(n_layers):
            layers.append(DenseLayer2(nhf, nhf, nhf, nhf, mlp_n_layers, mlp_hidden_size, fact,
                                   normalization=normalization))
            ylayers.append(nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nhf)))
            attention_layers.append(torch.nn.MultiheadAttention(nhf, 16, batch_first=True))

        self.linear_in_e = nn.Linear(nef_in, int(nhf / fact))
        self.linear_out_e = nn.Linear(int(nhf / fact), nef_out)
        self.layers = nn.Sequential(*layers)
        self.ylayers = nn.Sequential(*ylayers)
        self.attention_layers = nn.Sequential(*attention_layers)

        self.norm_out = norm_out
        if norm_out:
            if normalization == 'batch_norm':
                self.edge_norm = nn.BatchNorm1d(nef_out)
                self.node_norm = nn.BatchNorm1d(nhf)

            elif normalization == 'layer_norm':
                self.edge_norm = nn.LayerNorm(nef_out)
                self.node_norm = nn.LayerNorm(nhf)

            elif normalization == 'graph_norm':
                self.edge_norm = GraphNorm(nef_out)
                self.node_norm = GraphNorm(nhf)

        if conditional:
            self.n_nodes_pred = Mlp(int(3), config.data.max_num_nodes+1, [nhf, nhf, nhf])

    def forward(self, X_in, A_in, mask):
        X = self.linear_in_x(X_in)
        A = self.linear_in_e(A_in)

        for layer, ylayer, attention_layer in zip(self.layers, self.ylayers, self.attention_layers):
            Y = X.mean(dim=1, keepdim=True)
            Y = ylayer(Y).repeat(1, X.shape[1], 1)
            X = X + Y
            # X[..., -X_in.size(-1):] += X_in
            # A[..., -A_in.size(-1):] += A_in
            # X_new, A_new = layer(X, A, mask, skip_connection=True)
            X, A = layer(X, A, mask, skip_connection=True)
            # X = X + X_new
            # A = A + A_new
            # X[..., -X_in.size(-1):] += X_in
            X_new, _ = attention_layer(X, X, X, key_padding_mask=~mask)
            #X_new, _ = attention_layer(X, X, X, key_padding_mask=~mask)
            # X = X + X_new
            X = self.node_norm(X)


        X = self.linear_out_x(X)
        A = self.linear_out_e(A)
        A = (A + A.transpose(1, 2)) * 0.5
        return X, A

class DenseGNN3(nn.Module):
    def __init__(self, config, nnf_in, nef_in, nnf_out, nef_out, nhf, norm_out=False,
                 conditional=False, graph_predictor=False):
        super().__init__()
        self.sizes = (nnf_in, nef_in, nnf_out, nef_out, nhf)

        normalization = config.model.normalization
        mlp_n_layers = config.model.mlp_n_layers
        mlp_hidden_size = nhf
        fact = config.model.edge_node_ratio
        self.graph_predictor = graph_predictor

        self.linear_in_x = nn.Sequential(nn.Linear(nnf_in, nhf), nn.ReLU(), nn.Linear(nhf, nhf))
        self.linear_in_e = nn.Sequential(nn.Linear(nef_in, int(nhf / fact)), nn.ReLU(),
                                         nn.Linear(int(nhf / fact),int(nhf / fact)))
        if self.graph_predictor:
            self.linear_out_x = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nhf))
        else:
            self.linear_out_x = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, 1))
        self.linear_out_e = nn.Sequential(nn.Linear(int(nhf / fact), int(nhf / fact)),
                                          nn.ReLU(), nn.Linear(int(nhf / fact), nef_out))

        self.linear_out = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nnf_out))

        n_layers = config.model.n_layers
        layers = []
        ylayers = []
        attention_layers = []

        for layer in range(n_layers):
            layers.append(DenseLayer2(nhf, nhf, nhf, nhf, mlp_n_layers, mlp_hidden_size, fact,
                                   normalization=normalization))
            ylayers.append(nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nhf)))
            attention_layers.append(torch.nn.MultiheadAttention(nhf, 16, batch_first=True))

        self.linear_in_e = nn.Linear(nef_in, int(nhf / fact))
        self.linear_out_e = nn.Linear(int(nhf / fact), nef_out)
        self.layers = nn.Sequential(*layers)
        self.ylayers = nn.Sequential(*ylayers)
        self.attention_layers = nn.Sequential(*attention_layers)

        self.norm_out = norm_out
        self.normalization = normalization
        if norm_out:
            if normalization == 'batch_norm':
                self.edge_norm = nn.BatchNorm1d(nef_out)
                self.node_norm = nn.BatchNorm1d(nhf)

            elif normalization == 'layer_norm':
                self.edge_norm = nn.LayerNorm(nef_out)
                self.node_norm = nn.LayerNorm(nhf)

            elif normalization == 'graph_norm':
                # self.edge_norm = GraphNorm(nef_out)
                # self.node_norm = GraphNorm(nhf)
                self.edge_norm = DenseGraphNorm(nef_out)
                self.node_norm = DenseGraphNorm(nhf)

        if conditional:
            self.n_nodes_pred = Mlp(int(3), config.data.max_num_nodes+1, [nhf, nhf, nhf])

    def forward(self, X_in, A_in, mask):
        X = self.linear_in_x(X_in)
        A = self.linear_in_e(A_in)

        for layer, ylayer, attention_layer in zip(self.layers, self.ylayers, self.attention_layers):
            #Y = X.mean(dim=1, keepdim=True)
            #Y = ylayer(Y).repeat(1, X.shape[1], 1)
            #X = X + Y
            # X[..., -X_in.size(-1):] += X_in
            # A[..., -A_in.size(-1):] += A_in
            X_new, A_new = layer(X, A, mask, skip_connection=True)
            #X, A = layer(X, A, mask, skip_connection=True)
            X = X + X_new
            A = A + A_new
            # X[..., -X_in.size(-1):] += X_in
            X_new, _ = attention_layer(X, X, X, key_padding_mask=~mask)
            # X_new, _ = attention_layer(X, X, X, key_padding_mask=~mask)
            # X = X + X_new
            # if self.normalization == 'batch_norm':
            #     X = self.node_norm(X.permute(0, 2, 1)).permute(0, 2, 1)
            # else:
            #     X = self.node_norm(X)

        X = self.linear_out_x(X)
        if self.graph_predictor:
            X = X.sum(1)
            X = self.linear_out(X)
        A = self.linear_out_e(A)
        A = (A + A.transpose(1, 2)) * 0.5
        return X, A

class NodePredictor(nn.Module):
    def __init__(self, nf_in, nf_out, hiddens, n_gaussian):
        super().__init__()
        # self.mlp = Mlp(nf_in, 3 * n_gaussian, hiddens)
        self.mlp = Mlp(nf_in, nf_out, hiddens)
        self.n_bins = nf_out
        self.n_gaussian = n_gaussian

    def forward(self, x):
        x = self.mlp(x)
        # mu = x[..., :self.n_gaussian]
        # sigma = torch.exp(x[..., self.n_gaussian:2 * self.n_gaussian])
        # lambd_ = x[..., 2 * self.n_gaussian:].softmax(dim=-1)
        # pmf = self.pmf(mu, sigma, lambd_, self.n_bins)
        # print(pmf[0])
        # return
        return x

    def pmf(self, mu, sigma, lambd_, num_bins):
        """
        Calculate the binned probability mass function (PMF) using a mixture of Gaussians.

        Args:
            mu (Tensor): Tensor of shape (batch_size, n_components) representing means of Gaussian components.
            sigma (Tensor): Tensor of shape (batch_size, n_components) representing standard deviations.
            lambd_ (Tensor): Tensor of shape (batch_size, n_components) representing mixture weights.
            num_bins (int): Number of bins for PMF calculation.
            bin_range (tuple): Range of values to consider for the PMF.

        Returns:
            Tensor: PMF of shape (batch_size, num_bins).
        """
        # Create bin edges
        bin_range = (-5, 5)
        bin_edges = torch.linspace(bin_range[0], bin_range[1], num_bins + 1, device=mu.device)

        # Expand dimensions for broadcasting
        bin_edges = bin_edges.view(1, 1, num_bins + 1)  # Shape: (1, 1, num_bins + 1)
        mu = mu.unsqueeze(-1)  # Shape: (batch_size, n_components, 1)
        sigma = sigma.unsqueeze(-1)  # Shape: (batch_size, n_components, 1)
        lambd_ = lambd_.unsqueeze(-1)  # Shape: (batch_size, n_components, 1)

        # Compute CDF values at bin edges using the Gaussian CDF formula
        cdf = 0.5 * (1 + torch.erf((bin_edges - mu) / (sigma * torch.sqrt(torch.tensor(2.0, device=mu.device)))))

        # Compute bin probabilities as differences between consecutive CDF values
        bin_probs = cdf[..., 1:] - cdf[..., :-1]  # Shape: (batch_size, n_components, num_bins)

        # Convert mixture weights to log-space
        log_lambd_ = torch.log(lambd_)  # Small epsilon to prevent log(0)

        # Compute log-probabilities safely using log-sum-exp trick
        log_bin_probs = torch.log(bin_probs + 1e-18)  # Prevent log(0) issues

        # Sum in log-space: log(sum(exp(log_w + log_p)))
        log_weighted_pmf = log_lambd_ + log_bin_probs  # Shape: (batch_size, n_components, num_bins)

        # Use log-sum-exp to aggregate over components in a numerically stable way
        log_pmf = torch.logsumexp(log_weighted_pmf, dim=1)  # Shape: (batch_size, num_bins)

        return log_pmf

        return pmf