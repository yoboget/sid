class NodePredictor(nn.Module):
    def __init__(self, nf_in, nf_out, hiddens, n_gaussian, norm_out=False,
                 conditional=False):
        super().__init__()
        self.mlp = Mlp(nf_in, 3 * n_gaussian, hiddens)
        self.n_bins = nf_out
        self.n_gaussian = n_gaussian

    def forward(self, x):
        device = x.device
        x = self.mlp(x)
        mu = x[..., :self.n_gaussian]
        sigma = torch.exp(x[..., self.n_gaussian:2 * self.n_gaussian])
        lambd_ = torch.softmax(x[..., 2 * self.n_gaussian:], dim=-1)
        return self.pmf(mu, sigma, lambd_)

    def pmf(self, mu, sigma, lambd_):
        """
        Calculate the binned probability mass function (PMF).

        Args:
            mu: Tensor of means for each Gaussian component.
            sigma: Tensor of standard deviations for each Gaussian component.
            lambd_: Tensor of weights for each Gaussian component.

        Returns:
            Tensor of binned probabilities (PMF).
        """
        # Create the bins
        bins = torch.arange(0, self.n_bins + 1, device=self.device).float()
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Expand dimensions for broadcasting
        bin_centers = bin_centers.unsqueeze(-1)  # [n_bins, 1]
        mu = mu.unsqueeze(0)  # [1, n_gaussian]
        sigma = sigma.unsqueeze(0)  # [1, n_gaussian]
        lambd_ = lambd_.unsqueeze(0)

        # Compute the PDF for each component
        normal_dist = Normal(mu, sigma)
        pdf_values = torch.exp(normal_dist.log_prob(bin_centers))  # [n_bins, n_gaussian]

        # Weight the PDFs and sum across components
        weighted_pdfs = pdf_values * lambd_  # [n_bins, n_gaussian]
        mixture_pdf = weighted_pdfs.sum(dim=-1)  # [n_bins]

        # Calculate the CDF at bin edges.
        normal_dist = Normal(mu, sigma)
        cdf_upper = normal_dist.cdf(bins[1:].unsqueeze(0).unsqueeze(0).unsqueeze(0))  # [1, 1, n_bins, 1]
        cdf_lower = normal_dist.cdf(bins[:-1].unsqueeze(0).unsqueeze(0).unsqueeze(0))  # [1, 1, n_bins, 1]
        cdf_diff = cdf_upper - cdf_lower  # [batch, n, n_bins, n_gaussian]

        weighted_cdf_diff = cdf_diff * lambd_
        pmf = weighted_cdf_diff.sum(dim=-1)  # [batch, n, n_bins]
        return pmf