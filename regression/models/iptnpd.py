import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from attrdict import AttrDict

from regression.models.iptnp import IPTNP
from regression.utils.misc import forward_plot_func


class IPTNPD(IPTNP):
    def __init__(
            self,
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            num_layers,
            dropout,
            bound_std,
            num_induce=16,
            latent_dim_mult=1,
            use_H_A=False,
            H_A_dim=1,
            num_spin_heads=8,
            num_spin_blocks=1,
    ):
        super(IPTNPD, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            num_layers,
            dropout,
            bound_std,
            num_induce,
            latent_dim_mult,
            use_H_A,
            H_A_dim,
            num_spin_heads,
            num_spin_blocks
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    def forward(self, batch, reduce_ll=True, plot_func=False):
        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        pred_tar = Normal(mean, std)
        if plot_func:
            forward_plot_func(nt=10, batch=batch, mean=mean, std=std, ll=pred_tar.log_prob(batch.yt).sum(-1))
        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
        outs.loss = -outs.tar_ll

        return outs

    def predict(self, xc, yc, xt):
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2])).to(xt.device)

        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)
