import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from torch.distributions.normal import Normal

from regression.models.attention import MultiHeadAttn
from regression.models.modules import build_mlp
from regression.models.spin import Spin
from regression.utils.misc import forward_plot_func


class IPNP(nn.Module):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 data_emb_dim=256,
                 data_emb_depth=2,
                 use_H_A=False,
                 H_A_dim=64,
                 latent_dim_mult=1,
                 num_induce=16,
                 num_heads=8,
                 num_spin_blocks=8,
                 dec_dim=128,
                 bound_std=False):

        super().__init__()
        # Context embedding
        self.context_emb = build_mlp(dim_in=(dim_x+dim_y),
                                     dim_hid=data_emb_dim,
                                     dim_out=data_emb_dim,
                                     depth=data_emb_depth)

        # Deterministic path
        self.denc = Spin(data_dim=data_emb_dim,
                         use_H_A=use_H_A,
                         H_A_dim=H_A_dim,
                         latent_dim_mult=latent_dim_mult,
                         num_induce=num_induce,
                         num_heads=num_heads,
                         num_spin_blocks=num_spin_blocks)
        # Target-context Cross attention
        self.target_xattn = MultiHeadAttn(dim_q=dim_x,
                                          dim_k=H_A_dim if use_H_A else data_emb_dim,
                                          dim_v=H_A_dim if use_H_A else data_emb_dim,
                                          dim_out=H_A_dim if use_H_A else data_emb_dim,
                                          num_heads=num_heads)

        # Decoder
        self.bound_std = bound_std
        self.predictor = nn.Sequential(
            nn.Linear(H_A_dim if use_H_A else data_emb_dim, dec_dim),
            nn.ReLU(),
            nn.Linear(dec_dim, dim_y * 2)
        )

    def forward(self, batch, reduce_ll=True, plot_func=False):
        context = torch.cat([batch.xc, batch.yc], -1)
        context_embed = self.context_emb(context)
        encoded = self.denc(context=context_embed)
        query_attn = self.target_xattn(q=batch.x, k=encoded, v=encoded, mask=None, permute_dims=False)
        num_tar = batch.yt.shape[1]
        decoded = self.predictor(query_attn)
        mean, std = torch.chunk(decoded[:, -num_tar:], 2, dim=-1)
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
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2])).to(batch.yt.device)

        encoded = self.denc(batch)
        query_attn = self.target_xattn(q=batch.x, k=encoded, v=encoded, mask=None, permute_dims=False)
        num_tar = batch.yt.shape[1]
        decoded = self.predictor(query_attn)[:, -num_tar:]
        return decoded
