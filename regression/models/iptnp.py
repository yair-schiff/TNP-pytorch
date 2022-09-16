import torch
import torch.nn as nn

from regression.models.modules import build_mlp
from regression.models.spin import SpinBlock


class IPTNP(nn.Module):
    def __init__(self,
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
        super(IPTNP, self).__init__()#dim_x=dim_x, dim_y=dim_y,
                                    # d_model=d_model, emb_depth=emb_depth, dim_feedforward=dim_feedforward,
                                    # nhead=nhead, num_layers=num_layers,
                                    # dropout=dropout, bound_std=bound_std)
        self.num_induce = num_induce
        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        # self.spin = Spin(data_dim=d_model, latent_dim_mult=latent_dim_mult,
        #                  use_H_A=use_H_A, H_A_dim=H_A_dim,
        #                  num_induce=num_induce,
        #                  num_heads=num_spin_heads,
        #                  num_spin_blocks=num_spin_blocks)

        # self.spin = [SpinBlock(
        #     use_H_A=use_H_A,
        #     data_dim=d_model, latent_dim=H_A_dim if use_H_A else d_model, latent_dim_mult=latent_dim_mult,
        #     num_heads=num_spin_heads,
        # )]
        self.use_H_A = use_H_A
        if use_H_A:
            self.H_A_proj = nn.Linear(d_model, H_A_dim)
        self.H_D = self.init_h_d(num_inds=num_induce, latent_dim=H_A_dim if use_H_A else d_model)
        self.spin = nn.ModuleList([
            SpinBlock(
                use_H_A=use_H_A,
                data_dim=d_model, latent_dim=H_A_dim if use_H_A else d_model, latent_dim_mult=latent_dim_mult,
                num_heads=num_spin_heads,
            )
            for _ in range(num_layers)
        ])
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        # encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.bound_std = bound_std

    @staticmethod
    def init_h_d(num_inds, latent_dim):
        induced = nn.Parameter(torch.Tensor(1, num_inds, latent_dim))
        nn.init.xavier_uniform_(induced)
        return induced

    def construct_input(self, batch, autoreg=False):
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt, torch.zeros_like(batch.yt)), dim=-1)
        tar_start_idx = batch.xc.shape[1]
        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            tar_start_idx += batch.xt.shape[1]
            if self.training and self.bound_std:
                yt_noise = batch.yt + 0.05 * torch.randn_like(batch.yt)  # add noise to the past to smooth the model
                x_y_tar = torch.cat((batch.xt, yt_noise), dim=-1)
            else:
                x_y_tar = torch.cat((batch.xt, batch.yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp, tar_start_idx

    @staticmethod
    def create_spin_mask(batch, autoreg=False):
        bsz = batch.xc.shape[0]
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        if not autoreg:
            mask = torch.zeros(bsz, num_all).fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0
        else:
            # TODO: This is might be wrong
            mask = torch.zeros((bsz, num_all+num_tar)).fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0  # all points attend to context points
            # each real target point attends to itself and preceding real target points
            mask[:, num_ctx:num_all].triu_(diagonal=1)
            # each fake target point attends to preceding real target points
            mask[:, num_ctx:num_all].triu_(diagonal=0)
        return mask.to(batch.xc.device)

    def create_trans_mask(self, batch, autoreg=False):
        num_ctx = self.num_induce
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        if not autoreg:
            mask = torch.zeros(num_all, num_all).fill_(float('-inf'))
            # mask = torch.zeros(bsz, num_all).fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar)).fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0  # all points attend to context points
            # each real target point attends to itself and preceding real target points
            mask[:, num_ctx:num_all].triu_(diagonal=1)
            # each fake target point attends to preceding real target points
            mask[:, num_ctx:num_all].triu_(diagonal=0)
        return mask.to(batch.xc.device), num_tar

    def encode(self, batch, autoreg=False):
        inp, tar_start_idx = self.construct_input(batch, autoreg)
        embedding = self.embedder(inp)
        spin_mask = self.create_spin_mask(batch, autoreg)
        H_D = self.H_D.repeat(batch.xc.shape[0], 1, 1)
        H_A = self.H_A_proj(embedding) if self.use_H_A else None
        trans_mask, num_tar = self.create_trans_mask(batch, autoreg)

        # First spin-transformer layer
        H_A, H_D = self.spin[0](x=embedding, H_A=H_A, H_D=H_D, mask=spin_mask)
        inp_for_encode = torch.cat((H_D, embedding[:, tar_start_idx:]), dim=1)  # [:, tar_start_idx:, :]), dim=1)
        out = self.encoder[0](inp_for_encode, trans_mask)

        # Rest of spin-transformer layers
        for s, t in zip(self.spin[1:], self.encoder[1:]):
            H_A, H_D = s(x=embedding, H_A=H_A, H_D=H_D, mask=spin_mask)
            inp_for_encode = torch.cat((H_D, out[:, -num_tar:]), dim=1)  # [:, tar_start_idx:, :]), dim=1)
            out = t(inp_for_encode, trans_mask)

        # induce = self.spin[0](embeddings, mask=spin_mask)
        # inp_for_encode = torch.cat((H_D, embedding[:, tar_start_idx:]), dim=1)  # [:, tar_start_idx:, :]), dim=1)
        # out = self.encoder(inp_for_encode, mask=trans_mask)
        return out[:, -num_tar:]
