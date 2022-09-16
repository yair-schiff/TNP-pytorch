import torch
import torch.nn as nn

from regression.models.attention import MultiHeadAttn


class XABA(nn.Module):
    def __init__(self, data_dim, latent_dim, num_heads):
        super().__init__()
        self.attr_xattn = MultiHeadAttn(dim_q=latent_dim, dim_k=data_dim, dim_v=data_dim, dim_out=latent_dim,
                                        num_heads=num_heads)

    def forward(self, x, H_A, mask=None):
        return self.attr_xattn(q=H_A, k=x, v=x, mask=mask, permute_dims=True)


class ABLA(nn.Module):
    def __init__(self, latent_dim, latent_dim_mult, num_heads=8):
        super().__init__()
        self.attr_attn = MultiHeadAttn(dim_q=latent_dim, dim_k=latent_dim, dim_v=latent_dim,
                                       dim_out=latent_dim*latent_dim_mult,
                                       num_heads=num_heads)
        self.post_attn = nn.Linear(latent_dim*latent_dim_mult, latent_dim)

    def forward(self, H):
        attn = self.attr_attn(q=H, k=H, v=H, mask=None, permute_dims=True)
        return self.post_attn(attn)


class XABD(nn.Module):
    def __init__(self, latent_dim, latent_dim_mult, num_heads=8):
        super().__init__()
        self.data_xattn = MultiHeadAttn(dim_q=latent_dim, dim_k=latent_dim, dim_v=latent_dim,
                                        dim_out=latent_dim*latent_dim_mult,
                                        num_heads=num_heads)
        self.post_attn = nn.Linear(latent_dim*latent_dim_mult, latent_dim)

    @staticmethod
    def build_mask(H_A_shape, current_ctx_size):
        mask = torch.zeros(H_A_shape[0], H_A_shape[1]).fill_(float('-inf'))
        mask[:, :current_ctx_size] = 0.0
        return mask

    def forward(self, H_A, H_D, mask=None):
        attn = self.data_xattn(q=H_D, k=H_A, v=H_A, mask=mask, permute_dims=False)  # bsz x num_induce x latent_dim
        return self.post_attn(attn)


class SpinBlock(nn.Module):
    def __init__(self,
                 use_H_A,
                 data_dim, latent_dim, latent_dim_mult,
                 num_heads
                 ):
        super().__init__()

        self.use_H_A = use_H_A
        if use_H_A:
            # Cross Attn Between Attributes
            self.xaba = XABA(data_dim=data_dim, latent_dim=latent_dim,
                             num_heads=num_heads)

            # (Self) Attn Between Latent Attributes - attribute latents
            self.abla = ABLA(latent_dim=latent_dim, num_heads=num_heads)

        # Cross Attn Between Datapoints
        self.xabd = XABD(latent_dim=latent_dim, latent_dim_mult=latent_dim_mult, num_heads=num_heads)

        # (Self) Attn Between Latent Attributes - induced latents
        self.abla_induce = ABLA(latent_dim=latent_dim, latent_dim_mult=latent_dim_mult, num_heads=num_heads)

    def forward(self, x, H_A, H_D, mask=None):
        if self.use_H_A:
            H_A_prime = self.xaba(x=x, H_A=H_A, mask=mask)
            H_A = self.abla(H=H_A_prime)
        H_D_prime = self.xabd(H_A=H_A if self.use_H_A else x, H_D=H_D, mask=mask)
        H_D = self.abla_induce(H=H_D_prime)
        return H_A, H_D


class Spin(nn.Module):
    def __init__(self,
                 data_dim=256,
                 latent_dim_mult=1,
                 use_H_A=False,
                 H_A_dim=1,
                 num_induce=16,
                 num_heads=8,
                 num_spin_blocks=8):
        super().__init__()
        self.use_H_A = use_H_A
        if use_H_A:
            self.H_A_proj = nn.Linear(data_dim, H_A_dim)
        self.H_D = self.init_h_d(num_inds=num_induce, latent_dim=H_A_dim if use_H_A else data_dim)
        self.spin_blocks = nn.ModuleList([
            SpinBlock(
                use_H_A=use_H_A,
                data_dim=data_dim, latent_dim=H_A_dim if use_H_A else data_dim, latent_dim_mult=latent_dim_mult,
                num_heads=num_heads,
            )
            for _ in range(num_spin_blocks)
        ])

    @staticmethod
    def init_h_d(num_inds, latent_dim):
        induced = nn.Parameter(torch.Tensor(1, num_inds, latent_dim))
        nn.init.xavier_uniform_(induced)
        return induced

    def forward(self, context, mask=None):
        H_A = self.H_A_proj(context) if self.use_H_A else None
        H_D = self.H_D.repeat(context.shape[0], 1, 1)
        for sb in self.spin_blocks:
            H_A, H_D = sb(x=context, H_A=H_A, H_D=H_D, mask=mask)
        return H_D
