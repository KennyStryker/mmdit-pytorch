from typing import Optional

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


class Modulation(nn.Module):

    def __init__(self, *, input_dim: int, n_mods: int, **kwargs):
        super().__init__(**kwargs)

        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(input_dim, n_mods * input_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.modulation(x)


class MLP(nn.Module):

    def __init__(self, *, input_dim: int, hidden_size: int, **kwargs):
        super().__init__(**kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class MMDiTBlock(nn.Module):

    def __init__(
        self,
        *,
        dim_txt: int,
        dim_img: int,
        dim_timestep: Optional[int] = None,
        qk_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Modulation layers
        self.txt_modulation = Modulation(n_mods=6, input_dim=dim_txt)
        self.img_modulation = Modulation(n_mods=6, input_dim=dim_img)

        # Layers norms
        self.img_attn_layer_norm = nn.LayerNorm(dim_img)
        self.img_attn_layer_norm = nn.LayerNorm(dim_img)

    def forward(self, c: torch.Tensor, x: torch.Tensor, y: torch.Tensor):

        c_mod = self.txt_modulation(y)
        x_mod = self.img_modulation(y)

        alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c = rearrange(c_mod, 'b (n d) -> b n d', n = 6).chunk(6, dim=1)
        alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x = rearrange(x_mod, 'b (n d) -> b n d', n = 6).chunk(6, dim=1)

        return alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c, alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x

    

if __name__ == "__main__":
    block = MMDiTBlock(dim_txt=512, dim_img=512)

    x = torch.randn(1, 512)
    c = torch.randn(1, 512)
    t = torch.randn(1, 512)

    r = block(c, x, t)

    print(r)