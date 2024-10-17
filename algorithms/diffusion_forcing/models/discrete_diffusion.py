from typing import Optional, Callable
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .unet3d import Unet3D
from .transformer import Transformer
from .utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract, EinopsWrapper

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])

class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        x_shape: torch.Size,
        external_cond_dim: int,
        is_causal: bool,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape
        self.external_cond_dim = external_cond_dim
        self.is_causal = is_causal
        self.vocab_size_total = cfg.vocab_size + cfg.num_classes + 1  # Including mask token

        self._build_model()

    def _build_model(self):
        x_channel = self.x_shape[0]
        if len(self.x_shape) == 3:
            # video
            attn_resolutions = [self.arch.resolution // res for res in list(self.arch.attn_resolutions)]
            self.model = EinopsWrapper(
                from_shape="f b c h w",
                to_shape="b c f h w",
                module=Unet3D(
                    dim=self.arch.network_size,
                    attn_dim_head=self.arch.attn_dim_head,
                    attn_heads=self.arch.attn_heads,
                    dim_mults=self.arch.dim_mults,
                    attn_resolutions=attn_resolutions,
                    use_linear_attn=self.arch.use_linear_attn,
                    channels=x_channel,
                    out_dim=self.vocab_size_total,  # Output logits over the vocabulary
                    external_cond_dim=self.external_cond_dim,
                    is_causal=self.is_causal,
                    use_init_temporal_attn=self.arch.use_init_temporal_attn,
                    time_emb_type=self.arch.time_emb_type,
                ),
            )
        elif len(self.x_shape) == 1:
            self.model = Transformer(
                x_dim=x_channel,
                external_cond_dim=self.external_cond_dim,
                size=self.arch.network_size,
                num_layers=self.arch.num_layers,
                nhead=self.arch.attn_heads,
                dim_feedforward=self.arch.dim_feedforward,
                out_dim=self.vocab_size_total,  # Output logits over the vocabulary
            )
        else:
            raise ValueError(f"unsupported input shape {self.x_shape}")

    def model_predictions(self, x, t, external_cond=None):
        logits = self.model(x, t, external_cond, is_causal=self.is_causal)
        return logits

    def split_logits(self, logits):
        vocab_size_x = self.cfg.vocab_size
        logits_x = logits[..., :vocab_size_x]
        logits_y = logits[..., vocab_size_x:]
        return logits_x, logits_y

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        noise_levels: torch.Tensor,
        targets: Optional[dict] = None,
    ):
        logits = self.model_predictions(x=x, t=noise_levels, external_cond=external_cond)
        return logits  
