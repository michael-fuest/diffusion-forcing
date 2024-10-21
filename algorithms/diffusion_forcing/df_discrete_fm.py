from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from .dynamic_discretefm_v2 import DiscreteFM, Ucoupling, CubicScheduler
from algorithms.diffusion_forcing.df_base import DiffusionForcingBase
from algorithms.diffusion_forcing.models.model_discrete_latte import Latte_B_2

class DiscreteFMVideo(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        self.flow_matching = DiscreteFM(
            vocab_size=cfg.df_discrete_fm.vocab_size,
            coupling=Ucoupling(mask_token_id=cfg.df_discrete_fm.mask_token_id),
            kappa=CubicScheduler(),  # You can choose different schedulers
            device=self.device,
            input_tensor_type="bwh",
        )

        self.diffusion_model = Latte_B_2(
            vocab_size=cfg.df_discrete_fm.vocab_size, 
            input_size=cfg.df_discrete_fm.input_size,  
            num_frames=cfg.df_discrete_fm.n_frames,  
            in_channels=1, 
            use_checkpoint=False,
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        noise_levels = self._generate_noise_levels(xs)

        # Flow matching corruption and training
        loss_dict = self.flow_matching.training_losses(
            model=self.diffusion_model,
            x=xs,
            noise_levels=noise_levels,
        )
        loss = loss_dict["loss"]
        
        # Log loss every 20 steps
        if batch_idx % 20 == 0:
            self.log("training/loss", loss)
        
        xs_unstacked = self._unstack(xs)
        xs_pred_unstacked = self._unstack(loss_dict["x_corrupt"])

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred_unstacked,
            "xs": xs,
        }
        return output_dict

    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        noise_levels = self._generate_noise_levels(xs)
        
        loss_dict = self.flow_matching.training_losses(
            model=self.diffusion_model,
            x=xs,
            noise_levels=noise_levels
        )
        loss = loss_dict["loss"]
        
        xs_unstacked = self._unstack(xs)
        xs_pred_unstacked = self._unstack(loss_dict["x_corrupt"])
        self.validation_step_outputs.append((xs_pred_unstacked.detach().cpu(), xs_unstacked.detach().cpu()))

        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx, namespace="test")

    def _unstack(self, xs):
        return rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

