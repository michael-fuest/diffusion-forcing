from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance
)
from algorithms.diffusion_forcing.df_base import DiffusionForcingBase
from algorithms.diffusion_forcing.models.model_discrete_latte import Latte_B_2
from .dynamic_discretefm_v2 import SimpleSampler, DiscreteFM, Ucoupling, CubicScheduler, indices_to_diracp
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from tokenizer.utils_vq import vq_get_encoder_decoder

class DiscreteFMVideo(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.metrics = cfg.metrics
        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None
        self.encode_fn, self.decode_fn = vq_get_encoder_decoder(self.cfg, device=self.device)
        self.mask_token_id = cfg.tokenizer.mask_token_id
        self.sampler = SimpleSampler(mask_token_id=cfg.tokenizer.mask_token_id)
        self.n_tokens = cfg.n_frames // cfg.frame_stack
        
        self.flow_matching = DiscreteFM(
            vocab_size=cfg.tokenizer.vocab_size,
            coupling=Ucoupling(mask_token_id=cfg.tokenizer.mask_token_id),
            kappa=CubicScheduler(),  
            device=self.device,
            input_tensor_type="bwh",
        )

        self.diffusion_model = Latte_B_2(
            vocab_size=cfg.tokenizer.vocab_size, 
            input_size=cfg.tokenizer.latent_size,  
            num_frames=cfg.n_frames,  
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
        xs_pred_unstacked = loss_dict["x_corrupt"]

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred_unstacked,
            "xs": xs,
        }
        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)
        xs = xs.squeeze(2)
        n_frames, batch_size, H, W = xs.shape
        
        scheduling_matrix = self._generate_scheduling_matrix(n_frames) / 1000
        scheduling_matrix = torch.from_numpy(scheduling_matrix).float().to(self.device)
        scheduling_matrix = scheduling_matrix.unsqueeze(1).repeat(1, batch_size, 1)

        masked_x0, dirac_masked_x0 = self.sampler.construct_x0(
            sample_size=(n_frames, batch_size),
            device=self.device,
            vocab_size=self.cfg.tokenizer.vocab_size
        )
        xt = torch.ones((n_frames, batch_size, H, W), device=self.device) * self.mask_token_id

        for step in tqdm(range(scheduling_matrix.shape[0] - 1), desc="Sampling"):
            from_noise_levels = scheduling_matrix[step]
            to_noise_levels = scheduling_matrix[step + 1]

            xt, logits_t = self.sampler.sample_step_with_noise_schedule(
                xt=xt.long(),                        
                from_noise_levels=from_noise_levels,  
                to_noise_levels=to_noise_levels,      
                discretefm=self.flow_matching,        
                model=self.diffusion_model,                  
                n_steps=scheduling_matrix.shape[0] - 1,
            )

        targets = rearrange(xs, "f b h w -> b f h w")

        loss = F.cross_entropy(
            logits_t,  
            targets,  
            ignore_index=self.mask_token_id  
        )
        
        # if masks is not None:
        #     loss = self.reweight_loss(loss, masks)
        
        self.log("val/loss", loss, prog_bar=True)

        output_dict = {
            "loss": loss,
            "xs_pred": xt,
            "xs": xs,
        }
        
        return output_dict

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx, namespace="test")

    def _unstack(self, xs):
        return rearrange(xs, "f b fs ... -> b (f fs) ...", fs=self.frame_stack)

    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return
        xs_pred = []
        xs = []
        for pred, gt in self.validation_step_outputs: 
            pred_decoded = self.decode_fn(pred)
            gt_decoded = self.decode_fn(gt)
            pred_decoded_normalized = (pred_decoded / 255.0) * 2 - 1  
            gt_decoded_normalized = (gt_decoded / 255.0) * 2 - 1 
            xs_pred.append(pred_decoded_normalized)
            xs.append(gt_decoded_normalized)

        xs_pred = torch.cat(xs_pred, 1)
        xs = torch.cat(xs, 1)

        xs_pred = rearrange(xs_pred, "b f ... -> f b ...")
        xs = rearrange(xs, "b f ... -> f b ...")

        if self.logger:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )
        print('finished vid logging')
        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames :],
            xs[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
        )
        print('finished metrics')
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()



