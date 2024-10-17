import torch
from einops import repeat
import sys
import os
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from diffusers import AutoencoderKL
from wandb_utils import array2grid_pixel
import wandb


def out2img(samples):
    return torch.clamp(127.5 * samples + 128.0, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )


def vq_get_sample_size(bs, args):
    if args.input_tensor_type == "bt":
        return (bs, args.tokenizer.token_len)
    elif args.input_tensor_type == "bwh":
        return (bs, args.tokenizer.latent_size, args.tokenizer.latent_size)
    elif args.input_tensor_type == "bcwh":
        return (
            bs,
            args.tokenizer.in_channels,
            args.tokenizer.latent_size,
            args.tokenizer.latent_size,
        )
    elif args.input_tensor_type == "btwh":
        assert args.data.video_frames > 0, "video_frames must be > 0"
        return (
            bs,
            args.data.video_frames,
            args.tokenizer.latent_size,
            args.tokenizer.latent_size,
        )
    else:
        raise ValueError(f"Unknown tensor type: {args.input_tensor_type}")


def calculate_top_k_accuracy(logits, targets, target_mask, k=10):
    # Usage:
    # logits: shape (batch_size, sequence_length, vocab_size)
    # targets: shape (batch_size, sequence_length)
    # target_mask: shape (batch_size, sequence_length)
    # Get the top k predictions
    _, top_k_predictions = torch.topk(logits, k, dim=-1)

    # Create a boolean tensor indicating if the true label is in the top k predictions
    matches = torch.eq(
        top_k_predictions, targets.unsqueeze(-1).expand_as(top_k_predictions)
    ).any(dim=-1)

    # Calculate accuracy only for masked positions
    acc = (matches * target_mask).sum().float() / (target_mask.sum() + 1e-7)

    return acc


def vq_get_vae(args, device):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # stabilityai/sd-vae-ft-ema
    vae.eval()
    return vae


def vq_get_dynamic(args, device, is_train=True):
    if args.dynamic == "discretefm":
        from dynamics.dynamic_discretefm import (
            DiscreteFM,
            Ucoupling,
            Ccoupling,
            CubicScheduler,
            SimpleSampler,
            CorrectorSampler,
            LinearScheduler,
            CosineScheduler,
            QuadraticScheduler,
        )

        input_tensor_type = args.input_tensor_type
        _scheduler = args.discretefm.scheduler
        _coupling = args.discretefm.coupling
        _sampler_name = args.discretefm.sampler
        _adaptive_stepsize = args.discretefm.adaptive_stepsize
        smoothing_factor = args.discretefm.smooth
        if _scheduler == "cubic":
            _kappa = CubicScheduler(args.discretefm.cubic_a, args.discretefm.cubic_b)
        elif _scheduler == "linear":
            _kappa = LinearScheduler()
        elif _scheduler == "cosine":
            _kappa = CosineScheduler()
        elif _scheduler == "quadratic":
            _kappa = QuadraticScheduler()
        else:
            raise ValueError(f"scheduler={_scheduler} not supported")
        if _coupling == "ucoupling":
            _coupling = Ucoupling(mask_token_id=args.tokenizer.mask_token_id)
        elif _coupling == "ccoupling":
            _coupling = Ccoupling(
                mask_token_id=args.tokenizer.mask_token_id,
                msk_prop=args.discretefm.ccoupling_prob,
            )
        else:
            raise ValueError(f"coupling={_coupling} not supported")

        discretefm = DiscreteFM(
            vocab_size=args.tokenizer.vocab_size,
            coupling=_coupling,
            kappa=_kappa,
            device=device,
            input_tensor_type=input_tensor_type,
            smoothing_factor=smoothing_factor,
        )
        training_losses_fn = discretefm.training_losses

        if _sampler_name == "simple":
            sampler = SimpleSampler(
                mask_token_id=args.tokenizer.mask_token_id,
                input_tensor_type=input_tensor_type,
                adaptive_stepsize=_adaptive_stepsize,
            )
        elif _sampler_name == "corrector":
            sampler = CorrectorSampler(
                mask_token_id=args.tokenizer.mask_token_id,
                input_tensor_type=input_tensor_type,
                adaptive_stepsize=_adaptive_stepsize,
                alpha=args.discretefm.corrector.alpha,
                a=args.discretefm.corrector.a,
                b=args.discretefm.corrector.b,
            )
        else:
            raise ValueError(f"sampler name={_sampler_name} not supported")

        def sample_fn(sample_size, model, **model_kwargs):
            r = sampler(
                sample_size,
                discretefm,
                model=model,
                n_steps=args.discretefm.step_num,
                t_min=args.discretefm.sample_t_min,
                **model_kwargs,
            )
            r = torch.stack(r, dim=0).to(device)
            return r

    elif args.dynamic in ["campbell", "campbell_d3pm"]:
        from dynamics.dynamic_discretefm_campbell import DynamicDiscreteFMCampbell

        if args.dynamic in ["campbell"]:
            step_num = args.campbell.step_num
        elif args.dynamic == "campbell_d3pm":
            step_num = args.campbell_d3pm.step_num
        else:
            raise ValueError(f"dynamic={args.dynamic} not supported")

        if "imagenet" in args.data.name and is_train:
            assert (
                step_num <= 250
            ), f"step_num={step_num} for imagenet should <= 250, otherwise too slow"

        model_type = "flow" if args.dynamic == "campbell" else "d3pm"
        campbell_fm = DynamicDiscreteFMCampbell(
            model_type=model_type,
            token_len=args.tokenizer.token_len,
            vocab_size=args.tokenizer.vocab_size,
            mask_token_id=args.tokenizer.mask_token_id,
            device=device,
        )
        training_losses_fn = campbell_fm.training_losses

        def sample_fn(sample_size, model, **model_kwargs):
            r = campbell_fm.sample(
                sample_size=sample_size,
                model=model,
                num_steps=step_num,
                **model_kwargs,
            )
            if len(r.shape) == 2:
                r = repeat(r, "b t -> 7 b t").to(device).to(torch.int64)
            elif len(r.shape) == 3:
                r = repeat(r, "b w h -> 7 b w h").to(device).to(torch.int64)
            elif len(r.shape) == 4:
                r = repeat(r, "b c w h -> 7 b c w h").to(device).to(torch.int64)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(r.shape)}")
            return r

    elif args.dynamic in ["campbell_uni"]:
        from dynamics.dynamic_discretefm_campbell_uni import (
            DynamicDiscreteFMCampbell_UNI,
        )

        step_num = args.campbell.step_num
        if "imagenet" in args.data.name and is_train:
            assert (
                step_num <= 250
            ), f"step_num={step_num} for imagenet should <= 250, otherwise too slow"

        campbell_fm = DynamicDiscreteFMCampbell_UNI(
            token_len=args.tokenizer.token_len,
            vocab_size=args.tokenizer.vocab_size,
            mask_token_id=args.tokenizer.mask_token_id,
            num_classes=args.data.num_classes,
            mask_token_reindex=args.tokenizer.mask_token_reindex,
            device=device,
            weight_y=args.uni.weight_y,
            second_modal_type=args.model.params.second_modal_type,
        )
        training_losses_fn = campbell_fm.training_losses

        def sample_fn(sample_size, model, return_dict=False, **model_kwargs):
            res = campbell_fm.sample(
                sample_size=sample_size,
                model=model,
                num_steps=step_num,
                **model_kwargs,
            )

            assert isinstance(res, dict)
            if return_dict:
                return res
            r = res["x"]
            if len(r.shape) == 2:
                r = repeat(r, "b t -> 7 b t").to(device).to(torch.int64)
            elif len(r.shape) == 3:
                r = repeat(r, "b w h -> 7 b w h").to(device).to(torch.int64)
            elif len(r.shape) == 4:
                r = repeat(r, "b c w h -> 7 b c w h").to(device).to(torch.int64)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(r.shape)}")
            return r

    elif args.dynamic in ["campbell_uni_v2"]:
        from dynamics.dynamic_campbell_uni_v2 import (
            DynamicDiscreteFMCampbell_UNI,
        )

        step_num = args.campbell.step_num
        if "imagenet" in args.data.name and is_train:
            assert (
                step_num <= 250
            ), f"step_num={step_num} for imagenet should <= 250, otherwise too slow"

        campbell_fm = DynamicDiscreteFMCampbell_UNI(
            token_len=args.tokenizer.token_len,
            vocab_size=args.tokenizer.vocab_size,
            mask_token_id=args.tokenizer.mask_token_id,
            num_classes=args.data.num_classes,
            mask_token_reindex=args.tokenizer.mask_token_reindex,
            device=device,
            weight_y=args.uni.weight_y,
            second_modal_type=args.model.params.second_modal_type,
        )
        training_losses_fn = campbell_fm.training_losses

        def sample_fn(sample_size, model, return_dict=False, **model_kwargs):
            res = campbell_fm.sample(
                sample_size=sample_size,
                model=model,
                num_steps=step_num,
                **model_kwargs,
            )

            assert isinstance(res, dict)
            if return_dict:
                return res
            r = res["x"]
            if len(r.shape) == 2:
                r = repeat(r, "b t -> 7 b t").to(device).to(torch.int64)
            elif len(r.shape) == 3:
                r = repeat(r, "b w h -> 7 b w h").to(device).to(torch.int64)
            elif len(r.shape) == 4:
                r = repeat(r, "b c w h -> 7 b c w h").to(device).to(torch.int64)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(r.shape)}")
            return r

    elif args.dynamic == "d3pm":
        from dynamics.dynamic_d3pm import D3PM

        d3pm = D3PM(
            n_T=args.d3pm.step_num,
            num_classes=args.tokenizer.vocab_size,
            mask_token_id=args.tokenizer.mask_token_id,
        )
        training_losses_fn = d3pm.training_losses
        assert (
            not "imagenet" in args.data.name
        ), "imagenet not supported, as sampling with 1k steps for 5k images is too slow, 7 hours per GPU"

        def sample_fn(sample_size, model, **model_kwargs):
            r = d3pm.sample(sample_size=sample_size, model=model, **model_kwargs)
            if len(r.shape) == 2:
                r = repeat(r, "b t -> 7 b t").to(device).to(torch.int64)
            elif len(r.shape) == 3:
                r = repeat(r, "b w h -> 7 b w h").to(device).to(torch.int64)
            elif len(r.shape) == 4:
                r = repeat(r, "b c w h -> 7 b c w h").to(device).to(torch.int64)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(r.shape)}")
            return r

    elif args.dynamic == "maskgit":
        from dynamics.dynamic_maskgit.dynamic_maskgit import MaskGit

        _dynamic = MaskGit(args.maskgit, device=device)
        training_losses_fn = _dynamic.training_losses
        sample_config = args.maskgit.sampling

        def sample_fn(sample_size, model, **model_kwargs):
            r = _dynamic.generate(
                model,
                sample_size=sample_size,
                guidance_scale_pow=sample_config.guidance_scale_pow,
                randomize_temperature=sample_config.randomize_temperature,
                softmax_temperature_annealing=sample_config.softmax_temperature_annealing,
                num_sample_steps=sample_config.num_sample_steps,
                guidance_scale=sample_config.guidance_scale,
                guidance_decay=sample_config.guidance_decay,
                **model_kwargs,
            )
            if len(r.shape) == 2:
                r = repeat(r, "b t -> 7 b t").to(device)
            elif len(r.shape) == 3:
                r = repeat(r, "b w h -> 7 b w h").to(device)
            elif len(r.shape) == 4:
                r = repeat(r, "b c w h -> 7 b c w h").to(device)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(r.shape)}")
            return r

    elif args.dynamic == "dynradd":
        raise NotImplementedError("not used any more")
        from dynamics.dynamic_radd.dyn_radd import DynRadd
        from dynamics.dynamic_radd.noise_lib import get_noise

        noise = get_noise(
            args.dynradd.noise_type,
            args.dynradd.noise_sigma_min,
            args.dynradd.noise_sigma_max,
        )
        dyn_radd = DynRadd(
            token_dim=args.tokenizer.vocab_size,
            mask_token_id=args.tokenizer.mask_token_id,
            noise=noise,
            loss_type=args.dynradd.loss_type,
            sampling_predictor=args.dynradd.sampling_predictor,
            device=device,
        )
        training_losses_fn = dyn_radd.training_losses

        def sample_fn(sample_size, model, **model_kwargs):
            r = dyn_radd.sample(
                sampling_steps=args.dynradd.step_num,
                sample_shape=sample_size,
                model=model,
            )
            if len(r.shape) == 2:
                r = repeat(r, "b t -> 7 b t").to(device)
            elif len(r.shape) == 3:
                r = repeat(r, "b w h -> 7 b w h").to(device)
            elif len(r.shape) == 4:
                r = repeat(r, "b c w h -> 7 b c w h").to(device)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(r.shape)}")
            return r

    else:
        raise ValueError(f"dynamic={args.dynamic} not supported")

    return training_losses_fn, sample_fn


def vq_get_encoder_decoder(args, device):
    if args.tokenizer.name == "sd_vq_f8":

        use_id = args.input_tensor_type == "bt"
        vocab_size = args.tokenizer.vocab_size
        latent_size = args.tokenizer.latent_size
        config_path = args.tokenizer.config_path
        ckpt_path = args.tokenizer.ckpt_path

        
        sys.path.insert(0, os.path.abspath("./ldm"))
        from ldm.util import instantiate_from_config

        config = OmegaConf.load(config_path)  # Load the YAML config file
        pl_sd = torch.load(ckpt_path, map_location="cpu")  # Load the checkpoint
        sd = pl_sd["state_dict"]

        # Instantiate the VQ-VAE model from the configuration file
        _tokenizer = instantiate_from_config(config.model)
        _tokenizer.load_state_dict(sd, strict=False)
        _tokenizer.eval()  # Set the model to evaluation mode
        _tokenizer.requires_grad_(False)
        _tokenizer = _tokenizer.to(device)  # Move to device (e.g., GPU)

        @torch.no_grad()
        def tokenizer_encode_fn(img, mini_bs=25):
            img = img / 255.0  # Normalize input images
            img = (img - 0.5) * 2  # Rescale to the range [-1, 1]
            
            img_shape = img.shape
            if len(img_shape) == 5:  # Video data
                b, t, c, h, w = img.shape
                img = rearrange(img, "b t c h w -> (b t) c h w")

            # Process the input batch by mini-batch to avoid memory issues
            for i in range(0, len(img), mini_bs):
                _img = img[i: i + mini_bs]
                encode_res = _tokenizer.encode(_img)
                _indices = encode_res[2][-1]  # Extract the indices (discrete tokens)

                if i == 0:
                    indices = _indices
                else:
                    indices = torch.cat([indices, _indices], dim=0)

            # Reshape back to original format if it was video data
            if len(img_shape) == 5:
                indices = rearrange(
                    indices, "(b t h w) -> b t h w", b=b, t=t, h=latent_size, w=latent_size
                )
            elif len(img_shape) == 4:  # Image data
                indices = rearrange(indices, "(b h w) -> b h w", b=img_shape[0], h=latent_size, w=latent_size)
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(img_shape)}")

            return indices  # Return discrete token IDs

        @torch.no_grad()
        def tokenizer_decode_fn(indices, mini_bs=25):
            # Reindex any masked tokens
            indices[indices == args.tokenizer.mask_token_id] = args.tokenizer.mask_token_reindex
            indices_shape = indices.shape

            if len(indices_shape) == 4:  # Video data
                b, t, h, w = indices.shape
                indices = rearrange(indices, "b t h w -> (b t) (h w)")
            elif len(indices_shape) == 3:  # Image data
                indices = rearrange(indices, "b h w -> b (h w)")
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(indices_shape)}")

            # Process the input batch by mini-batch to avoid memory issues
            for i in range(0, len(indices), mini_bs):
                _indices = indices[i: i + mini_bs]
                _img = _tokenizer.decode_tokens(_indices)  # Decode the tokens back to images

                if i == 0:
                    img = _img
                else:
                    img = torch.cat([img, _img], dim=0)

            if len(indices_shape) == 4:  # Video data
                img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)

            img = img.clamp(-1, 1) 
            img = ((img + 1) * 0.5 * 255.0




def vq_debug(
    args,
    x,
    x_clean,
    loss_dict,
    text_table,
    tokenizer_decode_fn,
    wandb_dict,
    vis_corruption=True,
    vis_num=1,
):

    return wandb_dict
    if (
        not args.dynamic
        in [
            "discretefm",
            "campbell",
            "d3pm",
            "campbell_d3pm",
            "maskgit",
        ]
        or args.data.video_frames > 0
    ):
        return wandb_dict

    x_corrupt = loss_dict["x_corrupt"]
    logits = loss_dict["logits"]
    corrupted_vis = wandb.Image(
        array2grid_pixel(tokenizer_decode_fn(x_corrupt[0:vis_num]))
    )
    clean_vis = wandb.Image(array2grid_pixel(tokenizer_decode_fn(x_clean[0:vis_num])))

    x_recons = torch.argmax(logits, dim=1)
    table_x0_unique = str(list(x.unique().cpu().numpy()))
    table_x_corrupt = str(list(x_corrupt.unique().cpu().numpy()))
    table_x_recons = str(list(x_recons.unique().cpu().numpy()))
    text_table.add_data(table_x0_unique, table_x_corrupt, table_x_recons)
    wandb_dict.update({"text_table": text_table})

    if len(logits.shape) == 3:
        logits = rearrange(logits, "b k t -> b t k")
        x_corrupt = rearrange(x_corrupt, "b t -> b t")
        x_clean = rearrange(x_clean, "b t -> b t")
    elif len(logits.shape) == 4:
        logits = rearrange(logits, "b k h w -> b (h w) k")
        x_corrupt = rearrange(x_corrupt, "b h w -> b (h w)")
        x_clean = rearrange(x_clean, "b h w -> b (h w)")
    elif len(logits.shape) == 5:
        logits = rearrange(logits, "b k c h w -> b (c h w) k")
        x_corrupt = rearrange(x_corrupt, "b c h w -> b (c h w)")
        x_clean = rearrange(x_clean, "b c h w -> b (c h w)")
    else:
        raise ValueError(f"Unknown logits shape: {logits.shape}")
    _, _, logits_c = logits.shape
    target_mask = x_clean != x_corrupt
    corrupt_ratio = target_mask.sum().float() / (x_clean.numel() + 1e-7)
    first_pred_idx = torch.argmax(target_mask[0].float())
    zero_logit = logits[0, first_pred_idx, 0]
    one_logit = logits[0, first_pred_idx, 1]
    two_logit = logits[0, first_pred_idx, 2]

    predictions = torch.argmax(logits, dim=-1)
    samples = torch.multinomial(
        torch.softmax(logits, dim=-1).view(-1, logits_c),
        num_samples=1,
    )[:, 0].view(len(x), -1)

    # calculate accuracy
    # matches = samples == x_clean  # (B, T)
    # acc = (matches * target_mask).sum().float() / (target_mask.sum() + 1e-7)

    acc = calculate_top_k_accuracy(
        logits=logits, targets=x_clean, target_mask=target_mask, k=1
    )
    acc_top10 = calculate_top_k_accuracy(
        logits=logits, targets=x_clean, target_mask=target_mask, k=10
    )
    acc_top100 = calculate_top_k_accuracy(
        logits=logits, targets=x_clean, target_mask=target_mask, k=100
    )

    predictions[~target_mask] = x_corrupt[~target_mask]
    samples[~target_mask] = x_corrupt[~target_mask]
    predictions_vis = predictions[0:vis_num]
    samples_vis = samples[0:vis_num]
    if len(x.shape) == 2:  # B,T
        predictions_vis = rearrange(predictions_vis, "b t -> b t")
        samples_vis = rearrange(samples_vis, "b t -> b t")
    elif len(x.shape) == 3:  # B,W,H
        _b, _w, _h = x.shape
        predictions_vis = rearrange(predictions_vis, "b (w h) -> b w h", w=_w, h=_h)
        samples_vis = rearrange(samples_vis, "b (w h ) -> b w h", w=_w, h=_h)

    elif len(x.shape) == 4:  # B,C,H,W
        _b, _c, _w, _h = x.shape
        predictions_vis = rearrange(
            predictions_vis, "b (c h w) -> b c h w", c=_c, h=_h, w=_w
        )
        samples_vis = rearrange(samples_vis, "b (c h w) -> b c h w", c=_c, h=_h, w=_w)
    else:
        raise ValueError(f"Unknown x shape: {x.shape}")
    predictions_vis = wandb.Image(
        array2grid_pixel(tokenizer_decode_fn(predictions_vis))
    )
    samples_vis = wandb.Image(array2grid_pixel(tokenizer_decode_fn(samples_vis)))

    wandb_dict.update(
        {
            "vis/argmax_recon": predictions_vis,
            "vis/sample_recon": samples_vis,
            "vis/corrupted": corrupted_vis,
            "vis/clean": clean_vis,
            "analysis/acc": acc,
            "analysis/acc_top10": acc_top10,
            "analysis/acc_top100": acc_top100,
            "analysis/corrupt_ratio": corrupt_ratio,
            "analysis/zero_logit": zero_logit,
            "analysis/one_logit": one_logit,
            "analysis/two_logit": two_logit,
        }
    )
    return wandb_dict


def vq_get_generator(args, device, loader, accelerator, train_steps, vae):

    
    def get_data_generator(return_cls_id=True):
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=not accelerator.is_main_process,
                initial=_init,
                desc="data fetching",
            ):

                x = data["image"].to(device)

                try:
                    y = data["cls_id"].to(device)
                except:
                    try:
                        y = data["caption_feat"].to(device)
                    except:
                        y = None
                x = out2img(x)

                if return_cls_id:
                    yield x, y
                else:
                    yield x

    def get_caption_generator():
        while True:
            for data in tqdm(
                loader,
                disable=not accelerator.is_main_process,
                desc="gen caption",
            ):
                captiopn_feat = data["caption_feat"].to(device)
                caption = data["caption"]

                yield captiopn_feat, caption

    def get_indices_generator(return_cls_id=True):
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=not accelerator.is_main_process,
                initial=_init,
                desc="data fetching",
            ):

                x = data["indices"].to(device)

                try:
                    y = data["cls_id"].to(device)
                except:
                    try:
                        y = data["caption_feat"].to(device)
                    except:
                        y = None

                if return_cls_id:
                    yield x, y
                else:
                    yield x

    if "indices" in args.data.name:
        data_gen = get_indices_generator(return_cls_id=True)
        realimg_gen = get_indices_generator(return_cls_id=False)
    else:
        data_gen = get_data_generator(return_cls_id=True)
        realimg_gen = get_data_generator(return_cls_id=False)
    cap_gen = get_caption_generator()
    return data_gen, realimg_gen, cap_gen
