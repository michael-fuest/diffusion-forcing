# Imports
import torch
from torch import nn
from torch.nn import functional as F
import os
from typing import Tuple, List
from einops import rearrange
# from utils_common import print_rank_0


Prob = torch.Tensor
Img = torch.Tensor


def pad_like_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))


def indices_to_diracp(x: Img, vocab_size: int, type_data: str = "bt") -> Prob:
    assert torch.all(x >= 0) and torch.all(x < vocab_size), f"Indices out of bounds: min {x.min().item()}, max {x.max().item()}, vocab_size {vocab_size}"
    if type_data == "bt":
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b t k -> b k t")
    elif type_data == "bwh":
        b, w, h = x.shape
        x = rearrange(x, "b w h -> b (w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (w h) k -> b k w h", w=w, h=h)
    elif type_data == "bcwh":
        b, c, w, h = x.shape
        x = rearrange(x, "b c w h -> b (c w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (c w h) k -> b k c w h", c=c, w=w, h=h)
    else:
        raise ValueError(f"input_tensor_type {type_data} not supported")


def sample_p(pt: Prob, type_data: str) -> Img:
    if type_data == "bt":
        b, k, t = pt.shape
        pt = rearrange(pt, "b k t -> (b t) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, t)
    elif type_data == "bwh":
        b, k, h, w = pt.shape
        pt = rearrange(pt, "b k h w -> (b h w) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, h, w)
    elif type_data == "bcwh":
        b, k, c, h, w = pt.shape
        pt = rearrange(pt, "b  k c h w -> (b c h w) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, c, h, w)
    else:
        raise ValueError(f"input_tensor_type {type_data} not supported")


class KappaScheduler:
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError


class Coupling:
    def __init__(self) -> None:
        pass

    def sample(self, x1: Img) -> tuple[Img, Img]:
        raise NotImplementedError


class Ucoupling(Coupling):
    def __init__(self, mask_token_id) -> None:
        self.mask_token_id = mask_token_id

    def sample(self, x1: Img) -> tuple[Img, Img]:
        return torch.ones_like(x1) * self.mask_token_id, x1


class Ccoupling(Coupling):
    def __init__(self, mask_token_id: int, msk_prop: float = 0.8) -> None:
        if msk_prop is None:
            print("Ccoupling, msk_prop is None, using coupling by random prob")
        elif msk_prop > 0:
            print("Ccoupling, msk_prop: ", msk_prop, "data_prob", 1 - msk_prop)
        else:
            raise ValueError("msk_prop must be non-negative")
        self.mask_token_id = mask_token_id
        self.msk_prob = msk_prop

    def sample(self, x1: Img) -> tuple[Img, Img]:
        if self.msk_prob is None:
            _msk_prob = torch.rand_like(x1.float())
        else:
            _msk_prob = self.msk_prob
        _mask20 = torch.rand_like(x1.float()) > _msk_prob
        _mask_id = torch.ones_like(x1) * self.mask_token_id
        x0 = x1 * _mask20 + _mask_id * (~_mask20)
        return x0, x1


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(
        self, t: float | torch.Tensor
    ) -> float | torch.Tensor:  # Eq 33 in paper
        return (
            -2 * (t**3)
            + 3 * (t**2)
            + self.a * (t**3 - 2 * t**2 + t)
            + self.b * (t**3 - t**2)
        )

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return (
            -6 * (t**2)
            + 6 * t
            + self.a * (3 * t**2 - 4 * t + 1)
            + self.b * (3 * t**2 - 2 * t)
        )


class LinearScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1.0


class QuadraticScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**2

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 2 * t


class RootScheduler(KappaScheduler):
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**0.5

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 0.5 * t ** (-0.5)


class CosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - torch.cos(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t) * self.coeff


class SineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.cos(self.coeff * t) * self.coeff


class ArcCosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - self.coeff * torch.acos(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)


class ArcSinScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff * torch.asin(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)


class DiscreteFM:
    def __init__(
        self,
        vocab_size: int,
        coupling: Coupling,
        kappa: KappaScheduler,
        device: torch.device,
        input_tensor_type: str = "btw",
        smoothing_factor: float = 0.0,
        mask_ce=False,
    ) -> None:
        self.vocab_size = vocab_size
        self.coupling = coupling
        self.kappa = kappa
        self.device = device
        self.type_data = input_tensor_type
        self.smoothing_factor = smoothing_factor
        self.mask_ce = mask_ce
        #print_rank_0(f"smoothing_factor: {smoothing_factor}")

    def forward_u(
        self, t: float | torch.Tensor, xt: Img, model: nn.Module, **model_kwargs
    ) -> Prob:  # Eq 24 and Eq 57

        n_frames, batch_size, h, w = xt.shape
        xt_flat = xt.reshape(-1, h, w)

        dirac_xt = indices_to_diracp(
            xt_flat, self.vocab_size, self.type_data
        )  # z is xt in Eq 24

        logits = model(xt, t, **model_kwargs)
        p1t = torch.softmax(logits, dim=1)
        p1t = rearrange(p1t, "b c f h w -> (b f) c h w")
        
        kappa_coeff = self.kappa.derivative(t) / (1 - self.kappa(t))
        kappa_coeff = kappa_coeff.reshape(-1, 1, 1, 1)
        u = kappa_coeff * (p1t - dirac_xt)
        return u, logits

    def backward_u(
        self, t: float | torch.Tensor, xt: Img
    ) -> Prob:  # Eq 25, more in Eq 49 and Eq 59
        dirac_xt = indices_to_diracp(xt, self.vocab_size, self.type_data)
        x0 = (
            torch.ones_like(xt) * self.mask_token_id
        )  # taohufffix, fixed, @yunlu, should be right, check Eq 59.
        dirac_x0 = indices_to_diracp(x0, self.vocab_size, self.type_data)
        kappa_coeff = self.kappa.derivative(t) / self.kappa(t)
        return kappa_coeff * (dirac_xt - dirac_x0)

    def bar_u(
        self,
        t: float | torch.Tensor,
        xt: Img,
        alpha_t: float | torch.Tensor,
        beta_t: float | torch.Tensor,
        model: nn.Module,
        **model_kwargs,
    ) -> Prob:  # Eq 26
        return alpha_t * self.forward_u(
            t, xt, model, **model_kwargs
        ) - beta_t * self.backward_u(t, xt)

    def corrupt_data(
        self,
        p0: Prob,
        p1: Prob,
        t: torch.Tensor | float,
        kappa: KappaScheduler,
        type_data: str,
    ) -> Img:
        p0_shape = p0.shape
        assert len(t.shape) == 1
        t = t.view(-1, *([1] * (len(p0_shape) - 1)))  # automaticaly broadcast
        pt = (1 - kappa(t)) * p0 + kappa(t) * p1

        assert torch.all(pt >= 0), "Negative probabilities in pt"
        assert torch.allclose(pt.sum(dim=1), torch.ones_like(pt.sum(dim=1))), "Probabilities do not sum to 1"

        if self.smoothing_factor > 0.0:
            pt = pt + self.smoothing_factor * (1 - kappa(t)) * kappa(t)

        return sample_p(pt, type_data)

    def training_losses(self, model, x, noise_levels) -> torch.Tensor:
        """
        Compute the training loss using individual noise levels for each frame.

        Args:
            model (nn.Module): The model to train.
            x (torch.Tensor): Input tensor of shape (t, b, fs, 32, 32).
            noise_levels (torch.Tensor): Noise levels tensor of shape (t, b).

        Returns:
            dict: A dictionary containing loss, logits, corrupted inputs, and metrics.
        """
        n_frames, batch_size = x.shape[:2]
        device = x.device

        normalized_noise = noise_levels.float() / 1000.0
        x_flat = x.reshape(n_frames * batch_size, *x.shape[3:]) 
        t_flat = normalized_noise.reshape(n_frames * batch_size)

        x0_flat, x1_target_flat = self.coupling.sample(x_flat)
        dirac_x0_flat = indices_to_diracp(x0_flat.long(), self.vocab_size, self.type_data)  
        dirac_x1_flat = indices_to_diracp(x1_target_flat.long(), self.vocab_size, self.type_data) 

        xt_flat = self.corrupt_data(
            dirac_x0_flat, dirac_x1_flat, t_flat, self.kappa, self.type_data
        )  

        xt = xt_flat.reshape(batch_size, n_frames, *xt_flat.shape[1:])  
        x1_target = x1_target_flat.reshape(batch_size, n_frames, *x1_target_flat.shape[1:]) 
        dirac_x1 = dirac_x1_flat.reshape(batch_size, n_frames, *dirac_x1_flat.shape[1:])

        t = rearrange(normalized_noise, "f b -> b f")

        logits_x = model(
            x=xt,  
            t=t,    
            use_fp16=False,
            cond_drop_prob=0.0,  
        )

        loss = F.cross_entropy(
            logits_x, 
            x1_target, 
            ignore_index=16384,
            reduction="none"
        )  

        target_mask = (xt != x1_target).float()  
        target_mask_flat = target_mask.reshape(-1) 

        if self.mask_ce:
            loss = (loss * target_mask_flat).sum() / (target_mask_flat.sum() + 1e-7)
        else:
            loss = loss.mean()

        preds = logits_x.argmax(dim=1)
        preds_flat = preds.view(-1) 
        x1_target_flat = x1_target_flat.view(-1)

        acc = ((preds_flat == x1_target_flat).float() * target_mask_flat).sum() / (target_mask_flat.sum() + 1e-7)

        print(f"Traning step acc: {acc}, Training step loss: {loss}")

        ret_dict = {
            "loss": loss,
            "logits": logits_x.clone(),
            "x_corrupt": xt.clone(),
            "log/mask_ce": int(self.mask_ce),
            "log/acc": acc.clone(),
        }

        return ret_dict


class DiscreteSampler:
    def __init__(self, adaptative: bool = True) -> None:
        self.h = self.adaptative_h if adaptative else self.constant_h

    def u(
        self, t: float | torch.Tensor, xt: Img, discretefm: DiscreteFM, model: nn.Module
    ) -> Prob:
        raise NotImplementedError

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        raise NotImplementedError

    def constant_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        return h

    def construct_x0(
        self, sample_size: Tuple[int], device: torch.device, vocab_size: int
    ) -> Tuple[Img, Prob]:
        x0 = (
            torch.ones(sample_size, device=device, dtype=torch.long)
            * self.mask_token_id
        )
        dirac_x0 = indices_to_diracp(x0, vocab_size, self.data_type)
        return x0, dirac_x0

    def __call__(
        self,
        sample_size: Tuple[int],
        discretefm: DiscreteFM,
        model: nn.Module,
        n_steps: int,
        t_min: float = 1e-4,
        **model_kwargs,
    ) -> List[Img]:
        t = t_min * torch.ones(sample_size[0], device=discretefm.device)
        default_h = 1 / n_steps
        xt, dirac_xt = self.construct_x0(
            sample_size, discretefm.device, discretefm.vocab_size
        )
        list_xt = [xt]
        t = pad_like_x(t, dirac_xt)

        while t.max() <= 1 - default_h:
            h = self.h(default_h, t, discretefm)
            pt = dirac_xt + h * self.u(t, xt, discretefm, model, **model_kwargs)
            xt = sample_p(pt, discretefm.type_data)
            # Eq 12 in https://arxiv.org/pdf/2407.15595
            dirac_xt = indices_to_diracp(
                xt, discretefm.vocab_size, discretefm.type_data
            )
            t += h
            list_xt.append(xt)
        return list_xt

    def sample_step_with_noise_schedule(
        self,
        xt: torch.Tensor,
        from_noise_levels: torch.Tensor,
        to_noise_levels: torch.Tensor,
        discretefm: DiscreteFM,
        model: nn.Module,
        n_steps: int,
        model_kwargs: dict = {}
    ):
        default_h = 1 / n_steps
        h = self.h(default_h, from_noise_levels, discretefm)
        
        frames, batch_size, H, W = xt.shape
        xt = rearrange(xt, "f b ... -> b f ...")
        xt_flat = xt.reshape(batch_size * frames, *xt.shape[2:]).long()
        
        u, logits = self.u(from_noise_levels, xt, discretefm, model, **model_kwargs)

        dirac_xt = indices_to_diracp(
            xt_flat,
            discretefm.vocab_size,
            discretefm.type_data
        )

        h = h.reshape(-1, 1, 1, 1)
        
        pt = dirac_xt + h * u
        xt_new = sample_p(pt, discretefm.type_data)

        dirac_xt_new = indices_to_diracp(
            xt_new.long(),
            discretefm.vocab_size,
            discretefm.type_data
        )

        corrupted_xt = discretefm.corrupt_data(
            p0=dirac_xt_new,           
            p1=dirac_xt,               
            t=to_noise_levels.flatten(),         
            kappa=discretefm.kappa,
            type_data=discretefm.type_data
        )

        corrupted_xt = corrupted_xt.reshape(frames, batch_size, *corrupted_xt.shape[1:])
        return corrupted_xt, logits


class SimpleSampler(DiscreteSampler):
    def __init__(
        self, mask_token_id: int, input_tensor_type: str = "bt", adaptive_stepsize=True
    ):
        super().__init__(adaptive_stepsize)
        self.mask_token_id = mask_token_id
        self.data_type = input_tensor_type

    def u(
        self,
        t: float | torch.Tensor,
        xt: Img,
        discretefm: DiscreteFM,
        model: nn.Module,
        **model_kwargs,
    ) -> Prob:
        return discretefm.forward_u(t, xt, model, **model_kwargs)

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:  # Eq 30 in https://arxiv.org/pdf/2407.15595
        coeff = (1 - discretefm.kappa(t)) / discretefm.kappa.derivative(t)
        h = torch.tensor(h, device=discretefm.device)
        h_adapt = torch.minimum(h, coeff)
        return h_adapt


class CorrectorSampler(DiscreteSampler):
    def __init__(
        self,
        mask_token_id: int,
        adaptive_stepsize: bool = True,
        alpha: float = 12.0,
        a: float = 2.0,
        b: float = 0.25,
    ) -> None:
        super().__init__(adaptive_stepsize)
        self.alpha = alpha
        self.a, self.b = a, b
        self.alpha_t = lambda t: 1 + (self.alpha * (t**self.a)) * (
            (1 - t) ** self.b
        )  # Eq 75 in https://arxiv.org/pdf/2407.15595
        self.beta_t = (
            lambda t: self.alpha_t(t) - 1
        )  # Eq 75 in https://arxiv.org/pdf/2407.15595
        self.mask_token_id = mask_token_id

    def u(
        self,
        t: float | torch.Tensor,
        xt: Img,
        discretefm: DiscreteFM,
        model: nn.Module,
        **model_kwargs,
    ) -> Prob:
        return discretefm.bar_u(
            t, xt, self.alpha_t(t), self.beta_t(t), model, **model_kwargs
        )

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:  # Eq 31 in https://arxiv.org/pdf/2407.15595
        alpha_term = (
            self.alpha_t(t) * discretefm.kappa.derivative(t) / (1 - discretefm.kappa(t))
        )
        beta_term = (
            self.beta_t(t) * discretefm.kappa.derivative(t) / discretefm.kappa(t)
        )
        coeff = 1 / (alpha_term + beta_term)
        h = torch.tensor(h, device=discretefm.device)
        h_adapt = torch.minimum(h, coeff)
        return h_adapt


if __name__ == "__main__":
    pass
