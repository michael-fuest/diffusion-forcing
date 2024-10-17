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
        input_tensor_type: str = "bt",
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
        dirac_xt = indices_to_diracp(
            xt, self.vocab_size, self.type_data
        )  # z is xt in Eq 24
        p1t = torch.softmax(model(xt, t.flatten(), **model_kwargs), dim=1)
        kappa_coeff = self.kappa.derivative(t) / (1 - self.kappa(t))
        return kappa_coeff * (p1t - dirac_xt)

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

        if self.smoothing_factor > 0.0:
            pt = pt + self.smoothing_factor * (1 - kappa(t)) * kappa(t)

        return sample_p(pt, type_data)

    def training_losses(self, model, x, model_kwargs) -> torch.Tensor:
        t = torch.rand(len(x), device=x.device)
        x0, x1_target = self.coupling.sample(x)
        dirac_x0 = indices_to_diracp(x0.long(), self.vocab_size, self.type_data)
        dirac_x1 = indices_to_diracp(x1_target.long(), self.vocab_size, self.type_data)
        xt = self.corrupt_data(
            dirac_x0, dirac_x1, t, self.kappa, self.type_data
        )  # [B,T]
        logits_x = model(xt, t, **model_kwargs)
        target_mask = xt != x1_target  # following campbell  et al.
        loss = F.cross_entropy(
            logits_x,
            x1_target.long(),
            ignore_index=-1,
            reduction="none",
        )
        if self.mask_ce:
            loss = torch.sum(loss * target_mask) / (torch.sum(target_mask) + 1e-7)
        else:
            loss = loss.mean()
        acc = ((logits_x.argmax(dim=1) == x1_target) * target_mask).sum() / (
            torch.sum(target_mask) + 1e-7
        )
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
