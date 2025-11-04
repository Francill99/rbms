from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import softmax


@torch.jit.script
def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    # Bernoulli–Gaussian: h|v ~ N(mu, (beta*Gamma)^{-1}), with Gamma = Nv * I, Nv = num visibles
    #Nv = weight_matrix.shape[0]
    #gamma = float(Nv)  
    #inv_beta_gamma_sqrt = (1.0 / (beta * gamma)) ** 0.5
    mh = (hbias + (v @ weight_matrix)) #/ gamma  # (B, K)
    #eps = torch.randn_like(mu)
    #h = mu + inv_beta_gamma_sqrt * eps
    #h = torch.normal(mean=mh, std=torch.tensor(1.0/torch.sqrt(mh.shape[-1])).to(weight_matrix.device))
    h = torch.normal(mean=mh, std=torch.tensor(1.0).to(weight_matrix.device))
    return h, mh  # return sample and mean (hidden_mag)


@torch.jit.script
def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    mv = torch.tanh(vbias + h @ weight_matrix.T)
    v = torch.bernoulli(0.5*(1+mv))*2-1
    return v, mv


@torch.jit.script
def _compute_energy(
    v: Tensor,
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
) -> Tensor:
    # E(v,h) = -a^T v - b^T h - v^T W h + 0.5 * h^T Gamma h ; Gamma = Nv * I
    fields = torch.tensordot(vbias, v, dims=[[0], [1]]) + torch.tensordot(
        hbias, h, dims=[[0], [1]]
    )
    interaction = torch.multiply(
        v, torch.tensordot(h, weight_matrix, dims=[[1], [1]])
    ).sum(1)
    Nv = weight_matrix.shape[0]
    gamma = float(Nv)
    quad = 0.5 * gamma * (h * h).sum(1)
    return -fields - interaction + quad


@torch.jit.script
def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    # F(v) = -a^T v - 0.5 * (b + W^T v)^T Gamma^{-1} (b + W^T v) + const(Gamma)
    # with Gamma = Nv * I  => Gamma^{-1} = (1/Nv) I
    field = v @ vbias  # (B,)
    t = hbias + (v @ weight_matrix)  # (B,K)
    Nv = weight_matrix.shape[0]
    K = weight_matrix.shape[1]
    inv_gamma = 1.0 / float(Nv)
    quad_term = 0.5 * inv_gamma * (t * t).sum(1)  # (B,)

    # constants: + 0.5 * K * log|Gamma| - 0.5 * K * log(2π)
    # |Gamma| = (Nv)^K  => log|Gamma| = K * log(Nv)
    dtype = v.dtype
    device = v.device
    log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))
    const = 0.5 * float(K) * (torch.log(torch.tensor(float(Nv), dtype=dtype, device=device)) - log_two_pi)

    return -field - quad_term + const  # (B,)


@torch.jit.script
def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    # E(h) = -b^T h - sum_i log(1 + exp(a_i + (W h)_i)) + 0.5 * h^T Gamma h
    field = h @ hbias  # (B,)
    exponent = vbias + (h @ weight_matrix.T)  # (B,V)
    # stable softplus
    log_term = torch.where(exponent < 10, torch.log1p(torch.exp(exponent)), exponent)
    Nv = weight_matrix.shape[0]
    gamma = float(Nv)
    quad = 0.5 * gamma * (h * h).sum(1)  # (B,)
    return -field - log_term.sum(1) + quad


@torch.jit.script
def _compute_gradient(
    v_data: Tensor,
    h_data: Tensor,   
    w_data: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,   
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    centered: bool,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
) -> None:
    
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    chain_weights = softmax(-w_chain, dim=0)
    w_data_norm = w_data.sum()

    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    torch.clamp_(v_data_mean, min=1e-4, max=(1.0 - 1e-4))
    h_data_mean = (h_data * w_data).sum(0) / w_data_norm
    v_gen_mean = v_chain.mean(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1.0 - 1e-4))
    h_gen_mean = h_chain.mean(0)

    if centered:
        v_data_centered = v_data - v_data_mean
        h_data_centered = h_data - h_data_mean
        v_gen_centered = v_chain - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        #grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        #grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
        grad_vbias = torch.zeros(vbias.shape[0], device=vbias.device, dtype=vbias.dtype)
        grad_hbias = torch.zeros(hbias.shape[0], device=hbias.device, dtype=hbias.dtype)
    else:
        v_data_centered = v_data
        h_data_centered = h_data
        v_gen_centered = v_chain
        h_gen_centered = h_chain

        #grad_weight_matrix = ((v_data * w_data).T @ h_data_centered) / w_data_norm - ((v_chain * chain_weights).T @ h_chain)
        # Gradient
        grad_weight_matrix = ((v_data * w_data).T @ h_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        #grad_vbias = v_data_mean - v_gen_mean
        #grad_hbias = h_data_mean - h_gen_mean
        grad_vbias = torch.zeros(vbias.shape[0], device=vbias.device, dtype=vbias.dtype)
        grad_hbias = torch.zeros(hbias.shape[0], device=hbias.device, dtype=hbias.dtype)

    #if lambda_l1 > 0:
    #    grad_weight_matrix -= lambda_l1 * torch.sign(weight_matrix)
    #    grad_vbias -= lambda_l1 * torch.sign(vbias)
    #    grad_hbias -= lambda_l1 * torch.sign(hbias)
    #
    #if lambda_l2 > 0:
    #    grad_weight_matrix -= 2 * lambda_l2 * weight_matrix
    #    grad_vbias -= 2 * lambda_l2 * vbias
    #    grad_hbias -= 2 * lambda_l2 * hbias

    # attach
    weight_matrix.grad.set_(grad_weight_matrix)
    vbias.grad.set_(grad_vbias)
    hbias.grad.set_(grad_hbias)


@torch.jit.script
def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Optional[Tensor] = None,
):
    num_visibles, num_hiddens = weight_matrix.shape
    device = weight_matrix.device
    dtype = weight_matrix.dtype
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        mv = (
            torch.ones(size=(num_samples, num_visibles), device=device, dtype=dtype) / 2
        )
        v = torch.bernoulli(mv)
    else:
        mv = torch.zeros_like(start_v, device=device, dtype=dtype)
        v = start_v.to(device=device, dtype=dtype)

    h, mh = _sample_hiddens(v=v, weight_matrix=weight_matrix, hbias=hbias)
    #h = torch.randn(size=(num_samples,num_hiddens), device=device).type(torch.float32)
    #mh = h.clone()
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-6,
):
    _, num_visibles = data.shape
    eps = 1e-4
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    #frequencies = data.mean(0)
    #frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    #vbias = (torch.log(frequencies) - torch.log(1.0 - frequencies)).to(
    #    device=device, dtype=dtype
    #)
    vbias = torch.zeros(num_visibles, device=device, dtype=dtype)
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    return vbias, hbias, weight_matrix
