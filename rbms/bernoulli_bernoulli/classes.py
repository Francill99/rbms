from typing import List, Optional, TypeVar

# Backward-compatible replacement for typing.Self (Python < 3.11)
Self = TypeVar("Self", bound="RBM")

import numpy as np
import torch
from torch import Tensor

from rbms.bernoulli_bernoulli.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)
from rbms.classes import RBM


class BBRBM(RBM):
    """Parameters of the Bernoulli-Bernoulli RBM"""

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        K1: Tensor,
        K2: Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the parameters of the Bernoulli-Bernoulli RBM.

        Args:
            weight_matrix (Tensor): The weight matrix of the RBM.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            device (Optional[torch.device], optional): The device for the parameters.
                Defaults to the device of `weight_matrix`.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
                Defaults to the data type of `weight_matrix`.
        """
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.device = device
        self.dtype = dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.w_norm_0 = torch.norm(weight_matrix)
        self.vbias = vbias.to(device=self.device, dtype=self.dtype)
        self.v_norm_0 = torch.norm(vbias)
        self.hbias = hbias.to(device=self.device, dtype=self.dtype)
        self.K1 = K1.to(device=self.device, dtype=self.dtype)
        self.K2 = K2.to(device=self.device, dtype=self.dtype)
        self.K2_norm_0 = torch.norm(K2)
        self.name = "BBRBM"

    def __add__(self, other):
        return BBRBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
        )

    def __mul__(self, other):
        return BBRBM(
            weight_matrix=self.weight_matrix * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
        )

    def clone(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return BBRBM(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v: Tensor, h: Tensor) -> Tensor:
        return _compute_energy(
            v=v,
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        return _compute_energy_hiddens(
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        return _compute_energy_visibles(
            v=v,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_gradient(self, data, chains, use_fields, centered=True):
        _compute_gradient(
            v_data=data["visible"],
            mh_data=data["hidden_mag"],
            w_data=data["weights"],
            v_chain=chains["visible"],
            h_chain=chains["hidden"],
            w_chain=chains["weights"],
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            centered=centered,
        )

    def independent_model(self):
        return BBRBM(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        visible, hidden, mean_visible, mean_hidden = _init_chains(
            num_samples=num_samples,
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
            start_v=start_v,
        )
        if weights is None:
            weights = torch.ones(
                visible.shape[0], device=visible.device, dtype=visible.dtype
            )
        return dict(
            visible=visible,
            hidden=hidden,
            visible_mag=mean_visible,
            hidden_mag=mean_hidden,
            weights=weights,
        )

    @staticmethod
    def init_parameters(num_hiddens, dataset, device, dtype, beta, use_fields, var_init=0.0001):
        data = dataset.data
        # Convert to torch Tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
        vbias, hbias, weight_matrix = _init_parameters(
            num_hiddens=num_hiddens,
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
            beta=beta,
        )
        num_visible = len(data[0,:])
        K1 = torch.randn_like(weight_matrix, device=device, dtype=dtype)/np.sqrt(float(num_hiddens))
        K2 = torch.randn_like(weight_matrix, device=device, dtype=dtype)/np.sqrt(float(num_hiddens))

        return BBRBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, K1=K1, K2=K2)

    def named_parameters(self):
        return {
            "weight_matrix": self.weight_matrix,
            "vbias": self.vbias,
            "hbias": self.hbias,
            "K1": self.K1,
            "K2": self.K2,
        }

    def num_hiddens(self):
        return self.hbias.shape[0]

    def num_visibles(self):
        return self.vbias.shape[0]

    def parameters(self) -> List[Tensor]:
        return [self.weight_matrix, self.vbias, self.hbias]

    def ref_log_z(self):
        return (
            torch.log1p(torch.exp(self.vbias)).sum() + self.num_hiddens() * np.log(2)
        ).item()

    def sample_hiddens(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
            v=chains["visible"],
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
            beta=beta,
        )
        return chains

    def sample_visibles(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["visible"], chains["visible_mag"] = _sample_visibles(
            h=chains["hidden"],
            weight_matrix=self.weight_matrix,
            vbias=self.vbias,
            beta=beta,
        )
        return chains

    @staticmethod
    def set_named_parameters(named_params: dict[str, Tensor]) -> Self:
        names = ["vbias", "hbias", "weight_matrix"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        params = BBRBM(
            weight_matrix=named_params.pop("weight_matrix"),
            vbias=named_params.pop("vbias"),
            hbias=named_params.pop("hbias"),
            K1=named_params.pop("K1"),
            K2=named_params.pop("K2")
        )
        if len(named_params.keys()) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weight_matrix = self.weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = self.vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = self.hbias.to(device=self.device, dtype=self.dtype)
        self.K1 = self.K1.to(device=self.device, dtype=self.dtype)
        self.K2 = self.K2.to(device=self.device, dtype=self.dtype)
        return self
    
    
    # ───────────────────────── 1st-order PL (single visible site) ─────────────────────────
    def compute_loss_PL1(self, data, l, use_fields=True, use_hfield=True):
        x = data                                            # [M,N] entries ∈{0,1}
    
        # mean-field hidden expectation  ⟨h_a⟩ ≃ tanh(λ⋅pre)
        h_pre = torch.einsum("ia,mi->ma", self.K1, x)       # Wᵀx
        if use_hfield:
            h_pre = h_pre + self.hbias
        h = torch.tanh(l * h_pre)                           # ±1 hidden → tanh
    
        F = torch.einsum("ja,ma->mj", self.K1, h)           # local field on each visible
        if use_fields:
            F = F + self.vbias
    
        logZ = F.softplus(l * F) if hasattr(F, "softplus") else F           # F.softplus → log(1+e^{λF})
        e_i  = -x * F + (1. / l) * logZ                    # −log P(x_i|rest)/λ
        return e_i.mean()
    
    
    # ───────────────────────── 2nd-order PL (visible–hidden pair) ─────────────────────────
    def compute_loss_PL2(self, data, l, use_fields=True, use_hfield=True):
        x = data                                            # [M,N]
        # hidden mean-field
        h_pre = torch.einsum("ia,mi->ma", self.K2, x)
        if use_fields and use_hfield:
            h_pre = h_pre + self.hbias
        h = torch.tanh(l * h_pre)                           # [M,A]
    
        # leave-one-out effective fields
        b = torch.einsum("ja,ma->mj", self.K2, h)           # vis-field from h
        c = torch.einsum("ja,mj->ma", self.K2, x)           # hid-field from x
        if use_fields:
            b = b + self.vbias
            if use_hfield:
                c = c + self.hbias
    
        j_term = torch.einsum("ja,ma->mja", self.K2, h)     # W_ja h_a
        a_term = torch.einsum("ja,mj->mja", self.K2, x)     # W_ja x_j
        b_i_eff = b.unsqueeze(2) - j_term                   # b̂_j|¬a      [M,N,1]
        c_a_eff = c.unsqueeze(1) - a_term                   # ĉ_a|¬j      [M,1,A]
    
        # observed energy   E_obs = −(W_ja x_j h_a + b̂_j x_j + ĉ_a h_a)
        w_ai = torch.einsum("ma,ja,mj->mja", h, self.K2, x) # W_ja x_j h_a
        h_ai = b_i_eff * x.unsqueeze(2) + c_a_eff * h.unsqueeze(1)
    
        # partition Z_{ja} over x_j∈{0,1}, h_a∈{±1}
        z0 = torch.exp(-l * c_a_eff)                                      # (x=0,h=-1)
        z1 = torch.exp( l * c_a_eff)                                      # (x=0,h=+1)
        z2 = torch.exp( l * (b_i_eff - self.K2 - c_a_eff))                # (x=1,h=-1)
        z3 = torch.exp( l * (b_i_eff + self.K2 + c_a_eff))                # (x=1,h=+1)
        Z_ai = z0 + z1 + z2 + z3
    
        e_ij = -w_ai - h_ai + (1. / l) * torch.log(Z_ai + 1e-9)           # −log P(x_j,h_a|rest)/λ
        return e_ij.mean()
