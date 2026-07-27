"""RegisterEncoder: one-hot tensor encoding split into two single-register states.

The monolithic baseline encodes a ``d x d`` image ``M`` (Frobenius-normalised)
into the pure joint state ``|x> = vec(M)`` over ``d^2`` dims, i.e. the density
matrix ``rho = |x><x|`` on the bipartite system (X register = rows, Y register =
columns), one photon per register.

For the *distributed* model each QPU only physically receives the **reduced**
state of its register:

    rho_X = Tr_Y(|x><x|)        (d x d, on the row register)
    rho_Y = Tr_X(|x><x|)        (d x d, on the column register)

A short derivation shows these partial traces are exactly the Gram matrices of
the image:

    rho_X = M M^H               (eigenvalues = squared singular values of M)
    rho_Y = M^H M

Hence each register is **pure iff M is rank-1 (separable)**, and the amount of
information lost by distributing equals the off-diagonal X|Y correlation, which
is controlled by the Schmidt/SVD rank of M. This module therefore also exposes
the separability diagnostics used by the hypothesis test in a later task.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._baseline import OneHotEncoder


def _ensure_bchw_to_rho(encoder: OneHotEncoder, x: torch.Tensor) -> torch.Tensor:
    """Run the baseline one-hot encoder, returning rho of shape (b, d^2, d^2)."""
    return encoder(x)


class RegisterEncoder(nn.Module):
    """Encode a batch of ``d x d`` images into two reduced register states.

    Forward returns ``(rho_X, rho_Y)`` each of shape ``(b, d, d)`` and complex
    dtype, where ``rho_X`` lives on the row register and ``rho_Y`` on the column
    register. Uses the baseline :class:`OneHotEncoder` so the encoding basis is
    identical to the monolithic model, then performs the exact partial trace.
    """

    def __init__(self, dims: tuple[int, int]):
        super().__init__()
        if dims[0] != dims[1]:
            raise NotImplementedError("Only square images are supported.")
        self.dims = dims
        self.d = dims[0]
        self.one_hot = OneHotEncoder()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.d
        rho = _ensure_bchw_to_rho(self.one_hot, x)  # (b, d^2, d^2), complex
        b = rho.shape[0]
        # reshape joint indices a=(i,j), b=(m,n)
        rho5 = rho.reshape(b, d, d, d, d)  # (batch, i, j, m, n)
        # rho_X[i,m] = sum_j rho[i,j,m,j]
        rho_x = torch.einsum("bijmj->bim", rho5)
        # rho_Y[j,n] = sum_i rho[i,j,i,n]
        rho_y = torch.einsum("bijin->bjn", rho5)
        return rho_x, rho_y


# ---------------------------------------------------------------------------
# Separability diagnostics (used by the SVD-rank hypothesis test).
# ---------------------------------------------------------------------------
def register_purity(rho: torch.Tensor) -> torch.Tensor:
    """Tr(rho^2) per batch element. 1.0 == pure (separable register)."""
    prod = torch.bmm(rho, rho)
    return prod.diagonal(dim1=1, dim2=2).sum(dim=1).real


def image_singular_values(x: torch.Tensor) -> torch.Tensor:
    """Frobenius-normalised singular values of each ``d x d`` image (b, d)."""
    if x.dim() == 4:
        x = x.squeeze(1)
    x = x.to(torch.float32)
    norm = torch.linalg.matrix_norm(x, ord="fro").clamp_min(1e-12)
    x = x / norm.unsqueeze(-1).unsqueeze(-1)
    return torch.linalg.svdvals(x)


def effective_rank(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Entropy-based effective rank exp(H(p)), p = normalised squared sing. vals.

    Returns one value per image. 1.0 for a rank-1 (separable) image; grows with
    non-separability.
    """
    s = image_singular_values(x)
    p = s.pow(2)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(p * (p.clamp_min(eps)).log()).sum(dim=-1)
    return torch.exp(entropy)


__all__ = [
    "RegisterEncoder",
    "register_purity",
    "image_singular_values",
    "effective_rank",
]
