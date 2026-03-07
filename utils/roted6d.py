import torch

def project_rot6d(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 6]  (two 3D vectors)
    return: [..., 6]  Gram-Schmidt projected 6D (valid rotation-6d)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    dot = torch.sum(a2 * b1, dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)

    return torch.cat([b1, b2], dim=-1)