import torch


def sample_collocation(N, pde, device):
    x = (torch.rand(N, 1, device=device) * (pde.x_max - pde.x_min) + pde.x_min).requires_grad_(True)
    t = (torch.rand(N, 1, device=device) * pde.t_max).requires_grad_(True)
    return x, t


def sample_ic(N, pde, device):
    x = torch.rand(N, 1, device=device) * (pde.x_max - pde.x_min) + pde.x_min
    t = torch.zeros(N, 1, device=device)
    u = pde.initial_condition(x)
    return x, t, u


def sample_bc(N, pde, device):
    t = torch.rand(N, 1, device=device) * pde.t_max
    return t
