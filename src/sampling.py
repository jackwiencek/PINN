import torch

def sample_collocation(N, L, T, device):
    x = (torch.rand(N, 1, device=device) * L).requires_grad_(True)
    t = (torch.rand(N, 1, device=device) * T).requires_grad_(True)
    return x, t

def sample_ic(N, L, device, f=None):
    if f is None:
        f = lambda x: torch.sin(torch.pi * x / L)
    x = torch.rand(N, 1, device=device) * L
    t = torch.zeros(N, 1, device=device)
    return x, t, f(x)

def sample_bc(N, L, T, device, u_left=0.0, u_right=0.0):
    t = torch.rand(N, 1, device=device) * T
    x_left = torch.zeros(N, 1, device=device)
    x_right = torch.full((N, 1), L, device=device)
    u_l = torch.full((N, 1), u_left, device=device)
    u_r = torch.full((N, 1), u_right, device=device)
    return x_left, x_right, t, u_l, u_r