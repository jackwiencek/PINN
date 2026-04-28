import torch
from scipy.special import roots_hermite

class PDEProblem:
    """Base class for PDE definitions. Subclass per equation."""

    x_min: float
    x_max: float
    t_max: float

    def residual(self, model, x, t):
        """PDE residual (should -> 0). Use autograd for derivatives."""
        raise NotImplementedError

    def initial_condition(self, x):
        """u(x, 0)"""
        raise NotImplementedError

    def boundary_conditions(self):
        """Return list of BC dicts: [{"type": "dirichlet", "x": float, "value": float}, ...]"""
        raise NotImplementedError

    def analytical_solution(self, x, t) -> "torch.Tensor | None":
        """Optional closed-form solution. Return None if unavailable."""
        return None


class HeatEquation1D(PDEProblem):
    def __init__(self, L=1.0, T=1.0, alpha=1.0):
        self.x_min = 0.0
        self.x_max = L
        self.t_max = T
        self.alpha = alpha

    def residual(self, model, x, t):
        u = model(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t - self.alpha * u_xx

    def initial_condition(self, x):
        return torch.sin(torch.pi * x / self.x_max)

    def boundary_conditions(self):
        return [
            {"type": "dirichlet", "x": self.x_min, "value": 0.0},
            {"type": "dirichlet", "x": self.x_max, "value": 0.0},
        ]

    def analytical_solution(self, x, t):
        k = torch.pi / self.x_max
        return torch.sin(k * x) * torch.exp(-(k ** 2) * self.alpha * t)

class ViscousBurgers1D(PDEProblem):
    def __init__(self, x_min=-1.0, x_max=1.0, T=0.99, nu=0.01 / torch.pi):
        self.x_min = x_min
        self.x_max = x_max
        self.t_max = T
        self.nu = nu

    def residual(self, model, x, t):
        u = model(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx

    def initial_condition(self, x):
        # Standard benchmark IC: -sin(pi * x)
        return -torch.sin(torch.pi * x)

    def boundary_conditions(self):
        # Dirichlet BCs at x = -1 and x = 1
        return [
            {"type": "dirichlet", "x": self.x_min, "value": 0.0},
            {"type": "dirichlet", "x": self.x_max, "value": 0.0},
        ]

    def analytical_solution(self, x, t):
        # Cole-Hopf + Hermite quadrature on the heat-kernel convolution.
        # u = -∫sin(π y)·exp(A) e^{-z²} dz  /  ∫exp(A) e^{-z²} dz
        # where y = x - 2√(νt)·z and A = -cos(π y)/(2πν).
        # Bessel-series form suffers catastrophic cancellation at x≈0
        # (sum collapses to ~e^{-c}/I₀(c) ≈ 7e-44); this form does not.
        # Subtract per-point max(A) before exp for overflow safety.
        Nq = 50
        nodes, weights = roots_hermite(Nq)

        dev = x.device
        z = torch.tensor(nodes,   dtype=torch.float64, device=dev)
        w = torch.tensor(weights, dtype=torch.float64, device=dev)

        x_f = x.double().squeeze(-1)
        t_f = t.double().squeeze(-1)

        sqrt_4nut = torch.sqrt(4.0 * self.nu * t_f).unsqueeze(1)        # (M, 1)
        y = x_f.unsqueeze(1) - z.unsqueeze(0) * sqrt_4nut                # (M, Nq)

        A = -torch.cos(torch.pi * y) / (2.0 * torch.pi * self.nu)
        A_max = A.max(dim=1, keepdim=True).values
        expA = torch.exp(A - A_max)

        num = -(w * torch.sin(torch.pi * y) * expA).sum(dim=1)
        den =  (w * expA).sum(dim=1)

        return (num / den).to(x.dtype).unsqueeze(-1)


def compute_loss(model, pde, x_f, t_f, x_ic, t_ic, x_bcs, t_bcs, lambdas=(1.0, 1.0, 1.0)):
    # --- PDE residual ---
    residual = pde.residual(model, x_f, t_f)
    l_pde = torch.mean(residual ** 2)

    # --- IC ---
    u_pred_ic = model(x_ic, t_ic)
    u_ic = pde.initial_condition(x_ic)
    l_ic = torch.mean((u_pred_ic - u_ic) ** 2)

    # --- BC ---
    bcs = pde.boundary_conditions()
    bc_losses = []
    for bc in bcs:
        if bc["type"] == "dirichlet":
            N = t_bcs.shape[0]
            x_bc = torch.full((N, 1), bc["x"], device=t_bcs.device)
            u_target = torch.full((N, 1), bc["value"], device=t_bcs.device)
            u_pred_bc = model(x_bc, t_bcs)
            bc_losses.append(torch.mean((u_pred_bc - u_target) ** 2))
    if bc_losses:
        l_bc = torch.stack(bc_losses).mean()
    else:
        l_bc = torch.zeros(1, device=x_f.device).squeeze()

    lam_pde, lam_ic, lam_bc = lambdas
    total = lam_pde * l_pde + lam_ic * l_ic + lam_bc * l_bc
    return total, (l_pde, l_ic, l_bc)
