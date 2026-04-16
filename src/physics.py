import torch


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
