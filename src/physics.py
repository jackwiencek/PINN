import torch

def compute_loss(model, x_f, t_f, x_ic, t_ic, u_ic,
                 x_bc_left, x_bc_right, t_bc, u_bc_left, u_bc_right,
                 alpha, lambdas=(1.0, 1.0, 1.0)):

    # --- PDE residual ---
    u = model(x_f, t_f)
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x),create_graph=True)[0]
    residual = u_t - alpha * u_xx
    l_pde = torch.mean(residual**2)

    # --- IC ---
    u_pred_ic = model(x_ic, t_ic)
    l_ic = torch.mean((u_pred_ic - u_ic)**2)

    # --- BC ---
    u_pred_bc_l = model(x_bc_left,  t_bc)
    u_pred_bc_r = model(x_bc_right, t_bc)
    l_bc = 0.5*(torch.mean((u_pred_bc_l - u_bc_left)**2) + torch.mean((u_pred_bc_r - u_bc_right)**2))

    lam_pde, lam_ic, lam_bc = lambdas
    total = lam_pde * l_pde + lam_ic * l_ic + lam_bc * l_bc
    return total, (l_pde, l_ic, l_bc)