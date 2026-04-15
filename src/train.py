import torch
import torch.nn as nn
from model import PINN
from sampling import sample_collocation,sample_ic,sample_bc

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
    l_bc = torch.mean((u_pred_bc_l - u_bc_left)**2) + torch.mean((u_pred_bc_r - u_bc_right)**2)

    lam_pde, lam_ic, lam_bc = lambdas
    total = lam_pde * l_pde + lam_ic * l_ic + lam_bc * l_bc
    return total, (l_pde, l_ic, l_bc)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PINN(input_dim=2, hidden_layers=6, neurons=40)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = torch.nn.MSELoss()

loss_pde = 1
loss_ic = 1
loss_bc = 1

epochs = 1000

#All points, requires_grad = True
#fixed for now
x_f, t_f = sample_collocation(5000, 1.0, 1.0, device)
x_ic, t_ic, u_ic = sample_ic(200, 1.0, device)
x_bc_left, x_bc_right, t_bc, u_bc_left, u_bc_right = sample_bc(200, 1.0, 1.0, device)

#Training Loop
for epoch in range(epochs):
    model.train()
    
    #zero out gradients (clear old slopes from last backprop?)
    optimizer.zero_grad()

    #FORWARD PASS

    alpha = 1.0

    total_loss, (l_pde, l_ic, l_bc) = compute_loss(
            model, x_f, t_f, x_ic, t_ic, u_ic,
            x_bc_left, x_bc_right, t_bc, u_bc_left, u_bc_right, alpha
        )

    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"epoch {epoch}: total={total_loss.item():.4e} "
            f"pde={l_pde.item():.4e} ic={l_ic.item():.4e} bc={l_bc.item():.4e}")

    

