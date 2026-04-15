import torch
import torch.nn as nn
from model import PINN
from sampling import sample_collocation,sample_ic,sample_bc
from physics import compute_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PINN(input_dim=2, hidden_layers=6, neurons=40)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 1000

#All points, requires_grad = True
#fixed for now
x_f, t_f = sample_collocation(5000, 1.0, 1.0, device)
x_ic, t_ic, u_ic = sample_ic(200, 1.0, device)
x_bc_left, x_bc_right, t_bc, u_bc_left, u_bc_right = sample_bc(200, 1.0, 1.0, device)

total_loss_list = list()
col_loss_list = list()
ic_loss_list = list()
bc_loss_list = list()

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
    
    total_loss_list.append(total_loss)
    col_loss_list.append(l_pde)
    ic_loss_list.append(l_ic)
    bc_loss_list.append(l_bc)

    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}: total={total_loss.item():.4e} "
            f"pde={l_pde.item():.4e} ic={l_ic.item():.4e} bc={l_bc.item():.4e}")
        
