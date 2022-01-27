import numpy as np
import torch
from losses import loss_ic
from losses import euler as loss_pde


def IC(x):
    """
        We compute the initial conditions with this function.
    """
    N = len(x)
    rho_init = np.ones((x.shape[0]))  # rho - initial condition
    u_init = np.zeros((x.shape[0]))  # u - initial condition
    p_init = np.ones((x.shape[0]))  # eps - initial condition

    # rho - initial condition
    for i in range(N):
        if (x[i] <= 0.5):
            # rho_init[i] = 1.0
            p_init[i] = 1.0
            u_init[i] = 0.9
        else:
            # rho_init[i] = 1
            p_init[i] = 10.0

    return rho_init, u_init, p_init


def generate_data(num_x, num_t, num_f_train, num_i_train, device):
    # FIXME: fix limits

    # I think this x, t definitions come from Alexandros Papados paper
    # x = np.linspace(-1.5, 3.125,num_x)  # Partitioned spatial axis
    # t = np.linspace(0, 0.2, num_t)  # Partitioned time axis
    x = np.linspace(0, 1, num_x)  # Partitioned spatial axis
    t = np.linspace(0, 0.2, num_t)  # Partitioned time axis
    t_grid, x_grid = np.meshgrid(t, x)  # (t,x) in [0,2.0]x[a,b]
    # t_grid = t x 1000, x_grid = 1000 x x

    T = t_grid.flatten()[:, None]  # Vectorized t_grid
    X = x_grid.flatten()[:, None]  # Vectorized x_grid
    # flatten happens always in the same direction, so X is ordered and T is not.

    # Generate IDs
    id_ic = np.random.choice(num_x,
                             num_i_train,
                             replace=False)  # Random sample numbering for IC
    id_f = np.random.choice(num_x*num_t,
                            num_f_train,
                            replace=False)  # Random sample numbering for interior

    # Sampling at random. FIXME: maybe we can do it better
    # Define initial condictions
    x_ic = x_grid[id_ic, 0][:, None]  # Random x - initial condition
    t_ic = t_grid[id_ic, 0][:, None]  # random t - initial condition
    x_ic_train = np.hstack((t_ic, x_ic))  # Random (x,t) - vectorized
    rho_ic_train, u_ic_train, p_ic_train = IC(x_ic)  # Initial condition evaluated at random sample

    # Define interior points
    x_int = X[:, 0][id_f, None]  # Random x - interior
    t_int = T[:, 0][id_f, None]  # Random t - interior
    x_int_train = np.hstack((t_int, x_int))  # Random (x,t) - vectorized
    x_test = np.hstack((T, X))  # Vectorized whole domain

    # Generate tensors
    x_ic_train = torch.tensor(x_ic_train,
                              dtype=torch.float32).to(device)
    x_int_train = torch.tensor(x_int_train, requires_grad=True,
                               dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test,
                          dtype=torch.float32).to(device)

    rho_ic_train = torch.tensor(rho_ic_train,
                                dtype=torch.float32).to(device)
    u_ic_train = torch.tensor(u_ic_train,
                              dtype=torch.float32).to(device)
    p_ic_train = torch.tensor(p_ic_train,
                              dtype=torch.float32).to(device)

    return x_ic_train, x_int_train, x_test, rho_ic_train, u_ic_train, x_test, p_ic_train


def train(model, epoch, optimizer, x_int_train, x_ic_train, rho_ic_train, u_ic_train, p_ic_train):

    model.train()

    def closure():

        optimizer.zero_grad()

        # Loss function of PDE
        losspde, err = loss_pde(model, x_int_train)

        # Loss function of IC
        lossic = loss_ic(model,
                         x_ic_train,
                         rho_ic_train,
                         u_ic_train,
                         p_ic_train)
        # lambda_i = lr_ann(model,losspde,lossic)
        # lambda_1 = 0.1 * lambda_1 + 0.9*lambda_i
        # Total loss function G(theta)
        loss = losspde + lossic

        if epoch % 50 == 0:
            print(f'epoch {epoch} loss_pde: {losspde:.8e}, loss_ic: {lossic:.8e}')

        loss.backward()
        return loss

    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss, float) else loss
    return loss_value


def new_points(err, X_r, N, device):

    X_r = X_r.cpu().detach().numpy()
    err = err.cpu().detach().numpy()
    t_i, x_i = np.unravel_index(np.argmax(err, axis=None), err.shape)

    t = (X_r[t_i][0])
    x = (X_r[t_i][1])

    tmin = max(t - 0.05, 0)
    tmax = min(t + 0.05, 0.2)
    xmin = max(x - 0.2, 0)
    xmax = min(x + 0.3, 1)

    num_t = N
    num_x = N
    x = np.linspace(xmin, xmax, num_x)  # Partitioned spatial axis
    t = np.linspace(tmin, tmax, num_t)
    try:
        t_r_max = torch.FloatTensor(N, 1).uniform_(tmin, tmax)

        x_r_max = torch.FloatTensor(N, 1).uniform_(xmin, xmax)
    except (RuntimeError, TypeError, NameError):
        print(t_r_max, x_r_max)

    X_r_new = torch.cat([t_r_max, x_r_max], axis=1).to(device)
    X_r = torch.tensor(X_r, requires_grad=True, dtype=torch.float32).to(device)
    X_r = torch.cat((X_r, X_r_new))

    return X_r
