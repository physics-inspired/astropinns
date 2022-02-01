import torch


def gradients(outputs, inputs):
    """
        Compute t, x partial derivatives of variable u\
    Input
    -----
    outputs: variable u
    inputs: grid of t, x
    
    Output
    ------
    grad[:, :1]: du/dt 
    grad[:, 1:]: du/dx
    
    Note: grad[:, :1] gives shape [:, 1] instead of [:]
    """
    grad = torch.autograd.grad(outputs,
                               inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]
    
    return grad[:, :1], grad[:, 1:]


def euler(model, x):
        """
            Non-relativistic euler
        Input
        -----
        x: tensor containing variables (t, x)
        
        Output
        ------
        f: losses
        f_all: error
        """
        y = model.net(x)
        
        # Density, pressure and velocity
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        
        # Heat Capacity Ratio
        gamma = 1.4

        # Partial derivatives
        
        rho_t, rho_x = gradients(rho, x)                 
        u_t, u_x = gradients(u, x)
        p_t, p_x = gradients(p, x)
        _, u_xx = gradients(u_x, x)
        
        # Loss function
        f = ((rho_t + u * rho_x + rho * u_x) ** 2).mean() + \
            ((rho * (u_t + u * u_x) + (p_x)) ** 2).mean() + \
            ((p_t + gamma * p * u_x + u * p_x) ** 2).mean()
        
        # Error
        f_all = ((rho_t + u * rho_x + rho * u_x) ** 2) + \
            ((rho * (u_t + (u) * u_x) + (p_x)) ** 2) + \
            ((p_t + gamma * p * u_x + u * p_x) ** 2)
        
        return f ,f_all

def relativistic_euler(model, x):
    """
        Relativistic Euler
    Input
    -----
    x: tensor containing variables (t, x)

    Output
    ------
    f: losses
    f_all: error
    """

    y = model.net(x)  

    # Density, pressure and velocity
    rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]

    # Heat Capacity Ratio
    gamma = 1.4
    
    # Lorentz factor
    W = 1.0 / torch.sqrt( 1 - u * u)
    eps = p /((gamma - 1.0) * rho) 
    h = 1.0 + eps + p / rho

    #Conserved variables
    a = rho * h * W * W
    D = rho * W
    S = a * u 
    tau = a - p - D

    f1 = D * u
    f2 = S * u + p
    f3 = S - D * u

    dD_dt, _ = gradients(D, x)
    dS_dt, _ = gradients(S, x)
    dtau_dt, _ = gradients(tau, x)

    _, df1dx = gradients(f1, x)
    _, df2dx = gradients(f2, x)
    _, df3dx = gradients(f3, x)

    # Loss function
    f = ((dD_dt + df1dx) ** 2).mean() + \
        ((dS_dt + df2dx) ** 2).mean() + \
        ((dtau_dt + df3dx) ** 2).mean()

    # Error
    f_all = ((dD_dt + df1dx) ** 2) + \
        ((dS_dt + df2dx) ** 2) + \
        ((dtau_dt + df3dx) ** 2)

    return f ,f_all


def loss_ic(model, x_ic, rho_ic, u_ic, p_ic):
    """
       Loss function for initial condition.
    """
    
    y_ic = model.net(x_ic)  # Initial condition
    
    rho_ic_nn = y_ic[:, 0]
    p_ic_nn = y_ic[:, 1]
    u_ic_nn = y_ic[:, 2]

    # Loss function for the initial condition
    loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
           ((rho_ic_nn - rho_ic) ** 2).mean()  + \
           ((p_ic_nn - p_ic) ** 2).mean()

    return loss_ics
