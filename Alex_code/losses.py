import torch


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs,
                               inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

def euler(model, x):
        """
            Non-relativistic euler
        """
        y = model.net(x)                                               # Neural network
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]                     # NN_{rho}, NN_{u}, NN_{p}
        gamma = 1.4                                                    # Heat Capacity Ratio

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]                    # Partial derivatives rho_t, rho_x

        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            # Partial derivatives u_t, u_x

        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            # Partial derivatives p_t, p_x

        u_xx = gradients(u_x, x)[0][:, 1:]
        
        # Loss function for the Euler Equations
        f = ((rho_t + u * rho_x + rho * u_x) ** 2).mean() + \
            ((rho * (u_t + (u) * u_x) + (p_x)) ** 2).mean() + \
            ((p_t + gamma * p * u_x + u * p_x) ** 2).mean()

        f_all = ((rho_t + u * rho_x + rho * u_x) ** 2) + \
            ((rho * (u_t + (u) * u_x) + (p_x)) ** 2) + \
            ((p_t + gamma * p * u_x + u * p_x) ** 2)
        
        return f ,f_all

def relativistic_euler(model, x):
    """
        Relativistic Euler
    """

    y = model.net(x)  
    t = x[:,0]
    x1 = x[:,1]  # Neural network

    rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]  # NN_{rho}, NN_{u}, NN_{eps}
    
    gamma = 1.4  # Heat Capacity Ratio
    W = 1.0 / torch.sqrt( 1 - u * u) # Lorentz factor
    eps = p /((gamma - 1.0) * rho) 
    h = 1.0 + eps + p / rho

    #Conserved variables
    a = rho * h * W * W #
    D = rho * W
    S = a * u 
    tau = a - p - D
    
    f1 = D * u
    f2 = S * u + p
    f3 = S - D * u
    
    #FIXME: improve gradient efficiency
    dD_dt = gradients(D, x)[0][:, :1]
    dS_dt = gradients(S, x)[0][:, :1]
    dtau_dt = gradients(tau, x)[0][:, :1]

    df1dx = gradients(f1, x)[0][:,1:]
    df2dx = gradients(f2, x)[0][:, 1:]
    df3dx = gradients(f3, x)[0][:, 1:]

    f = ((dD_dt + df1dx) ** 2).mean() + \
        ((dS_dt + df2dx) ** 2).mean() + \
        ((dtau_dt + df3dx) ** 2).mean()

    f_all = ((dD_dt + df1dx) ** 2) + \
        ((dS_dt + df2dx) ** 2) + \
        ((dtau_dt + df3dx) ** 2)

    return f ,f_all


def loss_ic(model, x_ic, rho_ic, u_ic, p_ic):
    """
       Loss function for initial condition.
    """
    
    y_ic = model.net(x_ic)  # Initial condition
    rho_ic_nn, p_ic_nn,u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2] # rho, u, p - initial condition

    # Loss function for the initial condition
    loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
           ((rho_ic_nn - rho_ic) ** 2).mean()  + \
           ((p_ic_nn - p_ic) ** 2).mean()

    return loss_ics
