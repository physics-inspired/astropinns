from DNN_torch import DNN2
from utils import generate_data, train, new_points
import torch
from time import time
from losses import euler as loss_pde

device = torch.device('cuda')

# Run on CPU
lr = 0.0005  # Learning rate
epochs = 92300  # Number of iterations
num_x = 1000  # Number of points in t
num_t = 1000  # Number of points in x
num_f_train = 11000  # Random sampled points in interior
num_i_train = 1000  # Random sampled points from IC

x_ic_train, x_int_train, x_test, rho_ic_train, u_ic_train, x_test, p_ic_train = generate_data(num_x, num_t, num_f_train, num_i_train, device)

# Initialize neural network
model = DNN2(40, 3).to(device)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('Start training...')

tic = time()

for epoch in range(1, epochs + 1):

    loss_value = train(model, epoch, optimizer, x_int_train, x_ic_train, rho_ic_train, u_ic_train, p_ic_train)

    if epoch % 5000 == 0:

        losspde, err = loss_pde(model, x_int_train)
        x_int_train = new_points(err, x_int_train, 100, device)
        print(x_int_train.shape)

toc = time()
print(f'Total training time: {toc - tic}')
