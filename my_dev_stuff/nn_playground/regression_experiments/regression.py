import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# class for a ResNet: parameters are input size, output size, number of hidden layers, and number of nodes per hidden layer
class ResNet(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_hidden_layers,
        num_nodes_per_layer,
        dropout_prob,
        activation_function,
        normalize,
    ):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_size, self.num_nodes_per_layer))
        for i in range(self.num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(self.num_nodes_per_layer, self.num_nodes_per_layer))
        self.layers.append(torch.nn.Linear(self.num_nodes_per_layer, self.output_size))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.activation_function = activation_function
        self.normalize = normalize
        if self.normalize:
            self.batchnorm = torch.nn.BatchNorm1d(num_nodes_per_layer)

    def forward(self, x):
        for i in range(self.num_hidden_layers + 1):
            x = self.layers[i](x)
            if i != self.num_hidden_layers:
                x = self.activation_function(x)
                if self.dropout.p != 0:
                    x = self.dropout(x)
                if self.normalize:
                    x = self.batchnorm(x)
        return x


# training loop
def train(model, lossfun, x, y, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        y_pred = model(x)
        loss = lossfun(y_pred, y, x)
        if epoch % 10 == 0:
            print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


isize = 1
osize = 1
nhid = 10
nnodes = 50
dropout = 0.0
activation = torch.nn.Hardshrink()
normalize = False

targetfun = lambda x: torch.log(x)
invtargetfun = lambda x: torch.exp(x)

x_train = 0.5 + 0.5 * torch.rand(1000, isize).to(device)
y_train = targetfun(x_train)

resnet_regression = ResNet(isize, osize, nhid, nnodes, dropout, activation, normalize).to(device)
resnet_solver = ResNet(isize, osize, nhid, nnodes, dropout, activation, normalize).to(device)

lossfun_regression = lambda y_pred, y, x: torch.nn.MSELoss()(y_pred, y)
train(resnet_regression, lossfun_regression, x_train, y_train, 1000, 0.001)

lossfun_solver = lambda y_pred, y, x: torch.nn.MSELoss()(invtargetfun(y_pred), x)
train(resnet_solver, lossfun_solver, x_train, y_train, 1000, 0.001)

x_val = torch.linspace(1e-2, 1.5, 1000).reshape(-1, 1).to(device)
y_val = targetfun(x_val)

y_pred_regression = resnet_regression(x_val)
y_pred_solver = resnet_solver(x_val)


print("----------------Regression quality----------------")
print("Regression:", lossfun_regression(y_pred_regression, y_val, x_val).item())
print("Solver:", lossfun_regression(y_pred_solver, y_val, x_val).item())

print("----------------Solver quality----------------")
print("Regression:", lossfun_solver(y_pred_regression, y_val, x_val).item())
print("Solver:", lossfun_solver(y_pred_solver, y_val, x_val).item())

import matplotlib.pyplot as plt

plt.plot(x_val.cpu().detach().numpy(), y_pred_regression.cpu().detach().numpy(), label="Regression")
plt.plot(x_val.cpu().detach().numpy(), y_pred_solver.cpu().detach().numpy(), label="Solver")
plt.plot(x_val.cpu().detach().numpy(), y_val.cpu().detach().numpy(), label="True", linestyle="--")
plt.legend()
plt.savefig("plot.png")
