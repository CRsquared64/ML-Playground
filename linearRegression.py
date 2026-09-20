import torch
import torch.nn as nn
import torch.optim as optim

learning_rate = 0.01


class linearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_layer(x)


N = 10

D_in = 1
D_out = 1

X = torch.randn(N, D_in)
model = linearRegressionModel(D_in, D_out)
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.MSELoss()
tW = torch.tensor([[2.0]])
tb = torch.tensor(1.0)
y_true = X @ tW + tb + torch.randn(N, D_out) * 0.1
epoch = 1000
for i in range(epoch):
    y_hat = model(X)
    loss = loss_fn(y_hat, y_true)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if i % 10 == 0:
        print(f"epoch {i}: loss = {loss.item():.4f}")
