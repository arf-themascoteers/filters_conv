import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,3, (28,28), bias=False),
            nn.Flatten(),
            nn.Linear(3, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)


def plot_tensor(tensor,filename):
    mean = torch.mean(tensor)
    tensor = tensor.data.clone()
    tensor[tensor >= mean] = 255
    tensor[tensor < mean] = 0
    plt.imshow(tensor.numpy(), cmap="hot")
    plt.savefig(filename)


def plot_filters(filters, prefix):
    filters = filters.clone()
    filters = filters.reshape(filters.shape[0], filters.shape[2], filters.shape[3])
    for index,tensor in enumerate(filters):
        plot_tensor(tensor, f"{prefix}_{index}.png")


def train(model, data):
    NUM_EPOCHS = 1000
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch  in range(0, NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, torch.tensor([0]))
        loss.backward()
        optimizer.step()
        #print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    return model


data = cv2.imread("4.png")
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
data = torch.tensor(data, dtype=torch.float)
plot_tensor(data,"main.png")
data = data.reshape(1,1,data.shape[0],data.shape[1])
model = SimpleNet()
filters = model.net[0].weight.data

predefined = torch.rand((28,28))
model.net[0].weight.data[0,0]=predefined
model.net[0].weight.data[1,0]=predefined
model.net[0].weight.data[2,0]=predefined

p = model.net[2].weight.data[:,2]*2

model.net[2].weight.data[:,0] = p.detach().clone()
model.net[2].weight.data[:,1] = p.detach().clone()
model.net[2].weight.data[:,2] = p.detach().clone()

print(model.net[2].weight.data)

plot_filters(filters,"before")
train(model, data)

print(model.net[2].weight.data)

filters = model.net[0].weight.data
plot_filters(filters,"after")




