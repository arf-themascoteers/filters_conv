from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from my_conv_net import MyConvNet
import torch
import torch.nn as nn
import constants
import analyze


def train(mode="normal"):
    num_epochs = 3
    num_classes = 10
    learning_rate = 0.001
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root=constants.DATA_PATH, train=True, transform=trans, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=constants.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyConvNet()
    if mode != "normal":
        filters = model.conv.weight.data
        r = torch.rand(5, 5)
        filters[0, 0] = r
        filters[1, 0] = r

        if mode == "weight_same":

            p = torch.rand((10,36))

            model.fc1.weight.data[:, 0:36] = p
            model.fc1.weight.data[:, 36:72] = p


    analyze.analyze(model, mode, "before")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training...")
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if i==0:
                analyze.plot_fms(model, mode, epoch)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

    torch.save(model, constants.DEFAULT_MODEL_PATH)
    analyze.analyze(model, mode, "after")
    return model