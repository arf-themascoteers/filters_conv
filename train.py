from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from my_conv_net import MyConvNet
import torch
import torch.nn as nn
import constants


def train():
    num_epochs = 2
    num_classes = 10
    learning_rate = 0.001
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root=constants.DATA_PATH, train=True, transform=trans, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=constants.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyConvNet()


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
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%"\
                        .format(
                            epoch + 1,
                            num_epochs, i + 1,
                            total_step,
                            loss.item(),
                            (correct / total) * 100
                        )
                )
    torch.save(model, constants.DEFAULT_MODEL_PATH)
    return model