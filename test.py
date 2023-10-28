from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import constants
import torch


def test(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = torch.load(constants.DEFAULT_MODEL_PATH)
    model = model.to(device)
    model.eval()
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST(root=constants.DATA_PATH, train=False, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=constants.batch_size, shuffle=False)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Test accuracy of the model on the 10000 test images: {}%".format(
            correct / total * 100
        ))


