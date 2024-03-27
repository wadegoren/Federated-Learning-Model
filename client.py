from centralized import train_model, test, load_data, load_model
import flwr as fl
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import sys
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from collections import OrderedDict
from torchvision.datasets import ImageFolder
import os
from torchvision import transforms
from torchvision.transforms import Resize
from torchvision import datasets, models, transforms
import torchvision


def set_parameters(model, parameters): 
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def load_data(i_str):
    device = torch.device("cpu")
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
}
    data_path = '/Users/matthewmaceachern/Downloads/Federated-Learning-Model/Data'

    # Create custom datasets for training and validation sets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                            data_transforms[x])
                    for x in ['train', 'valid']}

    # Add all datasets to loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                    for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print("DATASET SIZES: ", dataset_sizes)

    # fine tune the classification layers on ct scans
    model_conv = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')

    # uncomment below to perform only fine-tuned training
    # for param in model_conv.parameters():
    #     param.requires_grad = False

    # Only the new fc layers will be trained
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 4)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

    model_conv = train_model(model=model_conv, criterion=criterion, 
                            optimizer=optimizer_conv, scheduler=exp_lr_scheduler, 
                            num_epochs=25, dataloaders=dataloaders, 
                            dataset_sizes=dataset_sizes, device=device)


net = load_model()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_number):
        super().__init__()
        self.client_number = client_number
        self.glob = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        # if (self.client_number == 1):
        #     self.glob = 1
        #     print("\nTESTING GLOBAL MODEL\n")
        #     self.evaluate(parameters, config)
        #     self.glob = 0
        set_parameters(net, parameters)
        train(net, trainloader, epochs=20)
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        # loss, accuracy = test(net, testloader)
        # if (self.client_number == 1 and self.glob == 1):
        #     print("GLOBAL ACCRUACY: ", accuracy)
        # return float(loss), len(testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    i = int(sys.argv[1])
    print("****************")
    print(i)
    print("****************")
    trainloader = load_data(i)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_number=i),
    )
