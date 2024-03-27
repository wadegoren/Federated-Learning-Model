from centralized import train, load_model, test
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


data_path = '/Users/matthewmaceachern/Downloads/Federated-Learning-Model/Data'
test_path = '/Users/matthewmaceachern/Downloads/Federated-Learning-Model/Data/test'

def load_data2(path, test_path): 
    device = torch.device("cpu")
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
    'test': transforms.Compose([
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


    # Create custom datasets for training and validation sets
    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                            data_transforms[x])
                    for x in ['train']}
    
    test_dataset = datasets.ImageFolder(test_path, data_transforms['test'])

    # Add all datasets to loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                    for x in ['train']}
    
    train_dataset = datasets.ImageFolder(os.path.join(path, 'train'), data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    print("DATASET SIZES: ", dataset_sizes)
    return trainloader, testloader


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
        train(net, trainloader, epochs=2)
        return self.get_parameters({}), len(trainloader.dataset), {}   

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        if (self.client_number == 1 and self.glob == 1):
            print("GLOBAL ACCRUACY: ", accuracy)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    i = int(sys.argv[1])
    print("****************")
    print(i)
    print("****************")
    trainloader, testloader = load_data2(data_path, test_path=test_path)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_number=i),
    )
