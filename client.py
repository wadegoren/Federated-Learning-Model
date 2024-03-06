from centralized import train, test, load_data, load_model
import flwr as fl
import torch
import sys
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from collections import OrderedDict
NUM_CLIENTS = 10

def set_parameters(model, parameters): 
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def load_data(i_str):
    i = int(i_str)  # Convert string to integer
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    
    num_classes = len(trainset.classes)
    classes_per_partition = num_classes // NUM_CLIENTS

    start_idx = (i - 1) * classes_per_partition
    end_idx = i * classes_per_partition if i < 10 else num_classes

    selected_classes_indices = list(range(start_idx, end_idx))
    print("Selectedf classes indicies: ", selected_classes_indices)
    selected_classes_names = [trainset.classes[idx] for idx in selected_classes_indices]  # Get class names
    print("Selected classes names:", selected_classes_names)


    train_indices = [idx for idx, (_, label) in enumerate(trainset) if label in selected_classes_indices]
    # print("TRain indicies: ", train_indices)
    test_indices = [idx for idx, (_, label) in enumerate(testset) if label in selected_classes_indices]
    # print("test indicies: ", test_indices)


    train_subset = torch.utils.data.Subset(trainset, train_indices)
    test_subset = torch.utils.data.Subset(testset, test_indices)

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)

    print("Number of samples in trainset:", len(trainset))
    print("Number of samples in testset:", len(testset))
    print("Number of train indices:", len(train_indices))
    print("Number of test indices:", len(test_indices))

    # train_subset = torch.utils.data.Subset(trainset, train_indices)
    # test_subset = torch.utils.data.Subset(testset, test_indices)

    print("Number of samples in train subset:", len(train_subset))
    print("Number of samples in test subset:", len(test_subset))

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)

    return trainloader, testloader

net = load_model()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_number):
        super.__init__(self)
        self.client_number = client_number
        self.glob = 0 #Variable to check to make sure that the current model is the global model

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        if (self.client_number == 1):
            self.glob = 1
            print("TESTING GLOBAL MODEL\n")
            self.evaluate(parameters, config)
            self.glob = 0
        set_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        if (self.client_number == 1 and self.glob == 1):
            print("GLOBAL MODEL ACCURACY: ", accuracy)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    i = int(sys.argv[1])
    print("****************")
    print(i)
    print("****************")
    trainloader, testloader = load_data(i)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_number=i),
    )