# Client.py
from centralized import train, test, load_data, load_model
import flwr as fl
import torch
from collections import OrderedDict
import sys

def set_parameters(model, parameters): # Utility function to set parameters of model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True) # Model should always have same dimensions 
    return model

net = load_model()
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient): # Update weights of client model
    def __init__(self, trainloader, classes):
        super().__init__()
        self.trainloader = trainloader
        self.classes = classes

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, self.trainloader, epochs=1)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}

def partition_data(trainloader, classes):
    partitioned_data = []
    for images, labels in trainloader:
        for cls in classes:
            cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
            cls_images = images[cls_indices]
            cls_labels = labels[cls_indices]
            partitioned_data.append((cls_images, cls_labels))
    return partitioned_data

# Assuming each client is responsible for one classes
classes_per_client = 1
num_clients = 9
clients_data = []
for i in range(num_clients):
    classes = list(range(i * classes_per_client, (i + 1) * classes_per_client))
    partitioned_data = partition_data(trainloader, classes)
    clients_data.append(partitioned_data)

clients = []
for data in clients_data:
    client = FlowerClient(data, classes)
    clients.append(client)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client,
)

