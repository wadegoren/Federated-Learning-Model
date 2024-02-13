from centralized import train, test, load_data, load_model, Net
import flwr as fl
import torch
from collections import OrderedDict
from torch.optim import lr_scheduler

def set_parameters(model, parameters): # Utility function to set parameters of model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True) # Model should always have same dimensions 
    return model

# net = load_model()
# trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient): # Update weights of client model
    def __init__(self, trainloader, testloader) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.testloader = testloader
        self.model = Net(num_classes=10)


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        optim = torch.optim.SGD(self.model.parameters(), lr=lr_scheduler, momentum=0.9)
        train(self.model, self.trainloader, optim, epochs=5)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),

)