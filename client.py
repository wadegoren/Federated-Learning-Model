from centralized import train, test, load_data, load_model
import flwr as fl
import torch
from collections import OrderedDict
import random
import cryptography

prime_list = [53, 61]

def gcd(a, b):
    #Calculates the greatest common divisor 
    while b:
        a, b = b, a % b
    return a

def set_parameters(model, parameters): # Utility function to set parameters of model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True) # Model should always have same dimensions 
    return model

net = load_model()
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient): # Update weights of client model
    def __init__(self):
        # Necessary Variables (calculated at the start only): p, q, n, phi, e, d
        # Public Key: n, e                         Private Key: n, d 

        # p and q will be the prime numbers chosen for this specific client model
        # n = p * q and phi = (p-1) * (q-1)  
        self.p = random.choice(prime_list)
        while (self.q != self.p and self.q != 0):
            self.q = random.choice(prime_list)
        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)

        # e must be 1 < e < phi and e must be coprime to phi
        while gcd(self.e, self.phi) != 1:
            self.e = random.randrange(3, self.phi, 2)

        # d must be 1 < d < phi and (d * e) % phi = 1
        while ((self.d * self.e) % self.phi) != 1:
            self.d = random.randrange(2, self.phi, 1)

    def encrypt(self, M):
        #For now, unencrypted weights (plaintext message) will be assumed to be in a tensor called M
        #This is assumed to be a single int for now

        #Return the ciphertext using our primes
        return (M ** self.e) % self.n

    def decrypt(self, C):
        #For now, encrypted weights (cyphertext) will be assumed to be in a tensor called C
        #This is assumed to be a single int for now

        #Return the plaintext using our primes
        return (C ** self.d) % self.n

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),

)