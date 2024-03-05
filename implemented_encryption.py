import torch
import flwr as fl
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

# Function to generate prime pairs
def generate_prime_pairs(num_sets, public_exponent=65537, key_size=2048):
    prime_sets = []
    same_n_and_e = False
    num_rounds = 0

    while not same_n_and_e:
        num_rounds += 1
        print(f"Starting round {num_rounds}")

        # Generate a single random prime number for p
        p = rsa.generate_private_key(
            public_exponent=public_exponent,
            key_size=key_size,
            backend=default_backend()
        ).public_key().public_numbers().n

        # Generate a random prime number for q, making sure it's different from p
        while True:
            q = rsa.generate_private_key(
                public_exponent=public_exponent,
                key_size=key_size,
                backend=default_backend()
            ).public_key().public_numbers().n
            
            if p != q:
                break

        # Calculate n
        n = p * q

        # Append the prime pair (p, q) and n to the list for the specified number of sets
        for _ in range(num_sets):
            prime_sets.append((p, q, n))

        # Ensure that all sets have the same n and public exponent
        same_n_and_e = all(set_[2:] == prime_sets[0][2:] for set_ in prime_sets)

    print(f"The number of sets until correct key was generated is {num_rounds}")
    return prime_sets

# Sample model for FL
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Flower client
class PrimePairClient(fl.client.NumPyClient):
    def __init__(self, model, prime_pairs):
        self.model = model
        self.prime_pairs = prime_pairs

    def get_parameters(self):
        return self.model.state_dict()

    def fit(self, parameters, config):
        # Update the model parameters with received parameters
        self.model.load_state_dict(parameters)
        
        # Perform local training (in a real FL scenario, you would use your training data)
        # ...

        # Return updated parameters and loss (for demonstration, using a dummy loss)
        return self.model.state_dict(), 0.1

    def evaluate(self, parameters, config):
        # Evaluate the model on local test data (in a real FL scenario, you would use your test data)
        # ...

        # Return evaluation result (for demonstration, using a dummy accuracy)
        return {"accuracy": 0.95}

# Flower server
class PrimePairServer(fl.server.Server):
    def __init__(self, model, prime_pairs):
        self.model = model
        self.prime_pairs = prime_pairs

    def get_parameters(self):
        return self.model.state_dict()

    def fit(self, ins):
        # Aggregate model parameters from all clients
        aggregated_params = fl.server.simple.average_models(ins)
        self.model.load_state_dict(aggregated_params)

    def evaluate(self, ins):
        # Aggregate evaluation results from all clients
        aggregated_results = fl.server.simple.sum_values(ins)
        return {"accuracy": aggregated_results / len(ins)}

# Main function
def main():
    # Generate prime pairs for the specified number of sets
    num_clients = 5
    exponent=65537
    prime_pairs = generate_prime_pairs(num_sets=num_clients,public_exponent=exponent)

    # Create a simple PyTorch model
    model = SimpleModel()

    # Create Flower clients, these will all get their own prime pairs that are generated above
    # These will be used to make a private/public key
    for i in range(len(prime_pairs)):
        client = PrimePairClient(model, prime_pairs[i])

    n = prime_pairs[0].n

    # The server will get the common n and common exponent that will be necessary for aggregating the weights
    # Enc(A) + Enc(B) = Enc(A + B), Enc(A) * Enc(B) = Enc(A * B), therefore Avg(Enc(A), Enc(B)) = Enc(Avg(A,B))
    server = PrimePairServer(model, n, exponent)

    # Start the Flower server
    fl.server.start_server("localhost:8080", server)

    # Start the Flower client
    fl.client.start_numpy_client("localhost:8080", client)

if __name__ == "__main__":
    main()
