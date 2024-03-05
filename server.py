import flwr as fl
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

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



fl.server.start_server( # Open server connection
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average # Global Weights
    ),
)