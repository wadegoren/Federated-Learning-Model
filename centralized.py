import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import sklearn

from sklearn.model_selection import train_test_split
import numpy as np

import datetime
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

def load_data():
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# def load_data(i):
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10("./data", train=True, download=True, transform=trf)
#     testset = CIFAR10("./data", train=False, download=True, transform=trf)
    
#     # Get unique classes in the dataset
#     classes = trainset.classes
    
#     # Calculate number of classes per partition
#     classes_per_partition = len(classes) // 5  # Assuming you want to divide into 5 partitions
    
#     # Calculate start and end index of classes for this partition
#     start_idx = i * classes_per_partition
#     end_idx = (i + 1) * classes_per_partition if i < 4 else len(classes)
    
#     # Select classes for this partition
#     selected_classes = classes[start_idx:end_idx]
    
#     # Filter samples based on selected classes
#     train_indices = [idx for idx, (_, label) in enumerate(trainset) if label in selected_classes]
#     test_indices = [idx for idx, (_, label) in enumerate(testset) if label in selected_classes]
    
#     # Subset datasets based on selected indices
#     train_subset = torch.utils.data.Subset(trainset, train_indices)
#     test_subset = torch.utils.data.Subset(testset, test_indices)
    
#     # Create dataloaders for train and test subsets
#     trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
#     testloader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
#     return trainloader, testloader



def load_model():
    return Net().to(DEVICE)










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


def split_dataset_by_class(X, y, x):
    """
    Split the dataset into x mini-datasets, each containing samples from a single class.

    Parameters:
    - X: Features of the dataset.
    - y: Labels of the dataset.
    - x: Number of mini-datasets.

    Returns:
    - List of mini-datasets, where each mini-dataset is a tuple (X_mini, y_mini).
    """
    # Get unique classes in the dataset
    unique_classes = np.unique(y)

    # Initialize list to store mini-datasets
    mini_datasets = []

    # Iterate over each class and split the dataset
    for class_label in unique_classes:
        # Filter samples of the current class
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]
        y_class = y[class_indices]

        # Split the class into x mini-datasets
        X_mini_datasets, _, y_mini_datasets, _ = train_test_split(
            X_class, y_class, test_size=1/x, stratify=y_class, random_state=42
        )

        # Add each mini-dataset to the list
        for i in range(x):
            mini_datasets.append((X_mini_datasets[i], y_mini_datasets[i]))

    return mini_datasets

if __name__ == "__main__":
    # Example usage:
    num_sets = 5  # Number of prime pairs to generate
    prime_sets = generate_prime_pairs(num_sets)

    n_list = []

    if prime_sets:
        for i, (p, q, n) in enumerate(prime_sets):
            print(f"Set {i + 1} - p: {p}, q: {q}, n: {n}")
            n_list.append(n)
    else:
        print("Error, did not share n and e values")

    print(f"The number of sets is {len(prime_sets)}")

    same_n = True
    for i in range(len(n_list)-1):
        if n_list[i] != n_list[i+1]:
            same_n = False

    if same_n:
        print("The n's are all the same!")
    else:
        print("The n's are not the same!")
    


    '''
    net = load_model()
    trainloader, testloader = load_data()
    train(net, trainloader, 5)
    loss, accuracy = test(net, testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
    '''
