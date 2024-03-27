import flwr as fl
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import matplotlib.pyplot as plt


tot_acc =[]

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    tot_acc.append(sum(accuracies) / sum(examples))
    print("*********BIG SUM**********: ", tot_acc)
    return {"accuracy": sum(accuracies) / sum(examples)}

def plot_and_show_graph():
    plt.plot(range(len(tot_acc)), tot_acc)
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.title('Graph of Total Accuracy')
    
    # Set ticks on x-axis to be whole numbers
    plt.xticks(range(len(tot_acc)))
    
    plt.savefig('tot_acc_graph.png')
    plt.show()

fl.server.start_server( # Open server connection
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average # Global Weights
    ),
)

plot_and_show_graph()
