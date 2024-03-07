#!/bin/bash

# Define the path to the Python scripts
client_script_path="/Users/matthewmaceachern/Downloads/Federated-Learning-Model/client.py"
server_script_path="/Users/matthewmaceachern/Downloads/Federated-Learning-Model/server.py"
venv_dir="/Users/matthewmaceachern/Downloads/Federated-Learning-Model/venv"

# Run the server.py script
osascript -e 'tell application "Terminal" to do script "source '"$venv_dir"'/bin/activate && pip install pyOpenSSL scikit-learn flwr torch torchvision && sleep 5 && python '"$server_script_path"'"'

# Wait for a brief moment before starting the clients
sleep 15

# Run the client.py script five times
for ((i=1; i<=5; i++))
do
    # osascript -e 'tell application "Terminal" to do script "source '"$venv_dir"'/bin/activate && pip install pyOpenSSL scikit-learn flwr torch torchvision && sleep 5 && python '"$client_script_path"'"'
    osascript -e 'tell application "Terminal" to do script "source '"$venv_dir"'/bin/activate && pip install pyOpenSSL scikit-learn flwr torch torchvision matplotlib && sleep 5 && python '"$client_script_path"' '"$i"'"'
done
