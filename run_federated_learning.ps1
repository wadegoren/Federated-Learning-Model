# To RUN: PowerShell.exe -ExecutionPolicy Bypass -File .\run_federated_learning.ps1
# Define the paths
$server_script_path = "C:\Users\wadeg\capstone\Federated-Learning-Model-WadeBranch\server.py"
$client_script_path = "C:\Users\wadeg\capstone\Federated-Learning-Model-WadeBranch\client.py"
$venv_dir = "C:\Users\wadeg\capstone\.venv"

# Start the server
Write-Output "Starting the federated learning server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$venv_dir'; .\Scripts\activate; pip install pyOpenSSL scikit-learn flwr torch torchvision; Start-Sleep -Seconds 5; python '$server_script_path'"

# Sleep for a short while to ensure the server starts properly
Start-Sleep -Seconds 15

# Start 10 client instances
Write-Output "Starting 10 client instances..."
for ($i = 1; $i -le 10; $i++) {
    Write-Output "Starting client $i..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$venv_dir'; .\Scripts\activate; pip install pyOpenSSL scikit-learn flwr torch torchvision matplotlib; Start-Sleep -Seconds 5; python '$client_script_path' $i"
    Start-Sleep -Seconds 1  # Adjust the sleep time if needed to ensure clients start properly
}

Write-Output "All clients started."
