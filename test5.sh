#!/bin/bash

# Function to run client.py with a specific test case in a new terminal
run_client() {
    i=$1
    open -a Terminal.app "./client.py"
}

# Function to run server.py in a new terminal
run_server() {
    open -a Terminal.app <<EOF
    echo "This is the Terminal window."
    ls
    cd /path/to/directory
    python3 script.py
EOF
}

# Run server.py in a separate terminal
echo "Running server.py in a separate terminal..."
run_server

# Run client.py in 5 different terminals with different test cases
echo "Running client.py in 5 different terminals with different test cases..."
for i in {1..5}
do
    echo "Terminal $i"
    run_client $i
done
