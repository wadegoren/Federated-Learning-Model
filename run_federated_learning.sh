# Start the server
echo "Starting the federated learning server..."
python server.py &

# Sleep for a short while to ensure the server starts properly
sleep 5

# Start 10 client instances
echo "Starting 10 client instances..."
for ((i=0; i<9; i++))
do
    echo "Starting client $i..."
    python client.py $i 10 &
    sleep 1  # Adjust the sleep time if needed to ensure clients start properly
done

echo "All clients started."
