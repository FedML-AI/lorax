#!/bin/bash

# Start the first process
python3 /usr/src/main.py &

# Start the second process, which get --model-id with a value
echo "$@"

lorax-launcher "$@"

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?