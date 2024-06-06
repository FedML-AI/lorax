#!/bin/bash

# Here, we shall check if the environmental variables of the default
# LoRAX service are passed during the container's initialization.
# For a full list of all possible environmental variables, please see
# the Launcher Markdown file: `lorax/docs/reference/launcher.md`
# Check if the environment variables are set. If they are export them.
if [ -n "${MODEL_ID}" ]; then
    echo "Environment variable MODEL_ID is set"
    export MODEL_ID="${MODEL_ID}"
fi

if [ -n "${QUANTIZE}" ]; then
    echo "Environment variable QUANTIZE is set"
    export QUANTIZE="${QUANTIZE}"
fi

# Start the first process.
nohup python3 /usr/src/main.py > /tmp/fedml_lorabreeze.txt & 2>&1

# Print all given arguments (if any).
echo "Given command line arguments: " "$@"

# Start LoRAX server, passing all the environmental
# variables down to the server.
lorax-launcher "$@"

# Wait for any process to exit.
wait -n

# Exit with status of process that exited first.
exit $?