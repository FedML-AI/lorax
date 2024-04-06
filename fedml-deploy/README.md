# Setup

1. Run `bash devops/build.sh` to build the new image, `predibase/lorax:fedml`.
2. Make any changes regarding the endpoint's specifications in the `endpoint.yaml` file.
3. Run `python deploy.py` to start the endpoint.

To validate that the endpoint is up and running, please run an inference query similar to the example below. 
Remember to change the ip address of environmental variable (set to localhost) and the listening port (set to 8888).
```
IP_ADDRESS=127.0.0.1
curl ${IP_ADDRESS}:8888/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
    -H 'Content-Type: application/json'
```