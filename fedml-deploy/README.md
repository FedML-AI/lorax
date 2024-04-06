# Setup

1. Run `bash devops/build.sh` to build the new image, `predibase/lorax:fedml`.
2. Run `bash devops/run.sh` to start the endpoint.

To validate that the endpoint is up and running, please run an inference query similar to the example below. Rememeber to change th ip address environmental variable.
```
IP_ADDRESS=38.101.196.134
curl ${IP_ADDRESS}:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
    -H 'Content-Type: application/json'
```