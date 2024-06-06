
# Develop

## Build Image
First you need to run the build script as: ``` bash ./build.sh```.

## Run Container
The command below will deploy a FEDML(LoRAX) server with the Zephyr model (w/ quantization) and 5 pre-loaded adapters. 
The format for passing the adapters is `"adapter_source_1:adapter_1|adapter_source_2:adapter_2"`, 
e.g., `hub:ernest/redline_v3_adapter_864|hub:ernest/intent_redline_v3_adapter_2064`.
The adapter_source_id and the adapter_id are separated using semicolon (:), and adapters are separated using colon (|). 

> NOTE: the fedml service is listening to port `9999`.

> NOTE: please create the `$PWD/../data` if you have not done already.

> PRO TIP: to debug the lorax service we pass the `RUST_BACKTRACE=full` environment variable.  

```bash
docker run \
  --detach \
  --gpus "device=7" \
  --shm-size 1g \
  -p 9999:2345 \
  -v $PWD/../data:/data \
  -e RUST_BACKTRACE=full \
  -e MODEL_ID="TheBloke/zephyr-7B-beta-AWQ" \
  -e QUANTIZE="awq" \
  -e ADAPTERS_PRELOADED="hub:ernest/redline_v3_adapter_864|hub:ernest/intent_redline_v3_adapter_2064|hub:ernest/redline_v3_adapter_864|hub:ernest/redline_v2_adapter_400|hub:ernest/redline_v1_adapter_676|hub:ernest/redline_v0_adapter_432" \
  lorabreeze
```

## Test API

> NOTE: For all requests below, please remember to change the `IP_ADDRESS` !!!

### Verify service readiness. First, Adapters need to be preloaded.
Please wait for 1-2 minutes till the base model is downloaded and warmed up and then execute the following commands.

```bash
export IP_ADDRESS=213.181.123.15
curl --location 'http://'${IP_ADDRESS}':9999/ready'
```


### LoRAX API (directly)
To test the lorax API directly, first login to the docker container:
```bash
docker exec -it <CONTAINER_ID> /bin/bash
```

and then run the following request - (this request will always be submitted to localhost):
```bash
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



### FEDML API 
```bash
export IP_ADDRESS=213.181.123.15
curl --location 'http://'${IP_ADDRESS}':9999/predict' \
--header 'Content-Type: application/json' \
--data '{ 
    "stream": true, 
    "messages": [{ 
        "role": "user", 
        "content": "Say this is a test" 
    }], 
    "max_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.7,
    "adapter_id": "ernest/redline_v0_adapter_432"
}'
````

