
# Develop

## Build Image
First you need to run the build script as: ``` bash ./build.sh```.

## Run Container
The command below will deploy a FEDML(LoRAX) server with the Zephyr model (w/ quantization) and 5 pre-loaded adapters. 
The format for passing the adapters is `"adapter_source_1:adapter_1|adapter_source_2:adapter_2"`, 
e.g., `hub:ernest/redline_v3_adapter_864|hub:ernest/intent_redline_v3_adapter_2064`.
The adapter_source_id and the adapter_id are separated using semicolon (:), and adapters are separated using colon (|). 

> NOTE: the fedml service is listening to port `9999` !!!

```bash
docker run \
  --detach \
  --gpus "device=3" \
  --shm-size 1g \
  -p 9999:2345 \
  -v $PWD/../data:/data \
  -e MODEL_ID="TheBloke/zephyr-7B-beta-AWQ" \
  -e QUANTIZE="awq" \
  -e ADAPTERS_PRELOADED="hub:ernest/redline_v3_adapter_864|hub:ernest/intent_redline_v3_adapter_2064|hub:ernest/redline_v3_adapter_864|hub:ernest/redline_v2_adapter_400|hub:ernest/redline_v1_adapter_676|hub:ernest/redline_v0_adapter_432" \
  fedml_lorabreeze
```

## Test API
### Preload Adapters
Wait for 1-2 minutes till the base model is downloaded and warmed up and then execute the following commands.
> Remember to change the `IP_ADDRESS` !!!
```bash
export IP_ADDRESS=213.181.123.15
curl --location 'http://'${IP_ADDRESS}':9999/ready'
```

### Run Inference
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
```