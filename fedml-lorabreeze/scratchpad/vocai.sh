#! /bin/bash

LORAX_IMAGE=predibase/lorax:fedml
SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
cd "${SCRIPT_PATH}" || exit
VOLUME=$PWD/../data

# Model 1
#MODEL=TheBloke/zephyr-7B-beta-AWQ
#docker run --detach --gpus "device=3" --shm-size 1g -p 8889:80 -v "$VOLUME":/data "$LORAX_IMAGE" --model-id "$MODEL" --quantize awq

# Model 2
#MODEL=ernest/zephyr_7b_beta_bnb_int4
MODEL=arnavgrg/zephyr-7b-beta-nf4-fp16-upscaled
docker run --detach --gpus "device=3" --shm-size 1g -p 8889:80 -v "$VOLUME":/data "$LORAX_IMAGE" --model-id "$MODEL" --quantize bitsandbytes-nf4 # --dtype bfloat16

cd - || exit

#docker run --detach --name fedmllorax_lorax --gpus "device=0" --shm-size 1g -v $volume:/data --network fedmllorax predibase/lorax:fedml --model-id $model --quantize awq
#docker run -it --name fedmllorax_fedml --gpus "device=3" --network fedmllorax fedml/fedml-default-inference-backend /bin/bash
