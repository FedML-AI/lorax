#! /bin/bash

SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
cd ${SCRIPT_PATH}
volume=$PWD/../data

# Model 1
# model=TheBloke/zephyr-7B-beta-AWQ
# docker run --detach --gpus "device=0" --shm-size 1g -p 8889:80 -v $volume:/data predibase/lorax:fedml --model-id $model --quantize awq

# Model 2
model=ernest/zephyr_7b_beta_bnb_int4
docker run --detach --gpus "device=3" --shm-size 1g -p 8889:80 -v $volume:/data predibase/lorax:fedml --model-id $model --trust-remote-code

cd -
