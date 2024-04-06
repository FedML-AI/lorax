#! /bin/bash

SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
cd "${SCRIPT_PATH}" || exit
docker build -t predibase/lorax:fedml .
cd - || exit