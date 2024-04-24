FROM ghcr.io/predibase/lorax:67157a9

# Override parent's image entrypoint.
ENTRYPOINT []

WORKDIR ./
RUN pip3 install fedml

COPY ./main.py main.py
COPY ./main.sh main.sh
COPY ./hf_token.txt /root/.cache/huggingface/token

RUN chmod +x /usr/src/main.sh

ENV LORA_BASE_MODEL_ID=""
ENV LORA_PREDOWNLOAD_ADAPTERS=""

ENTRYPOINT ["./main.sh"]
