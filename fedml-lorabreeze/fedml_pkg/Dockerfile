FROM ghcr.io/predibase/lorax:bd7db80

# Override parent's image entrypoint.
ENTRYPOINT []

WORKDIR ./
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/FedML-AI/FedML.git && cd FedML/python && git checkout dev/v0.7.0 && pip install -e .

# Copy required configuration files.
COPY ./main.py main.py
COPY ./main.sh main.sh
COPY ./hf_token.txt /root/.cache/huggingface/token

# Make executable.
RUN chmod +x /usr/src/main.sh

# We use /bin/bash to execute environmental variable substitution.
ENTRYPOINT [ "/bin/bash", "-c", "exec ./main.sh" ]
