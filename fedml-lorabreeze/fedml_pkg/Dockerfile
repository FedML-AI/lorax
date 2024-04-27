FROM ghcr.io/predibase/lorax:67157a9

# Override parent's image entrypoint.
ENTRYPOINT []

WORKDIR ./
RUN pip3 install fedml

# Copy required configuration files.
COPY ./main.py main.py
COPY ./main.sh main.sh
COPY ./hf_token.txt /root/.cache/huggingface/token

# Make executable.
RUN chmod +x /usr/src/main.sh

# We use /bin/bash to execute environmental variable substitution.
ENTRYPOINT [ "/bin/bash", "-c", "exec ./main.sh" ]
