FROM nvcr.io/nvidia/tritonserver:24.09-pyt-python-py3

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch transformers

CMD ["tritonserver", "--model-repository=/models"]
