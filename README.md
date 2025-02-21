# Triton Inference Server Customization

## Overview

This repository provides an example of customizing a Python backend for the [Triton Inference Server](https://github.com/triton-inference-server/server). The implementation demonstrates how to modify the Triton server to support specific deep learning inference workflows.

## Features

- Custom Python model for Triton inference
- Preprocessing and postprocessing pipelines
- Optimized request handling
- Support for multiple model versions

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Docker
- NVIDIA Triton Inference Server (>=2.x)
- Python 3.8+
- add model artifacts to `model` directory
### Installation

Clone the repository:
```sh
   git clone https://github.com/nvicuong/triton-test-t5.git
   cd triton-test-t5
```

### Running the Triton Server

You can start the Triton server with the custom model by running:

```
docker build -t tritonserver-custom .

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/  -v ${PWD}/model_repository:/models tritonserver-custom

```

### Testing the Inference

You can send health check requests using `curl`:

```
curl --location 'http://localhost:8000/v2/health/ready'
```

And send inference requests:

```
curl --location 'http://112.137.129.161:8000/v2/models/ensemble_model/infer?ab=sd' \
--header 'Content-Type: application/json' \
--data '{
        "inputs": [
            {
                "name": "input_text",
                "shape": [1],  
                "datatype": "BYTES",
                "data": ["abc"]
            }
        ]
}'
```

### Modifying the Custom Model
You can edit `model.py` in the model repository to modify the inference logic. Ensure that your script follows the Triton Python backend model structure.

## I hope it's helpful for you
