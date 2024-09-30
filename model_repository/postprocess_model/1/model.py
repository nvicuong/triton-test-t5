from transformers import AutoTokenizer
import numpy as np
import triton_python_backend_utils as pb_utils

import os

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("/workspace/model/")

    def execute(self, requests):
        responses = []
        for request in requests:
            
            input_ids = pb_utils.get_input_tensor_by_name(request, "output_ids").as_numpy()
            
            decoded_outputs = [self.tokenizer.decode(id, skip_special_tokens=True) for id in input_ids]

            output_array = np.array(decoded_outputs, dtype=object)

            output_tensor = pb_utils.Tensor("output_text", output_array)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses
