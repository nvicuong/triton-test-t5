from transformers import AutoTokenizer
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
       
        self.tokenizer = AutoTokenizer.from_pretrained("/workspace/model/")

    def execute(self, requests):
        responses = []
        for request in requests:
            
            input_text = pb_utils.get_input_tensor_by_name(request, "input_text").as_numpy()

            print(input_text)
            print(input_text.shape)
            print(input_text[0])

            input_text = [text.decode("utf-8") for text in input_text] 

            print('input_text:')
            print(input_text)
            
            
            tokenized_output = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="np", max_length=128)

            input_ids = tokenized_output["input_ids"].astype(np.int32)
            attention_mask = tokenized_output["attention_mask"].astype(np.int32)

            output_tensor_ids = pb_utils.Tensor("input_ids", input_ids)
            output_tensor_attention = pb_utils.Tensor("attention_mask", attention_mask)

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor_ids, output_tensor_attention])
            
            responses.append(response)
        
        return responses
