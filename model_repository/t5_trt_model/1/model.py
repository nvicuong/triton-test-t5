import torch
import numpy as np
from transformers import T5ForConditionalGeneration
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained("/workspace/model/").to(self.device)
        self.model.eval() 

    def execute(self, requests):
        responses = []
        for request in requests:
            
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()

            
            input_tensor = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            attention_tensor = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

            
            with torch.no_grad():
                output_ids = self.model.generate(
                            input_ids=input_tensor,
                            attention_mask=attention_tensor,
                            max_length=256,
                            num_beams=5,
                            temperature=0.7,
                            top_p=0.95,
                            top_k=50,
                            no_repeat_ngram_size=2,
                            early_stopping=True
                )

            
            output_ids_numpy = output_ids.cpu().numpy()

            output_ids_numpy = output_ids_numpy.astype(np.int32)

            
            output_tensor = pb_utils.Tensor("output_ids", output_ids_numpy)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses

    def finalize(self):
        pass
