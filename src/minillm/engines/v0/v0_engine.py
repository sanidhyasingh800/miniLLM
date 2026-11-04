import torch
from minillm.engines import BaseEngine

class v0Engine(BaseEngine):

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def inference(self, request, max_new_tokens = 32):
        # inference on a single request enforced
        assert len(request) == 1
        request = request[0]
        inputs = self.tokenizer(request, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.inference_mode():
            #autoregressively generate next token until max or eos is reached
            for i in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids, 
                                attention_mask = attention_mask, 
                                use_cache = False
                                )
                next_token_id = torch.argmax(outputs.logits[:,-1,:],
                                            dim=-1, 
                                            keepdim=True
                                            )
                input_ids = torch.cat([input_ids, next_token_id], dim = 1)
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim =1)

        return [self.tokenizer.decode(input_ids[0], skip_special_tokens = True)]

    def batch_inference(self, requests: list[str], max_new_tokens: int = 32) -> list[str]:
        outputs = []
        for request in requests:
            outputs.extend(self.inference([request], max_new_tokens = max_new_tokens))

        return outputs
        

