import torch
import time
from minillm.engines import BaseEngine
from minillm.schemas import Request, InferenceResponse

class v0Engine(BaseEngine):

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def single_inference(self, request: Request, max_new_tokens = 32):   

        request.n = 0
        request.t0 = time.perf_counter_ns()     

        prompt = request.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        ## pre-allocate a tensor for final result
        # this ensures that we do not have copy on cat in gpu mem
        buffer = torch.empty((1, prompt_len + max_new_tokens),
                             dtype = input_ids.dtype,
                             device = self.device)

        buffer[:, :prompt_len] = input_ids
        input_ids = buffer

        with torch.inference_mode():
            #autoregressively generate next token until max or eos is reached
            for i in range(max_new_tokens):
                current_len = prompt_len+i
                outputs = self.model(input_ids=input_ids[:, :current_len], 
                                use_cache = False,
                                logits_to_keep = 1
                                )
                next_token_id = torch.argmax(outputs.logits[:,-1,:],
                                            dim=-1, 
                                            keepdim=True
                                            )
                request.n += 1
                input_ids[:, current_len:current_len+1] = next_token_id
                if i == 0:
                    self.synchronize_device()
                    request.t1 = time.perf_counter_ns()
        self.synchronize_device()
        request.t2 = time.perf_counter_ns()
        request.response = self.tokenizer.decode(input_ids[0], skip_special_tokens = True)
        return request

    def inference(self, requests: list[Request], max_new_tokens: int = 32) -> InferenceResponse:
        self.synchronize_device()
        t_init = time.perf_counter_ns()
        outputs = []

        for request in requests:
            outputs.append(self.single_inference(request, max_new_tokens = max_new_tokens))
        self.synchronize_device()
        t_end = time.perf_counter_ns()
        response = InferenceResponse(outputs, t_init, t_end)
        return response
        