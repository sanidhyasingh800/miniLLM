import torch
from abc import ABC, abstractmethod
from minillm.schemas import Request, InferenceResponse

class BaseEngine(ABC):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def synchronize_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


    @abstractmethod
    def inference(self, requests: list[Request],max_new_tokens: int = 32) -> InferenceResponse:
        pass
