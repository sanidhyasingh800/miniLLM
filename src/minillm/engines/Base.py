from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # must assert request is a single query
    @abstractmethod
    def inference(self, request:list[str], max_new_tokens:int = 32) -> list[str]:
        pass

    @abstractmethod
    def batch_inference(self, requests: list[str],max_new_tokens: int = 32) -> list[str]:
        pass
