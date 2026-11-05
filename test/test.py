from minillm.setup import load_model
from minillm.engines import v0Engine
from minillm.setup import test
from minillm.schemas import Request, InferenceResponse
# a rudimentary test script testing against native huggingface model generation on same input
# using greedy decoding so randomness of output is not an issue




# test for v0 model
def v0_test_inference(model, tokenizer, device, requests):
    # single request
    engine = v0Engine(model, tokenizer, device)
    max_new_tokens = 64
    response = engine.inference(requests, max_new_tokens=max_new_tokens)
    assert(response_match(model, tokenizer, device, max_new_tokens, response))
    for output in response.requests:
        print(output.response)
    print("v0_inference_is_ok")
    return response



def response_match(model, tokenizer, device, max_new_tokens, response: InferenceResponse):
    for request in response.requests:
        expected = test(model,
                        tokenizer,
                        device, 
                        request.prompt, 
                        max_new_tokens= max_new_tokens
                        )
        if expected[0] != request.response:
            return False
    return True

def main():
    model, tokenizer, device = load_model()
    request1 = Request("Once upon a time")
    request2 = Request("The quick brown fox jumps over the")
    response = v0_test_inference(model, tokenizer, device, [request1, request2])
    print(response)


if __name__ == "__main__":
    main()