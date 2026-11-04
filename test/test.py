from minillm.setup import load_model
from minillm.engines import v0Engine
from minillm.setup import test
# a rudimentary test script testing against native huggingface model generation on same input
# using greedy decoding so randomness of output is not an issue




# test for v0 model
def v0_test_inference(model, tokenizer, device, request):
    # single request
    engine = v0Engine(model, tokenizer, device)
    max_new_tokens = 32
    response = engine.inference(request, max_new_tokens=max_new_tokens)
    assert(response == test(model, tokenizer, device, request, max_new_tokens = max_new_tokens))
    print(response)
    print("v0_single_inference_is_ok")


def v0_test_batch_inference(model, tokenizer, device, requests):
    # single request
    engine = v0Engine(model, tokenizer, device)
    max_new_tokens = 32
    response = engine.batch_inference(requests, max_new_tokens=max_new_tokens)
    assert(response == test(model, tokenizer, device, requests, max_new_tokens = max_new_tokens))
    print(response)
    print("v0_batch_inference_is_ok")


def main():
    model, tokenizer, device = load_model()
    single_request = ["Once upon a time"]
    requests = [
    "The quick brown fox jumps over the",
    "1, 2, 3, 4,",
    "Once upon a time, in a land far away, there lived a",
    "To write a program that prints 'Hello World' in Python, you use the following code:\n\n```python",
    "Question: What is the capital of France?\nAnswer:"
    ]
    v0_test_inference(model, tokenizer, device, single_request)
    v0_test_batch_inference(model, tokenizer, device, requests)


if __name__ == "__main__":
    main()