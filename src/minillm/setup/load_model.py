import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai-community/gpt2"


def load_model(device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def main():
    model, tokenizer, device = load_model()
    # Test run for checking model state is valid
    prompt = "if dogs say bark, then cats say"
    inputs = tokenizer(prompt, return_tensors="pt").to(device) # return the list of input tokens as tensor
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) # attention mask tells the model which words to look at and which are padding

    #  we execute one forward pass, without any kv caching, inference_mode just doesnt compute any gradients
    with torch.inference_mode():
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, use_cache = False)

    logits = outputs.logits

    #  greedy decode next-token prediction
    next_token_id = torch.argmax(logits[0, -1, :]).item()
    next_token = tokenizer.decode([next_token_id])

    print("Test Completed:", prompt + next_token)



if __name__ == "__main__":
    main()