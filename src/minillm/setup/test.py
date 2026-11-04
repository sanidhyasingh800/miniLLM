import torch

## batched,and cached hugging face implementation of batched inference for correctness 
def test(
    model,
    tokenizer,
    device,
    requests: str | list[str],
    max_new_tokens: int = 32
) -> list[str]:

    # Configure padding for decoder-only generation
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        requests,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )