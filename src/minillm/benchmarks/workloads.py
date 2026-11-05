# workloads.py use the monolithic prompt dataset
# accept the number of inputs and prompt size
# returns the list of request objects 

import json
from pathlib import Path
from minillm.schemas import Request

def get_workload(tokenizer, num_prompts, prompt_size):
    if num_prompts < 1 or prompt_size < 1:
        raise ValueError("expected positive number of prompts and prompt size")
    
    path = Path(__file__).parent / "prompts.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        prompts = [
            json.loads(line)["prompt"]
            for line in f
            if line.strip()
    ]

    selected_prompts = [prompts[i % len(prompts)] for i in range(num_prompts)]

    # tokenize against prompt_size to verify and limit size to prompt_size
    tokenized = tokenizer(selected_prompts,
                          add_special_tokens = False,
                          truncation = True, 
                          max_length = prompt_size,
                          return_attention_mask = False
                          )["input_ids"]

    if any(len(sequence) != prompt_size for sequence in tokenized):
        return ValueError("Cannot generated prompts with given token size")

    # return truncated sequence back to text form for the model
    final_prompts = tokenizer.batch_decode(tokenized,
                                           skip_special_tokens = False,
                                           clean_up_tokenization_spaces = False
                                           )

    return [Request(prompt = text) for text in final_prompts]
    
