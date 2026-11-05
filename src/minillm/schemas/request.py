

# a dataclass schema for tracking a single inference request into the model
# tracks the prompt, full response returned + additional metadata about timing and size


from dataclasses import dataclass


# prompt: initial prompt of the user
# response: full outputted response (including the prompt)
# t0: timestamp of when generation for this prompt started
# t1: timestamp of when first token was generated
# t2: timestamp of when the full output was generated (but not necc. returned to user)
# n: number of NEW tokens generated
@dataclass
class Request:
    prompt: str
    response: str = ""
    t0: int = 0
    t1: int = 0
    t2: int = 0
    n: int = 0
