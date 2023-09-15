import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def init_model(model_name: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", load_in_8bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    assert tokenizer.eos_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.eos_token
    return tokenizer, model
