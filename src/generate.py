import argparse
import os
import pandas as pd
import torch
from models import init_model
from tqdm import trange
from transformers import AutoConfig
from utils import dict_to_device
from watermarks import WATERMARKS

STOPWORDS = {
    "human-eval": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"],
    "none": [],
}

CHAT_TEMPLATES = {
    "vicuna": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}",
    "": None,
}



@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--watermark", choices=WATERMARKS.keys(), default="none")
    parser.add_argument("--prompt", default="data/pile-sample.jsonl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stopwords", choices=STOPWORDS.keys(), default="none")
    parser.add_argument("--disable_eos", action="store_true")
    parser.add_argument("--chat_template", choices=CHAT_TEMPLATES.keys(), default="")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stopwords = STOPWORDS[args.stopwords]
    chat_template = CHAT_TEMPLATES[args.chat_template]

    tokenizer, model = init_model(args.model)
    config = AutoConfig.from_pretrained(args.model)

    vocab_size = tokenizer.vocab_size
    model_max_length = config.max_position_embeddings
    model.eval()

    df = pd.read_json(args.prompt, lines=True)

    if args.debug:
        df = df.sample(32)

    prefix_all = df["prompt"].to_list()
    completion_all = []

    watermark_alg = WATERMARKS[args.watermark](vocab_size=vocab_size)
    for sample_idx in trange(0, len(prefix_all), args.batch_size):
        prefixes = prefix_all[sample_idx : sample_idx + args.batch_size]

        if isinstance(prefixes[0], list):
            prefixes_chat = [
                tokenizer.apply_chat_template(p, tokenize=False, chat_template=chat_template, add_generation_prompt=True)
                for p in prefixes
            ]
            encoded = dict_to_device(tokenizer(prefixes_chat, padding=True, return_tensors="pt", add_special_tokens=False), model.device)
        
        else:
            assert isinstance(prefixes[0], str)
            encoded = dict_to_device(
                tokenizer(prefixes, padding=True, return_tensors="pt"), model.device
            )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        past_key_values = None
        unfinished_sequence = torch.ones(
            (len(prefixes), 1), device=model.device, dtype=torch.bool
        )

        for _ in range(min(args.max_new_tokens, model_max_length - input_ids.shape[1])):
            inputs = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            outputs = model(**inputs)
            logits = outputs.logits[..., :vocab_size]
            past_key_values = outputs.past_key_values

            if args.disable_eos:
                assert tokenizer.eos_token_id is not None
                logits[:, :, tokenizer.eos_token_id] = -100

            next_tokens = torch.stack(
                [
                    watermark_alg.watermark(
                        input_ids[i][attention_mask[i].bool()].tolist(), logits[i, -1]
                    )
                    for i in range(len(prefixes))
                ]
            )
            unfinished_sequence = unfinished_sequence & (
                next_tokens != tokenizer.eos_token_id
            )

            input_ids = torch.concat(
                (
                    input_ids,
                    torch.where(
                        unfinished_sequence, next_tokens, tokenizer.eos_token_id
                    ),
                ),
                dim=1,
            )
            attention_mask = torch.cat(
                (attention_mask, unfinished_sequence.float()), dim=1
            )

            if not torch.any(unfinished_sequence):
                break

        prompts = tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)
        prompts_with_completions = tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        for p, pc in zip(prompts, prompts_with_completions):
            assert pc.startswith(p)
            c = pc[len(p) :]
            for stopword in stopwords:
                c = c.split(stopword)[0]
            completion_all.append(c)

    df["completion"] = completion_all
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    df.to_json(
        os.path.join(args.output_dir, f"{model_name}-{args.watermark}.jsonl"),
        lines=True,
        orient="records",
    )


if __name__ == "__main__":
    main()
