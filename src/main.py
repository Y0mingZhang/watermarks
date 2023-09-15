import argparse
import os
import pandas as pd
import torch
from models import init_model
from tqdm import trange
from utils import dict_to_device
from watermarks import WATERMARKS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--watermark", choices=WATERMARKS.keys(), default="openai")
    parser.add_argument("--prefix", default="data/pile-sample.jsonl")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer, model = init_model(args.model)
    vocab_size = tokenizer.vocab_size
    assert args.max_new_tokens <= tokenizer.model_max_length
    model.eval()

    prefix_df = pd.read_json(args.prefix, lines=True)
    prefix_all = prefix_df["prefix"].to_list()
    completion_all = []

    if args.debug:
        prefix_all = prefix_all[:32]

    watermark_alg = WATERMARKS[args.watermark](vocab_size=vocab_size)
    for sample_idx in trange(0, len(prefix_all), args.batch_size):
        prefixes = prefix_all[sample_idx : sample_idx + args.batch_size]
        encoded = dict_to_device(
            tokenizer(prefixes, padding=True, return_tensors="pt"), model.device
        )

        input_ids = encoded["input_ids"]
        initial_seq_len = input_ids.shape[1]
        attention_mask = encoded["attention_mask"]
        past_key_values = None
        unfinished_sequence = torch.ones(
            (len(prefixes), 1), device=model.device, dtype=torch.bool
        )

        for _ in range(args.max_new_tokens):
            inputs = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            outputs = model(**inputs)
            logits = outputs.logits[..., :vocab_size]
            past_key_values = outputs.past_key_values

            watermarked_probs = torch.stack(
                [
                    watermark_alg.watermark(
                        input_ids[i][attention_mask[i].bool()].tolist(), logits[i, -1]
                    )
                    for i in range(len(prefixes))
                ]
            )
            next_tokens = torch.argmax(watermarked_probs, 1, keepdim=True)
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

        output_ids = input_ids[:, initial_seq_len:]
        completion = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        completion_all.extend(completion)

    df = pd.DataFrame(
        {
            "prefix": prefix_all,
            "completion": completion_all,
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    df.to_json(
        os.path.join(args.output_dir, f"{model_name}-{args.watermark}.jsonl"),
        lines=True,
        orient="records",
    )


if __name__ == "__main__":
    main()
