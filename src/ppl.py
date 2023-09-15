import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from models import init_model
from tqdm import trange


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/ppl")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-hf")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer, model = init_model(args.model)
    model.eval()

    assert args.window_size <= model.config.max_position_embeddings

    alias = os.path.basename(args.text_file).replace(".jsonl", "")
    df = pd.read_json(args.text_file, lines=True)
    prefixes = df["prefix"]
    completions = df["completion"]

    utf8_bytes = sum(len(s.encode("utf-8")) for s in completions)
    tokens = []
    completion_masks = []

    for prefix, completion in zip(prefixes, completions):
        iids_p = tokenizer(prefix)["input_ids"][:512]
        iids_c = tokenizer(completion, add_special_tokens=False)["input_ids"][:512]
        iids = iids_p + iids_c
        assert len(iids) <= args.window_size
        iids = torch.tensor(
            iids + [tokenizer.pad_token_id] * (args.window_size - len(iids)),
            dtype=torch.long,
        )
        completion_mask = torch.zeros_like(iids, dtype=torch.float)
        completion_mask[len(iids_p) : len(iids_p) + len(iids_c)] = 1.0
        tokens.append(iids)
        completion_masks.append(completion_mask)

    tokens = torch.stack(tokens, 0)
    completion_masks = torch.stack(completion_masks, 0)

    nlls = []
    token_counts = []

    for offset in trange(0, len(tokens), args.batch_size):
        input_ids = tokens[offset : offset + args.batch_size].to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).float()
        completion_mask = completion_masks[offset : offset + args.batch_size].to(
            model.device
        )
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = rearrange(
            outputs.logits[:, :-1],
            "B S V -> (B S) V",
        )
        labels = rearrange(input_ids[:, 1:], "B S -> (B S)")
        completion_mask = rearrange(completion_mask[:, :-1], "B S -> (B S)")
        nll = F.cross_entropy(logits, torch.where(completion_mask.bool(), labels, -100))
        nlls.append(nll.item())
        token_counts.append(completion_mask.sum().item())

    NLL = np.average(nlls, weights=token_counts).item()
    PPL = math.exp(NLL)
    BPB = (sum(token_counts) / utf8_bytes) * NLL / math.log(2)

    results = {"judge-model": args.model, "NLL": NLL, "PPL": PPL, "BPB": BPB}

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, f"{alias}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("done!")


if __name__ == "__main__":
    main()
