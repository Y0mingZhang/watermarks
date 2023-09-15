import pandas as pd
import torch
from transformers import AutoTokenizer
from watermarks import OpenAIScheme

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
vocab_size = tokenizer.vocab_size
watermark = OpenAIScheme(vocab_size=vocab_size)


def determine_watermark_score(s: str) -> float:
    completion_tokens = tokenizer(s, return_tensors="pt")["input_ids"].flatten()
    scores = []
    for i in range(len(completion_tokens)):
        prev_tokens = completion_tokens[:i]
        r = watermark.get_r(prev_tokens.tolist())
        scores.append(-torch.log(1 - r[completion_tokens[i]]).item())
    return sum(scores) / len(scores)


def main():
    data_file = "outputs/mix/Llama2.jsonl"
    df = pd.read_json(data_file, lines=True).sample(200)
    df["text"] = df.apply((lambda r: r["prefix"] + r["completion"]), axis=1)
    df["score"] = df["text"].map(determine_watermark_score)

    # df["score"] = df["completion"].map(determine_watermark_score)
    df.to_json("outputs/mix/Llama2-scored.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    main()
