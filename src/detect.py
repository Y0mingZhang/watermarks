import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from watermarks import WATERMARKS, Kuditipudi


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--watermark", choices=WATERMARKS.keys(), default="none")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--generation", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = tokenizer.vocab_size

    df = pd.read_json(args.generation, lines=True, orient="records")
    if isinstance(df["completion"][0], str):
        completions = df["completion"]
    else:
        assert isinstance(df["completion"][0], dict)
        completions = df["completion"].apply(lambda d: d["content"])

    wm = WATERMARKS[args.watermark](vocab_size)
    p_values = []
    for s in tqdm(completions):
        tokens = tokenizer(s, add_special_tokens=False)["input_ids"]
        pv = wm.p_value(tokens)
        p_values.append(pv)

    df["p-value"] = p_values
    df.to_json(args.generation, lines=True, orient="records")


if __name__ == "__main__":
    main()
