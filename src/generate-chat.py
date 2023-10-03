import argparse
import json
import os
import pandas as pd
import sys
import time
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama.generation import Llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from pathlib import Path
from tqdm import trange
from watermarks import WATERMARKS, Watermark

"""
Modified based on https://github.com/facebookresearch/llama/blob/main/llama/generation.py
"""


class LlamaWithWatermark(Llama):
    @staticmethod
    def build(
        watermark_alg: Watermark,
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: int | None = None,
        seed: int = 1,
    ) -> "LlamaWithWatermark":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            watermark_alg (Watermark): An initialized watermark object
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return LlamaWithWatermark(model, tokenizer, watermark_alg)

    def __init__(
        self, model: Transformer, tokenizer: Tokenizer, watermark_alg: Watermark
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.watermark_alg = watermark_alg

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[list[int]],
        max_gen_len: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: bool = False,
        echo: bool = False,
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        assert top_p == 1.0, "top-p sampling is not well-defined with watermarking"

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            temperature = max(1e-5, temperature)
            logits = logits / temperature

            next_token = torch.stack(
                [
                    self.watermark_alg.watermark(
                        tokens[i, prev_pos:cur_pos].tolist(), logits[i, -1]
                    )
                    for i in range(bsz)
                ]
            )

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)


CKPT_PATH = "/data/datasets/models/meta-ai/llama2/weights/"
TOKENIZER_PATH = "/data/datasets/models/meta-ai/llama2/weights/tokenizer.model"
MODELS = ["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat"]
LLAMA_VOCAB_SIZE = 32000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/debug")
    parser.add_argument("--model", type=str, choices=MODELS, default="llama-2-7b-chat")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--watermark", choices=WATERMARKS.keys(), default="none")
    parser.add_argument("--prompt", default="data/rr-rlhf.jsonl")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    watermark = WATERMARKS[args.watermark](LLAMA_VOCAB_SIZE)
    llama = LlamaWithWatermark.build(
        watermark_alg=watermark,
        ckpt_dir=CKPT_PATH + args.model,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=512,
        max_batch_size=args.batch_size,
    )

    df = pd.read_json(args.prompt, lines=True)
    if args.debug:
        df = df.head(20)

    dialogs = df["prompt"]
    completions = []

    for offset in trange(0, len(dialogs), args.batch_size):
        prompts = dialogs[offset : offset + args.batch_size]
        generations = llama.chat_completion(
            prompts,
            max_gen_len=args.max_new_tokens,
            temperature=1.0,
            top_p=1.0,
        )
        completions.extend([g["generation"] for g in generations])

    df["completion"] = completions
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_json(
        os.path.join(args.output_dir, f"{args.model}-{args.watermark}.jsonl"),
        lines=True,
        orient="records",
    )


if __name__ == "__main__":
    main()
