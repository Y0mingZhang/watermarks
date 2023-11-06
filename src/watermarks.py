import itertools
import math
import numpy as np
import torch
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import gamma, norm
from src.levenshtein import levenshtein
from utils import obj_to_hash


class Watermark:
    def __init__(*_, **__):
        ...

    @abstractmethod
    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        # selects the next token using watermark
        ...

    @abstractmethod
    def p_value(self, tokens: list[int]) -> float:
        ...


class NoWatermark(Watermark):
    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        p = logits.softmax(-1)
        return torch.multinomial(p, 1)

    def p_value(self, tokens: list[int]) -> float:
        return 1.0


class Aaronson(Watermark):
    def __init__(
        self,
        vocab_size: int,
        seed: int = 42,
        n_grams: int = 3,
    ):
        self.vocab_size = vocab_size
        self.n_grams = n_grams
        self.seed = seed

    def pseudorandom_uniform(self, prev_tokens: list[int]) -> torch.Tensor:
        rng = torch.Generator()
        rng.manual_seed(obj_to_hash(self.seed, prev_tokens))
        return torch.rand((self.vocab_size,), generator=rng, device="cpu")

    def get_r(
        self,
        prev_tokens: list[int],
    ) -> torch.FloatTensor:
        r = self.pseudorandom_uniform(prev_tokens[-self.n_grams :])
        return r

    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        p = logits.softmax(-1)
        r = self.get_r(prev_tokens).to(logits.device)
        watermarked_p = r.pow(1 / p)
        return watermarked_p.argmax(keepdim=True)

    def p_value(self, tokens: list[int]) -> float:
        r_all = []
        N = len(tokens)
        for i in range(len(tokens)):
            t = tokens[i]
            r = self.get_r(tokens[:i])[t].item()
            r_all.append(r)

        r_all = np.array(r_all)
        mu = -np.log(1 - r_all).mean()

        # If unwatermarked, mu ~ Gamma(N, 1 / N)
        p_value = gamma.sf(mu, N, scale=1 / N)
        return p_value


class Kuditipudi(Watermark):
    def __init__(self, vocab_size: int, key_length: int = 256, seed: int = 42):
        self.vocab_size = vocab_size
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.xi = torch.rand(key_length, vocab_size, generator=rng)

    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        # NOTE: didn't implement shift since it has no effect on generation quality

        p = logits.softmax(-1)
        r = self.xi[len(prev_tokens) % len(self.xi)].to(logits.device)
        watermarked_p = r.pow(1 / p)
        return watermarked_p.argmax(keepdim=True)

    def p_value(self, tokens: list[int], n_runs: int = 100) -> float:
        tokens_np = np.array(tokens)
        xi = self.xi.numpy().astype(np.float32)
        test_result = self.test_statistic(tokens_np, xi)
        p_val = 0
        tpe = ProcessPoolExecutor(15)
        xi_alts = [
            np.random.rand(*self.xi.shape).astype(np.float32) for i in range(n_runs)
        ]
        for null_result in tpe.map(
            self.test_statistic, itertools.repeat(tokens_np), xi_alts
        ):
            p_val += null_result <= test_result

        return (p_val + 1.0) / (n_runs + 1.0)

    def test_statistic(self, tokens: np.ndarray, xi: np.ndarray, gamma: float = 0.0):
        m = len(tokens)
        n = len(xi)

        A = np.empty(n)
        for j in range(n):
            A[j] = levenshtein(tokens, xi[(j + np.arange(m)) % n], gamma)

        return np.min(A)


class Kirchenbauer(Watermark):
    def __init__(
        self,
        vocab_size: int,
        n_grams: int = 3,
        green_logit_bias: float = 1.0,
        green_list_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.vocab_size = vocab_size
        self.n_grams = n_grams
        self.green_logit_bias = green_logit_bias
        self.green_list_ratio = green_list_ratio
        self.green_list_size = round(green_list_ratio * vocab_size)
        self.seed = seed

    def get_green_mask(self, prev_tokens: list[int]):
        # generate a seed from previous tokens
        seed = obj_to_hash(self.seed, prev_tokens[-self.n_grams :])
        rng = torch.Generator(device="cuda")
        rng.manual_seed(seed)
        noise = torch.rand(self.vocab_size, generator=rng, device="cuda")

        # partition vocab into green-red lists
        green_mask = noise <= torch.kthvalue(noise, self.green_list_size).values
        return green_mask

    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        green_mask = self.get_green_mask(prev_tokens)
        logits[green_mask] += self.green_logit_bias

        return torch.multinomial(logits.softmax(-1), 1)

    def p_value(self, tokens: list[int]) -> float:
        greens = 0
        for i in range(len(tokens)):
            t = tokens[i]
            green_mask = self.get_green_mask(tokens[:i])
            if green_mask[t]:
                greens += 1

        expected_greens = self.green_list_ratio * len(tokens)
        z_score = (greens - expected_greens) / math.sqrt(
            len(tokens) * self.green_list_ratio * (1 - self.green_list_ratio)
        )
        p_value = norm.sf(z_score)
        return p_value


WATERMARKS: dict[str, type[Watermark]] = {
    "none": NoWatermark,
    "aaronson": Aaronson,
    "kuditipudi": Kuditipudi,
    "kirchenbauer": Kirchenbauer,
}
