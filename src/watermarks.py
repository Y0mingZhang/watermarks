import torch
from abc import abstractmethod
from utils import obj_to_hash


class Watermark:
    def __init__(*_, **__):
        ...

    @abstractmethod
    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # returns the watermarked probabilities (not logits!)
        ...


class NoWatermark(Watermark):
    def watermark(self, prev_tokens: list[int], logits: torch.FloatTensor):
        p = logits.softmax(-1)
        return p


class OpenAIScheme(Watermark):
    def __init__(
        self,
        vocab_size: int,
        seed: int = 42,
        n_grams: int = 5,
    ):
        self.vocab_size = vocab_size
        self.n_grams = n_grams
        self.seed = seed

    def pseudorandom_uniform(self, prev_tokens: list[int]) -> torch.Tensor:
        rng = torch.Generator()
        rng.manual_seed(obj_to_hash(self.seed, prev_tokens))
        return torch.rand((self.vocab_size,), generator=rng)

    def get_r(
        self,
        prev_tokens: list[int],
    ) -> torch.FloatTensor:
        r = self.pseudorandom_uniform(prev_tokens)
        return r

    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        p = logits.softmax(-1)
        r = self.get_r(prev_tokens).to(logits.device)
        watermarked_p = r.pow(1 / p)
        return watermarked_p / watermarked_p.sum()


WATERMARKS: dict[str, type[Watermark]] = {"noop": NoWatermark, "openai": OpenAIScheme}
