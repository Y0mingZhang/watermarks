import torch
from abc import abstractmethod
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


class NoWatermark(Watermark):
    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        p = logits.softmax(-1)
        return torch.multinomial(p, 1)


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
        self.green_list_size = round(green_list_ratio * vocab_size)
        self.seed = seed

    def watermark(
        self, prev_tokens: list[int], logits: torch.FloatTensor
    ) -> torch.LongTensor:
        # generate a seed from previous tokens
        seed = obj_to_hash(self.seed, prev_tokens[-self.n_grams :])
        rng = torch.Generator("cuda")
        rng.manual_seed(seed)

        # partition vocab into green-red lists
        noise = torch.rand(self.vocab_size, generator=rng)
        green_mask = noise <= torch.kthvalue(noise, self.green_list_size).values
        logits[green_mask] += self.green_logit_bias

        return torch.multinomial(logits.softmax(-1), 1)


WATERMARKS: dict[str, type[Watermark]] = {
    "none": NoWatermark,
    "aaronson": Aaronson,
    "kuditipudi": Kuditipudi,
    "kirchenbauer": Kirchenbauer,
}
