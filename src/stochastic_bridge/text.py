from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

import torch
from torch import Tensor


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class CocoWordTokenizer:
    """Small deterministic tokenizer whose vocabulary is learned from COCO."""

    def __init__(self, vocabulary: list[str], max_length: int = 48) -> None:
        if tuple(vocabulary[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"vocabulary must start with {SPECIAL_TOKENS}")
        if max_length < 3:
            raise ValueError("max_length must be at least 3")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary tokens must be unique")
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self.max_length = int(max_length)

    @classmethod
    def build_from_coco(
        cls,
        captions_file: str | Path,
        *,
        vocab_size: int = 16384,
        min_frequency: int = 2,
        max_length: int = 48,
    ) -> "CocoWordTokenizer":
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("vocab_size is too small")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        path = Path(captions_file).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        counts: Counter[str] = Counter()
        for annotation in raw.get("annotations", []):
            counts.update(cls.tokenize(str(annotation["caption"])))
        # Frequency first and lexical order second makes the file reproducible.
        candidates = sorted(
            (item for item in counts.items() if item[1] >= min_frequency),
            key=lambda item: (-item[1], item[0]),
        )
        vocabulary = list(SPECIAL_TOKENS)
        vocabulary.extend(token for token, _ in candidates[: vocab_size - len(vocabulary)])
        return cls(vocabulary, max_length=max_length)

    @classmethod
    def load(cls, path: str | Path) -> "CocoWordTokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(raw["vocabulary"]), max_length=int(raw["max_length"]))

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"max_length": self.max_length, "vocabulary": list(self.vocabulary)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def encode(self, text: str) -> tuple[Tensor, Tensor]:
        content = self.tokenize(text)[: self.max_length - 2]
        ids = [self.token_to_id["<bos>"]]
        ids.extend(self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in content)
        ids.append(self.token_to_id["<eos>"])
        mask = [True] * len(ids)
        padding = self.max_length - len(ids)
        ids.extend([self.token_to_id["<pad>"]] * padding)
        mask.extend([False] * padding)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def batch_encode(self, texts: Iterable[str]) -> tuple[Tensor, Tensor]:
        encoded = [self.encode(text) for text in texts]
        if not encoded:
            raise ValueError("at least one caption is required")
        return (
            torch.stack([item[0] for item in encoded]),
            torch.stack([item[1] for item in encoded]),
        )

    def __len__(self) -> int:
        return len(self.vocabulary)
