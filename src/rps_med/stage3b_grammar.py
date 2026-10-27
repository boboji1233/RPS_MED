"""Finite-state path grammar for development-only RPS-Med Stage 3B."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

SCALE_COUNT = 17
RELATION_COUNT = 5
CONSTANT_STATES = (0, 1, 2, 3, 4)
ADMISSIBLE_TRANSITIONS = (
    (0, 2),
    (0, 4),
    (1, 2),
    (1, 4),
    (3, 2),
    (3, 4),
)


@lru_cache(maxsize=1)
def canonical_paths() -> NDArray[np.int64]:
    """Return the frozen 101-path grammar in canonical tie-break order."""

    rows: list[NDArray[np.int64]] = []
    for state in CONSTANT_STATES:
        rows.append(np.full(SCALE_COUNT, state, dtype=np.int64))
    for start, end in ADMISSIBLE_TRANSITIONS:
        for boundary in range(1, SCALE_COUNT):
            path = np.full(SCALE_COUNT, end, dtype=np.int64)
            path[:boundary] = start
            rows.append(path)
    output = np.stack(rows)
    if output.shape != (101, SCALE_COUNT) or np.unique(output, axis=0).shape[0] != 101:
        raise RuntimeError("Stage 3B canonical grammar construction failed")
    output.setflags(write=False)
    return output


def canonical_path_ids(paths: NDArray[np.int64]) -> NDArray[np.int64]:
    """Map analytic paths to the unique canonical path identifier."""

    if paths.ndim != 2 or paths.shape[1] != SCALE_COUNT:
        raise ValueError("Stage 3B paths must have shape [pair, 17]")
    grammar = canonical_paths()
    matches = np.all(paths[:, None, :] == grammar[None, :, :], axis=2)
    counts = matches.sum(axis=1)
    if not bool(np.all(counts == 1)):
        raise ValueError("Stage 3B truth contains a path outside the frozen grammar")
    return cast(NDArray[np.int64], np.argmax(matches, axis=1).astype(np.int64))


def minimum_hamming_projection(paths: NDArray[np.int64]) -> NDArray[np.int64]:
    """Project hard paths by minimum Hamming distance and canonical tie-break."""

    if paths.ndim != 2 or paths.shape[1] != SCALE_COUNT:
        raise ValueError("Stage 3B hard predictions must have shape [pair, 17]")
    if bool((paths < 0).any()) or bool((paths >= RELATION_COUNT).any()):
        raise ValueError("Stage 3B hard predictions contain an unknown relation code")
    grammar = canonical_paths()
    distances = np.count_nonzero(paths[:, None, :] != grammar[None, :, :], axis=2)
    identifiers = np.argmin(distances, axis=1)
    return cast(NDArray[np.int64], grammar[identifiers].copy())


def torch_canonical_paths(device: torch.device) -> Tensor:
    """Materialize the frozen grammar on the requested torch device."""

    return torch.as_tensor(canonical_paths().copy(), dtype=torch.long, device=device)


def path_scores(logits: Tensor, grammar: Tensor | None = None) -> Tensor:
    """Return the additive score of every canonical path for each pair."""

    if logits.ndim != 3 or logits.shape[1:] != (SCALE_COUNT, RELATION_COUNT):
        raise ValueError("Stage 3B logits must have shape [pair, 17, 5]")
    if grammar is None:
        grammar = torch_canonical_paths(logits.device)
    if grammar.shape != (101, SCALE_COUNT) or grammar.dtype != torch.long:
        raise ValueError("Stage 3B torch grammar is invalid")
    pair_count = logits.shape[0]
    expanded_logits = logits[:, None, :, :].expand(-1, grammar.shape[0], -1, -1)
    indices = grammar[None, :, :, None].expand(pair_count, -1, -1, 1)
    return torch.gather(expanded_logits, dim=3, index=indices).squeeze(3).sum(dim=2)


def structured_path_nll(logits: Tensor, truth_path_ids: Tensor) -> Tensor:
    """Return normalized negative log likelihood over the 101-path family."""

    if truth_path_ids.ndim != 1 or truth_path_ids.shape[0] != logits.shape[0]:
        raise ValueError("Stage 3B truth path identifiers do not align with logits")
    if truth_path_ids.dtype != torch.long:
        raise ValueError("Stage 3B truth path identifiers must be torch.long")
    return torch.nn.functional.cross_entropy(path_scores(logits), truth_path_ids)


def grammar_decode(logits: Tensor, grammar: Tensor | None = None) -> Tensor:
    """Decode the maximum-score grammar path with the frozen tie-break."""

    if grammar is None:
        grammar = torch_canonical_paths(logits.device)
    identifiers = path_scores(logits, grammar).argmax(dim=1)
    return grammar[identifiers]


__all__ = [
    "ADMISSIBLE_TRANSITIONS",
    "CONSTANT_STATES",
    "RELATION_COUNT",
    "SCALE_COUNT",
    "canonical_path_ids",
    "canonical_paths",
    "grammar_decode",
    "minimum_hamming_projection",
    "path_scores",
    "structured_path_nll",
    "torch_canonical_paths",
]
