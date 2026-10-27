"""Deterministic synthetic interval data for relation-algebra smoke tests.

The generated records are deliberately simple geometric constructions.  They
exercise the five preregistered relations without representing patients,
clinical observations, or experimental results.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from nts_med import __version__
from nts_med.relations.algebra import Interval, Relation, RelationConfig, infer_relation

SYNTHETIC_STATUS = "SYNTHETIC_SMOKE_TEST"
DEFAULT_SEED = 20260823
DEFAULT_PAIRS_PER_RELATION = 50

_KNOWN_AXIS_NAMES = ("ma", "hem", "ex", "cws", "irma", "nv")
_RELATIONS = (
    Relation.LESS,
    Relation.GREATER,
    Relation.EQUIVALENT,
    Relation.STRUCTURAL_INCOMPARABLE,
    Relation.UNRESOLVED,
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for one deterministic synthetic smoke-test corpus."""

    seed: int = DEFAULT_SEED
    num_axes: int = 4
    noise: float = 0.05
    pairs_per_relation: int = DEFAULT_PAIRS_PER_RELATION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.num_axes, bool)
            or not isinstance(self.num_axes, int)
            or self.num_axes < 2
        ):
            raise ValueError("num_axes must be an integer of at least 2 to cover all relations")
        if (
            isinstance(self.noise, bool)
            or not isinstance(self.noise, (int, float))
            or not math.isfinite(self.noise)
            or self.noise < 0.0
        ):
            raise ValueError("noise must be a finite, non-negative number")
        if (
            isinstance(self.pairs_per_relation, bool)
            or not isinstance(self.pairs_per_relation, int)
            or self.pairs_per_relation < 50
        ):
            raise ValueError("pairs_per_relation must be an integer of at least 50")

    @property
    def axes(self) -> tuple[str, ...]:
        """Return stable, human-readable axis names for ``num_axes`` axes."""

        if self.num_axes <= len(_KNOWN_AXIS_NAMES):
            return _KNOWN_AXIS_NAMES[: self.num_axes]
        generated = tuple(
            f"axis_{index + 1:02d}"
            for index in range(len(_KNOWN_AXIS_NAMES), self.num_axes)
        )
        return _KNOWN_AXIS_NAMES + generated


@dataclass(frozen=True)
class SyntheticPair:
    """A directed pair and its construction-time expected relation."""

    pair_id: str
    left_id: str
    right_id: str
    expected_relation: Relation


@dataclass(frozen=True)
class SyntheticDataset:
    """In-memory deterministic corpus plus the inference configuration."""

    config: SyntheticConfig
    relation_config: RelationConfig
    instances: Mapping[str, Mapping[str, Interval]]
    pairs: tuple[SyntheticPair, ...]


def _interval(center: float, width: float) -> Interval:
    half_width = width / 2.0
    return Interval(lower=center - half_width, upper=center + half_width)


def _nearby_centers(
    *,
    base: float,
    rng: random.Random,
    noise: float,
) -> tuple[float, float]:
    common = base + rng.uniform(-noise, noise)
    small_shift = 0.02 * noise
    return (
        common + rng.uniform(-small_shift, small_shift),
        common + rng.uniform(-small_shift, small_shift),
    )


def _make_pair_intervals(
    relation: Relation,
    *,
    axes: Sequence[str],
    base: float,
    rng: random.Random,
    noise: float,
    scale: float,
) -> tuple[dict[str, Interval], dict[str, Interval]]:
    narrow_width = 0.20 * scale
    wide_width = 4.00 * scale
    separation = 3.00 * scale
    left: dict[str, Interval] = {}
    right: dict[str, Interval] = {}

    for axis_index, axis in enumerate(axes):
        if relation in (Relation.LESS, Relation.GREATER):
            low = base + rng.uniform(-noise, noise)
            high = base + separation + rng.uniform(-noise, noise)
            left_center, right_center = (
                (low, high) if relation is Relation.LESS else (high, low)
            )
            width = narrow_width
        elif relation is Relation.STRUCTURAL_INCOMPARABLE and axis_index < 2:
            low = base + rng.uniform(-noise, noise)
            high = base + separation + rng.uniform(-noise, noise)
            left_center, right_center = (high, low) if axis_index == 0 else (low, high)
            width = narrow_width
        elif relation is Relation.UNRESOLVED:
            left_center, right_center = _nearby_centers(
                base=base,
                rng=rng,
                noise=noise,
            )
            width = wide_width
        else:
            left_center, right_center = _nearby_centers(
                base=base,
                rng=rng,
                noise=noise,
            )
            width = narrow_width

        left[axis] = _interval(left_center, width)
        right[axis] = _interval(right_center, width)

    return left, right


def generate_synthetic_dataset(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """Construct deterministic directed pairs spanning all five relations.

    At least 50 pairs are generated for each relation.  Every pair owns two
    instances so there are no accidental constraints between constructions.
    The configured noise perturbs coordinates, while scale-aware margins keep
    the expected relation invariant.
    """

    resolved_config = config or SyntheticConfig()
    axes = resolved_config.axes
    scale = 1.0 + resolved_config.noise
    rng = random.Random(resolved_config.seed)

    zeros = {axis: 0.0 for axis in axes}
    epsilon = {axis: 0.10 * scale for axis in axes}
    max_width = {axis: 0.25 * scale for axis in axes}
    relation_config = RelationConfig(
        delta=zeros,
        epsilon=epsilon,
        max_width=max_width,
        numeric_tolerance=1.0e-12,
    )

    instances: dict[str, Mapping[str, Interval]] = {}
    pairs: list[SyntheticPair] = []
    serial = 0
    for relation in _RELATIONS:
        relation_slug = relation.name.lower()
        for relation_index in range(resolved_config.pairs_per_relation):
            pair_id = f"{relation_slug}_{relation_index:03d}"
            left_id = f"{pair_id}_left"
            right_id = f"{pair_id}_right"
            base = serial * 10.0 * scale
            left, right = _make_pair_intervals(
                relation,
                axes=axes,
                base=base,
                rng=rng,
                noise=resolved_config.noise,
                scale=scale,
            )
            instances[left_id] = left
            instances[right_id] = right
            pairs.append(
                SyntheticPair(
                    pair_id=pair_id,
                    left_id=left_id,
                    right_id=right_id,
                    expected_relation=relation,
                )
            )
            serial += 1

    return SyntheticDataset(
        config=resolved_config,
        relation_config=relation_config,
        instances=instances,
        pairs=tuple(pairs),
    )


def _format_float(value: float) -> str:
    return format(value, ".17g")


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _software_versions() -> dict[str, str]:
    return {
        "nts_med": __version__,
        "python": platform.python_version(),
        "pyyaml": _installed_version("PyYAML"),
    }


def _relation_thresholds(config: RelationConfig, axes: Sequence[str]) -> dict[str, object]:
    return {
        "axes": list(axes),
        "direction": "higher_is_worse",
        "delta": {axis: config.delta[axis] for axis in axes},
        "epsilon": {axis: config.epsilon[axis] for axis in axes},
        "max_width": {axis: config.max_width[axis] for axis in axes},
        "numeric_tolerance": config.numeric_tolerance,
        "precedence": [
            "dominance",
            "structural_incomparability",
            "equivalence",
            "unresolved",
        ],
    }


def write_synthetic_dataset(
    dataset: SyntheticDataset,
    output_dir: str | Path,
) -> dict[str, object]:
    """Validate a corpus with the real algebra and write all smoke artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    axes = dataset.config.axes

    instance_header = ["instance_id"]
    for axis in axes:
        instance_header.extend((f"{axis}_lower", f"{axis}_upper"))
    instance_rows: list[list[object]] = []
    for instance_id in sorted(dataset.instances):
        row: list[object] = [instance_id]
        for axis in axes:
            interval = dataset.instances[instance_id][axis]
            row.extend((_format_float(interval.lower), _format_float(interval.upper)))
        instance_rows.append(row)
    _write_csv(destination / "instances.csv", instance_header, instance_rows)

    pair_rows: list[list[object]] = []
    inferred_rows: list[list[object]] = []
    expected_counts: Counter[str] = Counter()
    inferred_counts: Counter[str] = Counter()
    matched_pairs = 0
    for pair in dataset.pairs:
        expected = pair.expected_relation.name
        inferred_relation = infer_relation(
            dataset.instances[pair.left_id],
            dataset.instances[pair.right_id],
            dataset.relation_config,
        )
        inferred = inferred_relation.name
        matched = inferred_relation is pair.expected_relation
        expected_counts[expected] += 1
        inferred_counts[inferred] += 1
        matched_pairs += int(matched)
        pair_rows.append(
            [pair.pair_id, pair.left_id, pair.right_id, expected]
        )
        inferred_rows.append(
            [
                pair.pair_id,
                pair.left_id,
                pair.right_id,
                expected,
                inferred,
                str(matched).lower(),
            ]
        )

    _write_csv(
        destination / "pairs.csv",
        ("pair_id", "left_id", "right_id", "expected_relation"),
        pair_rows,
    )
    _write_csv(
        destination / "inferred_relations.csv",
        (
            "pair_id",
            "left_id",
            "right_id",
            "expected_relation",
            "inferred_relation",
            "matched",
        ),
        inferred_rows,
    )

    relation_names = [relation.name for relation in _RELATIONS]
    total_pairs = len(dataset.pairs)
    summary: dict[str, object] = {
        "status": SYNTHETIC_STATUS,
        "seed": dataset.config.seed,
        "all_expected_relations_matched": matched_pairs == total_pairs,
        "num_axes": dataset.config.num_axes,
        "axes": list(axes),
        "noise": dataset.config.noise,
        "pairs_per_relation": dataset.config.pairs_per_relation,
        "total_instances": len(dataset.instances),
        "total_pairs": total_pairs,
        "matched_pairs": matched_pairs,
        "mismatched_pairs": total_pairs - matched_pairs,
        "expected_relation_counts": {
            name: expected_counts.get(name, 0) for name in relation_names
        },
        "inferred_relation_counts": {
            name: inferred_counts.get(name, 0) for name in relation_names
        },
        "software_versions": _software_versions(),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    config_snapshot: dict[str, object] = {
        "status": SYNTHETIC_STATUS,
        "generation": {
            "seed": dataset.config.seed,
            "num_axes": dataset.config.num_axes,
            "axes": list(axes),
            "noise": dataset.config.noise,
            "pairs_per_relation": dataset.config.pairs_per_relation,
        },
        "relation": _relation_thresholds(dataset.relation_config, axes),
        "software_versions": _software_versions(),
    }
    (destination / "config.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    readme = f"""# NTS-Med synthetic smoke-test artifacts

Status: `{SYNTHETIC_STATUS}`

These files contain deterministic, constructed intervals only. They contain no patient data,
real observations, trained-model output, clinical claims, or publishable experimental results.

- Seed: `{dataset.config.seed}`
- Axes: `{", ".join(axes)}`
- Directed pairs per relation: `{dataset.config.pairs_per_relation}`
- Matched pairs: `{matched_pairs}/{total_pairs}`

`config.yaml` records the complete generation and relation configuration plus software versions.
`inferred_relations.csv` is the row-level algebra check; `summary.json` is its aggregate status.
"""
    (destination / "README.md").write_text(
        readme,
        encoding="utf-8",
        newline="\n",
    )
    return summary


def run_synthetic_smoke(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    num_axes: int = 4,
    noise: float = 0.05,
    pairs_per_relation: int = DEFAULT_PAIRS_PER_RELATION,
) -> dict[str, object]:
    """Generate, infer, validate, and persist one synthetic smoke-test run."""

    config = SyntheticConfig(
        seed=seed,
        num_axes=num_axes,
        noise=noise,
        pairs_per_relation=pairs_per_relation,
    )
    return write_synthetic_dataset(generate_synthetic_dataset(config), output_dir)


__all__ = [
    "DEFAULT_PAIRS_PER_RELATION",
    "DEFAULT_SEED",
    "SYNTHETIC_STATUS",
    "SyntheticConfig",
    "SyntheticDataset",
    "SyntheticPair",
    "generate_synthetic_dataset",
    "run_synthetic_smoke",
    "write_synthetic_dataset",
]
