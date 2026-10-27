"""Infer auditable pair relations from point or interval burden CSV rows."""

from __future__ import annotations

import argparse
import csv
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nts_med.cli._common import (
    load_yaml_mapping,
    print_status,
    software_versions,
    unavailable_payload,
    write_json,
)
from nts_med.relations.algebra import (
    Interval,
    RelationConfig,
    infer_relation_with_evidence,
)
from nts_med.relations.pair_sampling import DirectedPair, sample_directed_pairs


def build_parser() -> argparse.ArgumentParser:
    """Return the relation-builder parser."""

    parser = argparse.ArgumentParser(
        description="Infer five-way relations from a burden CSV using a frozen YAML config."
    )
    parser.add_argument("--burdens", type=Path, required=True, help="input burden .csv")
    parser.add_argument("--relation-config", type=Path, required=True, help="relation YAML")
    parser.add_argument("--output", type=Path, required=True, help="output relation .csv")
    parser.add_argument("--id-column", default="instance_id", help="unique identifier column")
    parser.add_argument(
        "--pairs-per-anchor",
        type=int,
        default=None,
        help="deterministically sample this many directed partners per instance",
    )
    parser.add_argument("--seed", type=int, default=20260823, help="pair-sampling seed")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100000,
        help="hard safety limit for generated pair rows",
    )
    return parser


def _float_mapping(config: Mapping[str, Any], key: str, axes: Sequence[str]) -> dict[str, float]:
    raw = config.get(key)
    if not isinstance(raw, Mapping) or set(raw) != set(axes):
        raise ValueError(f"{key} must map exactly the configured axes")
    return {axis: float(raw[axis]) for axis in axes}


def _load_relation_config(path: Path) -> tuple[tuple[str, ...], RelationConfig, dict[str, Any]]:
    snapshot = load_yaml_mapping(path)
    raw_axes = snapshot.get("axes")
    if not isinstance(raw_axes, list) or not raw_axes or not all(
        isinstance(axis, str) and axis for axis in raw_axes
    ):
        raise ValueError("relation config requires a non-empty axes list")
    axes = tuple(raw_axes)
    if len(set(axes)) != len(axes):
        raise ValueError("relation config axes must be unique")
    direction = snapshot.get("direction")
    if direction != "higher_is_worse":
        raise ValueError("only direction=higher_is_worse is supported")
    config = RelationConfig(
        delta=_float_mapping(snapshot, "delta", axes),
        epsilon=_float_mapping(snapshot, "epsilon", axes),
        max_width=_float_mapping(snapshot, "max_width", axes),
        numeric_tolerance=float(snapshot.get("numeric_tolerance", 1.0e-12)),
    )
    return axes, config, snapshot


def _parse_interval(row: Mapping[str, str], axis: str, row_number: int) -> Interval:
    point = row.get(axis)
    lower = row.get(f"{axis}_lower")
    upper = row.get(f"{axis}_upper")
    if point not in (None, ""):
        if lower not in (None, "") or upper not in (None, ""):
            raise ValueError(f"row {row_number}: {axis} mixes point and interval columns")
        value = float(point)
        return Interval(value, value)
    if lower in (None, "") or upper in (None, ""):
        raise ValueError(f"row {row_number}: missing {axis} point or lower/upper interval")
    return Interval(float(lower), float(upper))


def _read_instances(
    path: Path,
    id_column: str,
    axes: Sequence[str],
) -> dict[str, dict[str, Interval]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or id_column not in reader.fieldnames:
            raise ValueError(f"burden CSV requires identifier column {id_column!r}")
        instances: dict[str, dict[str, Interval]] = {}
        for row_number, row in enumerate(reader, start=2):
            identifier = row.get(id_column, "").strip()
            if not identifier:
                raise ValueError(f"row {row_number}: empty identifier")
            if identifier in instances:
                raise ValueError(f"row {row_number}: duplicate identifier {identifier!r}")
            instances[identifier] = {
                axis: _parse_interval(row, axis, row_number) for axis in axes
            }
    return dict(sorted(instances.items()))


def _pairs(
    identifiers: Sequence[str],
    pairs_per_anchor: int | None,
    seed: int,
) -> tuple[DirectedPair, ...]:
    if pairs_per_anchor is not None:
        return sample_directed_pairs(
            identifiers,
            pairs_per_anchor=pairs_per_anchor,
            seed=seed,
        )
    return tuple(
        DirectedPair(left, right) for left, right in itertools.combinations(identifiers, 2)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build pair rows with axis evidence and a complete sidecar snapshot."""

    args = build_parser().parse_args(argv)
    if not args.burdens.is_file():
        print_status(unavailable_payload(args.burdens, kind="burden_table"))
        return 0
    try:
        if args.burdens.suffix.lower() != ".csv":
            raise ValueError("burden input must end in .csv during M0-M1")
        if args.output.suffix.lower() != ".csv":
            raise ValueError("relation output must end in .csv during M0-M1")
        resolved_output = args.output.resolve()
        if resolved_output in {args.burdens.resolve(), args.relation_config.resolve()}:
            raise ValueError("relation output must not overwrite an input or configuration file")
        if args.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        axes, config, config_snapshot = _load_relation_config(args.relation_config)
        instances = _read_instances(args.burdens, args.id_column, axes)
        pairs = _pairs(tuple(instances), args.pairs_per_anchor, args.seed)
        if len(pairs) > args.max_pairs:
            raise ValueError(
                f"pair count {len(pairs)} exceeds --max-pairs={args.max_pairs}; "
                "sample partners explicitly"
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            fields = [
                "pair_id",
                "left_id",
                "right_id",
                "relation",
                "left_worse_axes",
                "right_worse_axes",
                "dominance_axes",
                "equivalent_axes",
                "unresolved_axes",
                "unresolved_reason",
                "minimum_margin",
                "reason",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index, pair in enumerate(pairs):
                evidence = infer_relation_with_evidence(
                    instances[pair.left_id],
                    instances[pair.right_id],
                    config,
                )
                writer.writerow(
                    {
                        "pair_id": f"pair_{index:09d}",
                        "left_id": pair.left_id,
                        "right_id": pair.right_id,
                        "relation": evidence.relation.value,
                        "left_worse_axes": ";".join(evidence.left_worse_axes),
                        "right_worse_axes": ";".join(evidence.right_worse_axes),
                        "dominance_axes": ";".join(evidence.dominance_axes),
                        "equivalent_axes": ";".join(evidence.equivalent_axes),
                        "unresolved_axes": ";".join(evidence.unresolved_axes),
                        "unresolved_reason": (
                            ""
                            if evidence.unresolved_reason is None
                            else evidence.unresolved_reason.value
                        ),
                        "minimum_margin": (
                            "" if evidence.minimum_margin is None else evidence.minimum_margin
                        ),
                        "reason": evidence.reason,
                    }
                )
        summary: dict[str, object] = {
            "status": "RELATIONS_WRITTEN",
            "input": str(args.burdens),
            "output": str(args.output),
            "instance_count": len(instances),
            "pair_count": len(pairs),
            "config_snapshot": config_snapshot,
            "pair_sampling": {
                "pairs_per_anchor": args.pairs_per_anchor,
                "seed": args.seed,
                "max_pairs": args.max_pairs,
            },
            "software_versions": software_versions(),
        }
        write_json(Path(f"{args.output}.metadata.json"), summary)
    except (OSError, TypeError, ValueError) as error:
        print_status(
            {
                "status": "RELATION_BUILD_FAILED",
                "error": f"{type(error).__name__}: {error}",
                "software_versions": software_versions(),
            }
        )
        return 2
    print_status(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
