"""Build a stable, root-bounded file manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nts_med.cli._common import print_status, software_versions, unavailable_payload, write_json
from nts_med.data.manifests import (
    ManifestDependencyError,
    ManifestStatus,
    build_and_write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for testable ``--help`` behavior."""

    parser = argparse.ArgumentParser(
        description="Build a deterministic SHA256 manifest without following external symlinks."
    )
    parser.add_argument("--dataset", default=None, help="optional registry name for provenance")
    parser.add_argument("--root", type=Path, required=True, help="authorized dataset root")
    parser.add_argument("--output", type=Path, required=True, help=".csv or .parquet manifest")
    parser.add_argument(
        "--image-metadata",
        action="store_true",
        help="decode supported images to record width, height, channels, and corruption",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default=None,
        help="override the format inferred from --output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run manifest construction or emit an explicit no-data state."""

    args = build_parser().parse_args(argv)
    if not args.root.exists() or not args.root.is_dir():
        print_status(unavailable_payload(args.root, kind="dataset_root"))
        return 0

    try:
        records = build_and_write_manifest(
            args.root,
            args.output,
            include_image_metadata=args.image_metadata,
            output_format=args.format,
        )
    except (ManifestDependencyError, OSError, ValueError) as error:
        print_status(
            {
                "status": "MANIFEST_FAILED",
                "error": f"{type(error).__name__}: {error}",
                "software_versions": software_versions(),
            }
        )
        return 2

    issue_count = sum(record.status is not ManifestStatus.OK for record in records)
    summary: dict[str, object] = {
        "status": "MANIFEST_WRITTEN",
        "dataset": args.dataset,
        "root": str(args.root.resolve()),
        "output": str(args.output),
        "file_count": len(records),
        "issue_count": issue_count,
        "config_snapshot": {
            "include_image_metadata": args.image_metadata,
            "output_format": args.format or args.output.suffix.removeprefix("."),
        },
        "software_versions": software_versions(),
    }
    write_json(Path(f"{args.output}.metadata.json"), summary)
    print_status(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
