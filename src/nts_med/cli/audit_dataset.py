"""Compatibility entry point for the Prompt 2 dataset qualification audit."""

from nts_med.cli.audit_dataset_v2 import build_parser as build_parser
from nts_med.cli.audit_dataset_v2 import main as main

if __name__ == "__main__":
    raise SystemExit(main())
