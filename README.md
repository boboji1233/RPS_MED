# RPS-Med

Relation-Persistence Spectrum Learning for Medical Images.

RPS-Med represents pairwise pathology relations across an explicit tolerance grid
rather than reducing each image pair to a single severity ordering. The five
states are `LESS`, `GREATER`, `EQUIVALENT`, `STRUCTURAL_INCOMPARABLE`, and `UNRESOLVED`.
Structural incomparability denotes crossing pathology axes; unresolved denotes
insufficient observational evidence.

## Components

- `src/rps_med/`: tolerance spectra, critical tolerances, source adapters, model
  training and evaluation, pair-bootstrap audits, and Stage 3 analyses.
- `src/nts_med/`: shared relation algebra, lesion measurements, data adapters,
  reproducibility checks, and statistical utilities used by RPS-Med.
- `configs/`: versioned experiment, relation, and Stage 3 configurations.
- `metadata/preregistration/`: historical experiment contracts.
- `tests/`: RPS-Med unit and synthetic integration tests.

## Installation

Use Python 3.11 or 3.12 in a dedicated environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,model]"
```

The model extra uses PyTorch 2.7 and torchvision 0.22. Install the appropriate
CPU or CUDA build for your machine. Core spectrum operations use NumPy and do
not require trained models.

## Tests

```bash
python -m pytest -q
```

Tests use constructed arrays and temporary files. Four checks additionally bind
to historical execution records and are skipped when those separately held
records are absent. Test outputs are software checks, not estimates of research
performance.

## Synthetic spectrum check

```bash
python -m rps_med.cli.synthetic_r1 \
  --config configs/experiment/rps_med_r1_v0.5.yaml \
  --output reports/synthetic_r1.json
```

The output path must not already exist. Additional commands are available through
`python -m rps_med.cli.<module> --help` and the `rps-med-*` entry points.

## Data and reproducibility

Medical images, patient identifiers, local data manifests, pretrained weights,
experiment outputs, execution passports, and credentials are not distributed
with this source release. Obtain datasets from their official providers under
their respective terms. Configure local input locations explicitly.

Versioned configurations preserve the original experimental settings and may
reference local paths or separately held evidence files. They are not portable
execution authorizations. Do not reinterpret earlier exposed results as new
held-out evidence or overwrite frozen experiment records. The source tests can
run without those private inputs; full study reproduction additionally requires
the corresponding authorized data, weights, and original execution records.

The package does not infer clinical treatment priority.

## License

The source is distributed under the [MIT license](LICENSE). Dataset and model
licenses remain with their respective owners.
