# DEGP

## Decoupled Entropic-Geometric Planning — Artifact

Artifact for the paper:

> **Decoupled Entropic-Geometric Planning (DEGP): Verified Conditioning, Measured Mechanisms, and a Mapped Contact-Prediction Boundary in Latent-World-Model Deployment**  
> M. Mostafa, 2026

DEGP decouples a latent world model into:

- An encoder trained using a variance–covariance objective only.
- A dynamics predictor trained on detached latents with residual straightening and a Jacobian penalty.
- A leashed asymmetric planner operating under a runtime contract consisting of an `ε · T` budget, watchdog, and fallback.

The paper reports the complete pre-registered program, including its negative results:

- Deployment on Push-T achieved `0/50` successes in every condition.
- An 18-condition pre-registered attribution battery located the failure boundary.
- Self-supervised latents retained a contact signal with `5.8×` free/contact discrimination.
- No predictor version learned to use that signal.
- The probe-decoded cost was also measured as part of the attribution analysis.

Every claim in the paper is labeled by evidence level:

- Analytical proof
- Executable verification
- Empirical validation

This repository contains the executable half of that contract.

## Overview

Latent world models for robotics face a structural tension:

- Vision backbones need high-entropy representations to remain robust under domain shift.
- Gradient-based planners require strictly conditioned latent geometries.

DEGP addresses this tension architecturally rather than by compromising between the two objectives.

| Module | Objective |
| --- | --- |
| Encoder | Variance–covariance regularization only — “blind perception” |
| Predictor | Residual straightening and Jacobian penalty on detached latents — “sharp control” |
| Planner | `μ`-leashed asymmetric quasi-metric cost |
| Runtime | `ε · T` conditioning budget, watchdog, fallback, and oracle-rollout fidelity gate |

## Main Findings

### Certified conditioning

Conditioning is certified and dimension-invariant:

- The condition number `κ` is statistically flat across `K ∈ {64, 128, 256}`.
- TOST was performed with a margin of `ln(1.25)`.
- The analysis used 30 paired seeds.
- Each pair achieved `p < 10⁻⁴`.
- The runtime budget was satisfied at every tested horizon.

### Push-T deployment failure

Deployment on Push-T achieved `0/50` successes in every condition.

The falsified pre-registered hypotheses are reported as falsified rather than being omitted or reframed.

### Attribution battery

The 18-condition attribution battery located the failure boundary:

- Self-supervised latents retained a contact signal with `5.8×` free/contact discrimination.
- Predictor selectivity remained at or below `1.3×`.
- Planner family and order were exonerated by direct controls.
- Perception input was exonerated by direct controls.
- The conditioning leash was exonerated by direct controls.

Additional rollout results:

- Oracle dynamics with the same planner moved the T-block by `72 px`.
- All Equation (1)-latent models moved it by at most `2.3 px`.
- A reconstruction world model moved it in `6/10` episodes.

These results support the distinction:

> Information present in a representation is not necessarily information learnable from it.

## Quickstart

### Verify the theory

The verification suite runs on CPU and takes approximately one minute.

```bash
git clone https://github.com/<USER>/degp-artifact.git
cd degp-artifact

pip install -e .

python checks/run_checks.py
```

The report is written to:

```text
checks/REPORT.md
```

Expected output:

```text
12/12 checks passed
```

The suite verifies:

- Stop-gradient isolation proposition
- Quasi-metric zero-set predicate
- `κ` bound
- Dimension invariance of `κ`
- `ε · T` budget sweep
- Leash escape law
- Orthogonal-component preservation
- Wendel statistics
- Runtime tripwires

Each check includes documented seeds, tolerances, and theorem linkage.

### Reproduce the experiments

The full experiment battery requires a GPU and may take several hours.

```bash
pip install -e ".[battery]"
```

The battery additionally requires:

- `pymunk==6.11.1`
- `gym-pusht`
- `gymnasium`
- `jupyter`

Launch the reproduction notebook:

```bash
jupyter notebook notebooks/degp_battery.ipynb
```

The battery is checkpointed and idempotent. Every episode and job is saved upon completion, allowing interrupted sessions to resume without repeating completed work.

Under the pinned seeds, reruns agree to:

```text
max |x₁ − x₂| < 10⁻⁸
```

## Notebook Cell Order

The reproduction notebook contains six main cells:

1. **PRE-FLIGHT**  
   Installs and checks packages, validates the environment, and regenerates or loads cached data.

2. **DBATCH-DEF**  
   Defines the battery machinery, definitions, and supplementary analysis.

3. **E-BASELINES**  
   Runs TD-MPC2-lite and Dreamer-v3-style RSSM baselines.

4. **DRIVER**  
   Executes the battery in priority order. Runs are idempotent.

5. **FIGURES**  
   Generates all paper figures, including movement, selectivity, and arena figures.

6. **K30**  
   Runs the Tier-1 30-seed replication. This can be executed in an independent session.

## Repository Layout

```text
degp-artifact/
├── degp/
│   ├── planner/
│   ├── leash/
│   ├── planning/
│   ├── phaseb/
│   ├── phasec2/
│   ├── predictor/
│   └── action_diagnostics/
├── notebooks/
│   └── degp_battery.ipynb
├── checks/
│   ├── run_checks.py
│   └── REPORT.md
├── tools/
│   ├── pull_package.py
│   └── validate_notebook.py
├── preregistration/
│   └── REGISTER.md
├── results/
│   ├── per-seed CSV files
│   ├── JSON records
│   └── pickle records
├── checkpoints/
│   ├── model weights
│   ├── original v4 identity anchor
│   └── V4_SHA256.txt
├── figures/
│   └── paper figures
├── data/
│   └── README.md
└── README.md
```

The dataset itself is not stored in Git. It is regenerated deterministically or downloaded as described in [`data/README.md`](data/README.md).

## Checkpoints and Identity

The original deployment-study checkpoint is:

```text
checkpoints/c2_world_model_v4.pt
```

Its SHA-256 hash is recorded in:

```text
checkpoints/V4_SHA256.txt
```

The checkpoint has:

```text
CDR_dec = 2.726
```

This reproduces the previously reported value of `2.714` within protocol noise and anchors checkpoint identity across revisions.

All other checkpoints, including `v1–v3`, `B1`, `B1b`, `D4`, `E1`, and `E2`, regenerate deterministically under the pinned seeds.

The `B1b` verdict was additionally confirmed to be stable across three independent retrains.

## Data

The Push-T dataset contains:

- 50,000 transitions
- 250 episodes
- Seeded collection

The dataset is regenerated by Cell 1 in approximately 15 minutes.

A frozen copy is not stored in Git. A release asset will be provided at publication.

The collection protocol, data split, and file schemas are documented in:

[`data/README.md`](data/README.md)

## Verification-Suite Provenance

`checks/run_checks.py` was reassembled from the paper's Appendix A specification. The original Phase-A session scripts were not preserved as standalone files.

The historical run parameters and tolerances are encoded in the verification suite, including:

- 4,061 zero-set predicate points
- `5.2 × 10⁻¹⁶` orthogonal-preservation measurement
- Escape timing of no more than one step

Checks tagged `[spec]` verify the mathematical property directly.

Checks tagged `[impl]` import the shipped package where available.

Continuous integration runs the verification suite on every push.

## Reproducibility Disclosures

The released Tier-1 instrument implements `float32` where its docstring specifies `float64`. The artifact governs this implementation detail, as documented in Appendix B of the paper.

The 30-seed replication used a recipe recovered through pre-registered fingerprint calibration against the original artifact medians.

The oracle-rollout fidelity gate detected that naive Pymunk state teleportation is invalid, producing a `55.9 px` free-space divergence. The harness therefore automatically selects exact replay-based rollouts.

The fidelity gate is included in the reproduction harness.

The `E1` and `E2` baselines share a CEM random stream and are counted as one effective condition, as specified in Amendment A11.

Reproduction is notebook-based: a single checkpointed and idempotent harness maps its cells one-to-one to every table and figure.

## Environment

The artifact was developed and executed on Kaggle using:

- Python 3.12
- PyTorch 2.x
- Pymunk 6.11.1
- A single P100- or T4-class GPU
- Internet access enabled for package installation and dataset attachment

CPU execution is sufficient for:

- The verification suite
- Cell 6 of the reproduction notebook

## Citation

If you use this artifact, please cite:

```bibtex
@misc{mostafa2026degp,
  author       = {Mostafa, Mohsen},
  title        = {{DEGP}: Decoupled Entropic-Geometric Planning (artifact)},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/<USER>/degp-artifact},
  note         = {Artifact for ``Verified Conditioning, Measured Mechanisms,
                   and a Mapped Contact-Prediction Boundary in
                   Latent-World-Model Deployment''}
}
```

## License

License information will be added before publication.
