# DEGP
### DEGP: Decoupled Entropic-Geometric Planning — Artifact

Artifact for the paper "Decoupled Entropic-Geometric Planning (DEGP):Verified Conditioning, Measured Mechanisms, and a MappedContact-Prediction Boundary in Latent-World-Model Deployment"(M. Mostafa, 2026).

DEGP decouples a latent world model into an encoder trained with avariance–covariance objective only, a dynamics predictor trained ondetached latents with residual straightening and a Jacobian penalty,and a leashed asymmetric planner — under a runtime contract(ε·T budget, watchdog, fallback). The paper reports the fullpre-registered program including its negative results: deploymenton Push-T fails at 0/50 successes in every condition, and an18-condition pre-registered attribution battery locates the failureboundary: the self-supervised latents retain a contact signal(5.8× free/contact discrimination) that no predictor version learnsto use, together with the probe-decoded cost.

Every claim in the paper is labeled by evidence level (analyticalproof / executable verification / empirical validation), and falsifiedhypotheses are reported. This repository is the executable half ofthat contract.

### Overview
Latent world models for robotics face a structural tension: vision backbones needhigh-entropy representations to survive domain shift, while gradient-based plannersneed strictly conditioned latent geometries. DEGP resolves this tensionarchitecturally rather than by compromise:

Module	                  Objective
Encoder	       Variance–covariance regularization only ("blind perception")
Predictor	     Residual straightening + Jacobian penalty on detached latents ("sharp control")
Planner	       μ-leashed asymmetric quasi-metric cost
Runtime      	 ε·T conditioning budget, watchdog, fallback, oracle-rollout fidelity gate

The paper reports the full pre-registered program including its negativeresults, and this repository is the executable half of that contract:

Conditioning is certified and dimension-invariant — κ statistically flatacross K ∈ {64, 128, 256} (TOST at margin ln 1.25, 30 paired seeds,p < 10⁻⁴ per pair), budget-compliant at every horizon tested.

Deployment on Push-T fails at 0/50 in every condition — the falsifiedpre-registered hypotheses are reported as such.

An 18-condition attribution battery locates the boundary: theself-supervised latents retain a contact signal (5.8× free/contactdiscrimination) that no predictor version learns to use (selectivity≤ 1.3×), while planner family and order, perception input, and theconditioning leash are exonerated by direct controls. Oracle dynamics withthe same planner moves the T-block 72 px where all Eq. (1)-
latent modelsmove it ≤ 2.3 px; a reconstruction world model moves it in 6/10 episodes.Information present in a representation is not information learnablefrom it.

Every claim in the paper carries an evidence label (analytical proof /executable verification / empirical validation), and the pre-registrationregister (protocol + amendments A1–A11) ships in this repository.

### Quickstart

1. Verify the theory (~1 min, CPU)

git clone https://github.com/<USER>/degp-artifact.gitcd degp-artifactpip install -e .              # torch, numpy, scipy, matplotlib, Pillowpython checks/run_checks.py   # 12-check verification 

suite → checks/REPORT.md

Expected output: 12/12 checks passed. The suite verifies the stop-gradient
isolation proposition, the quasi-metric zero-set predicate, the κ bound and
its K-invariance, the ε·T budget sweep, the leash's escape law and
orthogonal-component preservation, Wendel statistics, and the runtime
tripwires — each with documented seeds, tolerances, and theorem linkage.

2. Reproduce the experiments (GPU; hours; resumable)
         pip install -e ".[battery]"   # + pymunk 6.11.1, gym-pusht, gymnasium
         jupyter notebook notebooks/degp_battery.ipynb

The battery is checkpointed and idempotent: every episode and job saves
on completion, so interrupted sessions resume without repeating work.
Reruns agree to max |x₁ − x₂| < 10⁻⁸ under pinned seeds.

Cell order:
         1  PRE-FLIGHT     package + environment + data (regenerates or loads cache)
         2  DBATCH-DEF     all battery machinery (definitions + supplement)
         3  E-BASELINES    TD-MPC2-lite and Dreamer-v3-style RSSM baselines
         4  DRIVER         runs the battery in priority order (idempotent)
         5  FIGURES        all paper figures (movement, selectivity, arena, …)
         6  K30            Tier-1 30-seed replication (independent session OK)


Repository layout
          degp/                the package: planner, leash, planning, phaseb,
                     phasec2, predictor, action_diagnostics
          notebooks/           degp_battery.ipynb — the 6-cell harness
          checks/              12-check verification suite + REPORT.md
          tools/               pull_package.py, validate_notebook.py 
          preregistration/     REGISTER.md — protocol + amendments A1–A11
          results/             per-seed CSV/JSON + pickle records for every table
          checkpoints/         model weights incl. the original v4 identity anchor
          figures/             every figure in the paper
          data/                not in git — regenerated (seeded) or downloaded


#### Checkpoints and identity
checkpoints/c2_world_model_v4.pt is the original v4 model of the
deployment study (SHA-256 in checkpoints/V4_SHA256.txt): its
CDR_dec = 2.726 reproduces the previously reported 2.714 within protocol
noise, anchoring checkpoint identity across revisions. All other checkpoints
(v1–v3, B1, B1b, D4, E1, E2) regenerate deterministically under pinned
seeds; the B1b verdict was additionally confirmed stable across three
independent retrains.   

#### Data
The Push-T dataset (50,000 transitions, 250 episodes, seeded collection)
is regenerated by Cell 1 (~15 min, exact) and not stored in git; a
frozen copy is available as a Release asset (link added at publication).
Collection protocol, split, and file schemas:
data/README.md.

#### Verification-suite provenance
checks/run_checks.py is re-assembled from the paper's Appendix A
specification (the original Phase-A session scripts were not preserved as
standalone files); the historical run's parameters and tolerances — 4061
zero-set predicate points, the 5.2×10⁻¹⁶ orthogonal-preservation
measurement, ≤1-step escape timing — are encoded as the checks' seeds and
tolerances. Checks tagged [spec] verify the mathematical property;
[impl] checks import the shipped package where available. CI runs the
suite on every push.

#### Reproducibility disclosures
The released Tier-1 instrument implements float32 where its docstring
states float64 (the artifact governs; paper App. B).
The 30-seed replication used a recipe recovered by pre-registered
fingerprint calibration against the original artifact medians.
The oracle-rollout fidelity gate detected that naive pymunk state
teleportation is invalid (55.9 px free-space divergence) and
auto-selected exact replay-based rollouts; the gate ships in the harness.
E1/E2 baselines share a CEM random stream and are counted as one
effective condition (Amendment A11).
Reproduction is notebook-based: a single checkpointed, idempotent harness
whose cells map one-to-one to every table and figure.

#### Environment
Developed and executed on Kaggle: Python 3.12, PyTorch 2.x, pymunk 6.11.1,
single P100/T4-class GPU, internet enabled for installs and dataset
attachment. CPU is sufficient for checks/ and Cell 6.

#### Citation
               @misc{mostafa2026degp,
               author  = {Mostafa, Mohsen},
               title   = {{DEGP}: Decoupled Entropic-Geometric Planning (artifact)},
               year    = {2026},
               version = {1.0.0},
               url     = {https://github.com/<USER>/degp-artifact},
               note    = {Artifact for ``Verified Conditioning, Measured Mechanisms,
                      and a Mapped Contact-Prediction Boundary in Latent-World-Model
                      Deployment''}
             }
