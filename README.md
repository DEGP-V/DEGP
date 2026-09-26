# DEGP
### DEGP: Decoupled Entropic-Geometric Planning — Artifact

Artifact for the paper "Decoupled Entropic-Geometric Planning (DEGP):Verified Conditioning, Measured Mechanisms, and a MappedContact-Prediction Boundary in Latent-World-Model Deployment"(M. Mostafa, 2026).

DEGP decouples a latent world model into an encoder trained with avariance–covariance objective only, a dynamics predictor trained ondetached latents with residual straightening and a Jacobian penalty,and a leashed asymmetric planner — under a runtime contract(ε·T budget, watchdog, fallback). The paper reports the fullpre-registered program including its negative results: deploymenton Push-T fails at 0/50 successes in every condition, and an18-condition pre-registered attribution battery locates the failureboundary: the self-supervised latents retain a contact signal(5.8× free/contact discrimination) that no predictor version learnsto use, together with the probe-decoded cost.

Every claim in the paper is labeled by evidence level (analyticalproof / executable verification / empirical validation), and falsifiedhypotheses are reported. This repository is the executable half ofthat contract.
