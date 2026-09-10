# Development milestones

The first milestone prepares this repository for public inspection. Later milestones are planned work, not implemented features or validated results.

## 01 · Public analysis-code baseline

Correct the repository description, add dependency/setup instructions, fix redundant forest fitting and loader failure cases, make the forest script import-safe with explicit input paths, and add synthetic checks. Acceptance: tests and help command pass without animal recordings; absent transformer code and legacy trial splits are clearly documented.

## 02 · Configured, animal-held-out baseline

Replace experiment-specific script paths with reviewed configuration; document signal units, channel maps, event definitions and QC. Build evaluation with animals held out and preprocessing fitted only on training data. Acceptance: split-overlap checks and a reproducible baseline report; no silent reinterpretation of historical within-animal results.

## 03 · Dual-branch transformer study

Implement the planned raw-EEG and spectrogram branches only after the baseline protocol is fixed. Compare raw-only, spectral-only and fused models with matched animal-level splits, seeds and model-selection rules. Acceptance: architecture/configuration, branch ablations, calibration and untouched-test evaluation are reproducible; performance remains unclaimed until measured.
