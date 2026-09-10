# ViTPain — Intracranial EEG Analysis

Exploratory Python code for analysing cortical responses to peripheral nerve stimulation, with preprocessing, channel-quality assessment, feature extraction, and classical machine-learning experiments.

The repository began with a vision-transformer direction, which explains its name. **The current code contains no ViT model or training pipeline.** The implemented work is MNE-based electrophysiology analysis and random-forest/clustering experiments. Transformer development remains a future milestone.

## Repository map

| Directory | Contents |
| --- | --- |
| `general/` | MATLAB and TDT loading helpers |
| `preprocessing_ECoG/` | Filtering, epoch creation, RANSAC, and montage exploration |
| `preprocessing_MEA/` | Microelectrode-array preprocessing and historical diagnostic plots |
| `feature_extraction/` | Response latency and mean-amplitude analysis |
| `ml/` | Random forests and clustering experiments |
| `tests/` | Small synthetic checks; no animal recordings required |

## Setup and checks

Python 3.11 is the verification baseline. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m ml.random_forest --help
```

To run the legacy random-forest baseline against your own compatible MNE epochs:

```sh
python -m ml.random_forest --data-dir path/to/epochs
```

The default experiment names and file convention are shown in `--help`. This command trains a model; the unit checks do not. Other historical scripts still contain experiment-specific paths and may execute processing when run or imported. Review their input/output paths, channel mappings, units, and parameters first. Raw recordings are not included, and the saved figures alone do not reproduce an analysis.

## Research scope

This work concerns stimulation-related activity and nociception under anaesthesia. It does not establish conscious pain or a validated clinical biomarker.

The legacy random-forest script splits trials **within each animal**, then combines those partitions. Its accuracy is a within-animal exploratory baseline, not evidence of generalisation to unseen animals. The split is preserved in this cleanup; subject-held-out evaluation is a separate milestone before any generalisation claim.

## Next steps

The [roadmap](ROADMAP.md) separates the current analysis code from planned data configuration, animal-level evaluation, and transformer modelling. The longer-term modelling direction is a dual-branch approach combining spectrograms and raw intracranial EEG; it is not an implemented capability of this repository.

[Nickolaj Ajay Atchuthan](https://atchuthan.com/)
