# Intracranial EEG Analysis

Research scripts for studying cortical responses to peripheral nerve stimulation using micro-ECoG and microelectrode-array recordings. The work covers preprocessing, channel-quality assessment, response features, and exploratory machine learning.

**Completed exploratory project.** This repository preserves the analysis code and historical diagnostic figures. It contains MNE-based electrophysiology analysis, random forests, and clustering experiments.

## Research focus

- Comparing channel-quality approaches, including RANSAC and AutoReject, across epidural and intracortical recordings.
- Examining evoked responses, spectral activity, response latency, and amplitude.
- Exploring whether features of the recordings distinguish stimulation conditions.

The experiments concern stimulation-related activity and nociception under anaesthesia. The analyses do not establish conscious pain or a validated clinical biomarker.

## Repository contents

| Directory | Contents |
| --- | --- |
| `general/` | MATLAB and TDT loading helpers |
| `preprocessing_ECoG/` | Filtering, epoch creation, RANSAC, and montage exploration |
| `preprocessing_MEA/` | Microelectrode-array preprocessing and historical diagnostic figures |
| `feature_extraction/` | Latency and mean-amplitude analyses |
| `ml/` | Random-forest and clustering experiments |

## Reading and running the code

The scripts retain experiment-specific paths, channel maps, and processing choices. Raw recordings are not included. Review those assumptions before adapting an analysis to another dataset; some historical scripts start processing when run or imported.

The random-forest experiment splits trials within each animal before combining the partitions. Its accuracy therefore describes an exploratory within-animal baseline, rather than generalisation to unseen animals.

Dependencies are listed in `requirements.txt`; Python 3.11 was used for the small software checks:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m ml.random_forest --help
```

The checks use synthetic inputs and cover loader behaviour and the random-forest helper. They do not reproduce the experimental results.
