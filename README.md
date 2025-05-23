# GNNs Meet Sequence Models Along the Shortest-Path: An Expressive method for Link Prediction

## Anonymized Repository for NeurIPS 2025 Submission

This repository contains the source code and instructions to reproduce the experiments for our NeurIPS 2025 submission.

### Requirements

This project was developed and tested with Python 3.9. All dependencies can be installed via:

```bash
pip install -r requirements.txt
```
This project was developed and tested with Python 3.9.

### Real-world Experiments
#### Download the datasets

Before running experiments on real-world datasets, download the required files with:

```bash
bash download_data.sh
```

#### Run the experiments
You can change the dataset name in the arguments of the file

Small datasets

```bash
cd small
python main_shortestpath_CoraCiteseerPubmed.py
```

OGB datasets
```bash
python main_shortestpath_ogb.py  
```