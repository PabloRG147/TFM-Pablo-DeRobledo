# Scripts

## Description

This folder contains two Python scripts and a bash script that automate the process of downloading and preprocessing asset data.

### Files

* `download_assets_data.py`: Downloads historical market data for S&P 500 assets using the Yahoo Finance API.
* `preprocessing_data.py`: Performs DataFrame concatenation, asset anonymization, and grouping of assets by sector.
* `prepare_data.sh`: A bash script that automates the entire data pipeline, executing both Python scripts sequentially.

## How to Use

Run `prepare_data.sh` to download and preprocess the asset data:
```bash
bash prepare_data.sh
```

> **Note:** The preprocessed dataset with anonymized asset identifiers is already available in the `datasets/` folder for reproducibility. Running this script will overwrite the existing data.
