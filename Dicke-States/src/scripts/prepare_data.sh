#!/bin/bash

cd ..
mkdir datasets
curl -L -o datasets/sp-500-stocks.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/sp-500-stocks
cd datasets
unzip sp-500-stocks.zip
rm sp500_index.csv
rm sp500_stocks.csv
rm sp-500-stocks.zip
cd ..
cd scripts
python download_assets_data.py
python preprocessing_data.py
cd ..
cd datasets
rm sp500_companies.csv

