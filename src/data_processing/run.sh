#!/bin/bash

# training

python data_processing.py \
  --features_path '/mnt/data1/chong/2021-CRC/updated_data/encoding_files/' \
  --features_filename 'data_CC0yr_expr_features.pkl' \
  --data_filename 'data_CC0yr_expr.pkl' \
  --output_path '/mnt/data1/chong/2021-CRC/updated_data/'
