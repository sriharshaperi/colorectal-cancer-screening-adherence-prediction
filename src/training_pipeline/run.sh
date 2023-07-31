#!/bin/bash

# training

python training.py \
  --train_test_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/encoding_files/' \
  --output_path '/Users/cdang/Documents/Github/BMI/colorectal-cancer-predication/' \
  --model_type 'lr' \
  --predication_window 1 \
  --number_of_jobs 10 \
  --n_iterations 20

