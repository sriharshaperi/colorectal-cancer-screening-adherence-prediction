# Colorectal Cancer Prediction

## Table of Contents

Introduction
Objective
Data
Modeling Approach
Project Structure
Installation
Usage
Results
Contributing
License

## Introduction

Colorectal cancer is a significant public health concern worldwide. Early detection and adherence to colorectal cancer screening guidelines play a crucial role in improving public health outcomes and reducing cancer-related mortality. This project aims to predict the onset of colorectal cancer in individuals under 50 years old using real-world data and machine learning algorithms.

## Objective

The main objective of this project is to develop predictive models that can identify individuals at risk of colorectal cancer before the age of 50. The models will utilize demographic and clinical information, such as age, sex, income, insurance type, pre-medical history, etc., to predict the likelihood of adherence to colorectal cancer screening.

## Data

The dataset used in this project contains demographic and clinical information of individuals under 50 years old. The features include age, sex, income, insurance type, pre-medical history, and the target variable "adherence" indicating whether the patient is adherent to colorectal cancer screening guidelines (1: adherent, 0: non-adherent). The data is obtained from real-world sources and has undergone preprocessing to ensure data quality and consistency.

## Modeling Approach

Various machine learning algorithms will be employed for this prediction task, including:

## Logistic Regression (LR)

Support Vector Machines (SVMs)
Random Forests (RFs)
Gradient Boosting (GBT)
Hyperparameter tuning will be performed for each model to optimize their performance. The models will be evaluated using metrics such as F1-score, precision, recall, and AUC-ROC to assess their effectiveness in predicting colorectal cancer onset.

## Project Structure

The project is organized into the following directories and files:

data/: Contains the dataset used for modeling.
models/: Stores the trained machine learning models.
notebooks/: Jupyter notebooks for data exploration, modeling, and analysis.
scripts/: Python scripts for data preprocessing and model training.
utils/: Utility functions used throughout the project.
README.md: Project documentation.
requirements.txt: Lists the required Python packages for the project.

## Installation

To run this project locally, follow these steps:

### Clone the repository:

git clone https://github.com/your-username/colorectal-cancer-prediction.git

### Navigate to the project directory:

cd colorectal-cancer-prediction

### Install the required Python packages:

pip install -r requirements.txt

### Usage

To train the machine learning models and evaluate their performance, run the main script:

python scripts/train_models.py --train_test_path data/ --output_path models/ --model_type M_RF --run_times 100

Replace M_RF with the desired model type (M_LR, M_SVM, or M_GBT) to train and evaluate the corresponding model.

## Results

The results of the machine learning models, including F1-score, precision, recall, and AUC-ROC, will be displayed with confidence intervals. The models' effectiveness in predicting colorectal cancer onset will be analyzed based on these results.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code for your own purposes. Attribution is appreciated but not required.
