import enum
import numpy as np

class ModelType(enum.Enum):
    M_LR = "lr"
    M_RF = "rf"
    M_SVM = 'svm'
    M_GBDT = "gbdt"

# Remove models with outlier for 100-interation test
# Calculate SHAP values

# 4/8/2021
lr_tuned_parameters = {
    'max_iter': 100,
    'tol': 0.10,
    'C': 1.25
    }

# 4/8/2021
rf_tuned_parameters = {
    'n_estimators': 500,
    'max_depth': 50,
    'min_samples_leaf': 20,
    'min_samples_split': 8,
    'criterion': ["entropy"]
}

# 4/8/2021
svm_tuned_parameters = {
    'C': 500,
    'kernals': 'poly'
    }

# 4/8/2021
gbdt_tuned_parameters = {
    'learning_rate': 0.25,
    'n_estimators': 500,
    'min_samples_splits': 0.6
    }