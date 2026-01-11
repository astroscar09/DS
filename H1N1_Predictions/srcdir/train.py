import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_cleaning import *

# Multi-label target columns
ycols = ['h1n1_vaccine', 'seasonal_vaccine']

# SageMaker injects these environment variables
train_dir = os.environ["SM_CHANNEL_TRAIN"]
model_dir = os.environ["SM_MODEL_DIR"]

# Load data from the mapped S3 channel
X = pd.read_csv(os.path.join(train_dir, "training_set_features.csv"))
X = clean_data_function(X).values
y = pd.read_csv(os.path.join(train_dir, "training_set_labels.csv"))[ycols].values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base XGBoost model
base_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
)

# Multi-label wrapper
model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate ROC-AUC
probs = model.predict_proba(X_val)
auc_h1n1 = roc_auc_score(y_val[:, 0], probs[0][:, 1])
auc_seasonal = roc_auc_score(y_val[:, 1], probs[1][:, 1])
print("H1N1 AUC:", auc_h1n1)
print("Seasonal AUC:", auc_seasonal)

# Save trained model to SageMaker model directory
joblib.dump(model, os.path.join(model_dir, "xgb_model.joblib"))
