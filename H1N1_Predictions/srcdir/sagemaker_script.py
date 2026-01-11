from sagemaker.xgboost import XGBoost
import sagemaker
import boto3
import os

os.environ['AWS_DEFAULT_REGION'] = "us-east-2"

boto_session = boto3.Session(region_name="us-east-2")
sess = sagemaker.Session(boto_session=boto_session)


bucket = "h1n1-project"

s3_train_path = f"s3://{bucket}/raw_data/"  # folder containing both CSVs

role = os.environ['SAGEMAKER_ROLE']

xgb_estimator = XGBoost(
    entry_point="train.py",
    role=role,
    sagemaker_session=sess,
    instance_type="ml.m5.large",
    instance_count=1,
    source_dir = '/home/ec2-user/Script/srcdir',
    framework_version="1.7-1",
    output_path=f"s3://{bucket}/xgb-output",
    hyperparameters={
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    }
)

xgb_estimator.fit({"train": s3_train_path})
