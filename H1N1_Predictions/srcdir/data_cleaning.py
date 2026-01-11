import boto3
import pandas as pd
import io
import warnings
#from timing import Timer
#warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
#warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopy)


s3 = boto3.client('s3')

#timer = Timer()

def list_buckets(s3 = s3):
    #List available buckets
    buckets = s3.list_buckets()
    for bucket in buckets['Buckets']:
        print(bucket['Name'])

def read_data(bucket_name = 'h1n1-project', file_key = 'raw_data/training_set_features.csv'):
    # Get the object
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    
    # Read it into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    return df


def replace_nan_by_filter(df, column, method):
    
    nan_mask = df[column].isna()
    good_data = df[column][~nan_mask]
    
    if method == 'mean':
        value = int(good_data.mean())

    elif method == 'median':
        value = int(good_data.median())

    elif method == 'mode':
        value = int(good_data.mode())
    
    data = df[column]
    data[nan_mask] = value
    df[column] =  data
    
    return df


def change_column(df, column):

    nan_mask = df[column].isna()
    education = df[column].values
    education[nan_mask] =  df[column][~nan_mask].mode().values
    df[column] = education

    return df

def validate_data(df):
    assert not df.isnull().any().any(), "NaNs detected!"
    assert df.shape[0] > 0, "Empty dataset!"

def clean_data_function(df):
    copy = df.copy()
    check = replace_nan_by_filter(copy, 'h1n1_concern', 'median')
    check = replace_nan_by_filter(check, 'h1n1_knowledge', 'median')
    check = replace_nan_by_filter(check, 'behavioral_antiviral_meds', 'median')
    check = replace_nan_by_filter(check, 'behavioral_avoidance', 'median')
    check = replace_nan_by_filter(check, 'behavioral_face_mask', 'median')
    check = replace_nan_by_filter(check, 'behavioral_wash_hands', 'median')
    check = replace_nan_by_filter(check, 'behavioral_large_gatherings', 'median')
    check = replace_nan_by_filter(check, 'behavioral_outside_home', 'median')
    check = replace_nan_by_filter(check, 'behavioral_touch_face', 'median')
    check = replace_nan_by_filter(check, 'doctor_recc_h1n1', 'median')
    check = replace_nan_by_filter(check, 'doctor_recc_seasonal', 'median')
    check = replace_nan_by_filter(check, 'chronic_med_condition', 'median')
    check = replace_nan_by_filter(check, 'child_under_6_months', 'median')
    check = replace_nan_by_filter(check, 'health_worker', 'median')
    check = replace_nan_by_filter(check, 'opinion_h1n1_vacc_effective', 'median')
    check = replace_nan_by_filter(check, 'opinion_h1n1_risk', 'median')
    check = replace_nan_by_filter(check, 'opinion_h1n1_sick_from_vacc', 'median')
    check = replace_nan_by_filter(check, 'opinion_seas_vacc_effective', 'median')
    check = replace_nan_by_filter(check, 'opinion_seas_risk', 'median')
    check = replace_nan_by_filter(check, 'opinion_seas_sick_from_vacc', 'median')
    check = replace_nan_by_filter(check, 'household_adults', 'median')
    check = replace_nan_by_filter(check, 'household_children', 'median')
    
    check = change_column(check, 'education')
    check = change_column(check, 'marital_status')
    check = change_column(check, 'rent_or_own')
    check = change_column(check, 'employment_status')
    check = change_column(check, 'income_poverty')

    age_map = {
                '18 - 34 Years': 1,
                '35 - 44 Years': 2,
                '45 - 54 Years': 3,
                '55 - 64 Years': 4,
                '65+ Years': 5
            }
    check['age_group'] = check['age_group'].map(age_map)

    edu_map = {
        '< 12 Years': 1,
        '12 Years': 2,
        'Some College': 3,
        'College Graduate': 4
    }
    check['education'] =check['education'].map(edu_map)

    check = pd.get_dummies(check, columns=['race'], drop_first=True, dtype = int)

    check['sex'] = check['sex'].map({'Female': 0, 'Male': 1})

    income_map = {
        'Below Poverty': 1,
        '<= $75,000, Above Poverty': 2,
        '> $75,000': 3
    }
    check['income_poverty'] = check['income_poverty'].map(income_map)

    check['marital_status'] = check['marital_status'].map({'Not Married': 0, 'Married': 1})

    check['rent_or_own'] = check['rent_or_own'].map({'Rent': 0, 'Own': 1})

    employment_map = {
        'Not in Labor Force': 0,
        'Employed': 1,
        'Unemployed': 2
    }
    check['employment_status'] = check['employment_status'].map(employment_map)

    cols_to_use = ['h1n1_concern', 'h1n1_knowledge',
                   'behavioral_antiviral_meds', 'behavioral_avoidance',
                   'behavioral_face_mask', 'behavioral_wash_hands',
                   'behavioral_large_gatherings', 'behavioral_outside_home',
                   'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
                   'chronic_med_condition', 'child_under_6_months', 'health_worker',
                   'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                   'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
                   'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
                   'education', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
                   'employment_status', 'household_adults',
                   'household_children',
                   'race_Hispanic', 'race_Other or Multiple', 'race_White']
    
    feature_df = check[cols_to_use]
    validate_data(feature_df)
    return feature_df

def ml_ready_data(bucket_name, file_key):
    
    df = read_data(bucket_name, file_key)
    clean_df = clean_data_function(df)
    
    return clean_df
