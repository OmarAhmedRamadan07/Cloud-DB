# preprocessing.py
from pyspark.ml.feature import StringIndexer

def handle_missing_values(df):
    """
    Handle missing values by dropping incomplete rows.
    """
    print("--- Handling Missing Values ---")
    return df.dropna()

def encode_categorical_variables(df):
    """
    Convert categorical text variables into numerical indices.
    """
    print("--- Encoding Categorical Variables ---")
    categorical_columns = [
        'Attrition', 'BusinessTravel', 'Department', 'EducationField', 
        'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'
    ]
    encoded_df = df
    for col_name in categorical_columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_Index")
        encoded_df = indexer.fit(encoded_df).transform(encoded_df)
    return encoded_df

def run_preprocessing_pipeline(df):
    """
    Execute all preprocessing steps sequentially.
    """
    df_no_missing = handle_missing_values(df)
    return encode_categorical_variables(df_no_missing)