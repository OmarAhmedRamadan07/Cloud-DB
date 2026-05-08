# data_loader.py

def load_hr_data(spark, table_name):
    """
    Load HR data directly from Databricks Unity Catalog Table.
    """
    print(f"--- Loading HR Data from Catalog ---")
    # Read the data as a table from the catalog
    df = spark.table(table_name)
    
    print(f"Data loaded successfully!")
    print(f"Total Rows: {df.count()}")
    print(f"Total Columns: {len(df.columns)}")
    return df