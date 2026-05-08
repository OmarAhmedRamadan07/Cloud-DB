# main.py
from pyspark.sql import SparkSession

# Import functions from the modular files
from data_loader import load_hr_data
from preprocessing import run_preprocessing_pipeline
from eda_analysis import analyze_attrition_distribution, analyze_salary_vs_performance, analyze_department_insights
from feature_engineering import create_indicators
from models_classification import build_attrition_model, build_performance_model
from models_clustering import segment_employees

if __name__ == "__main__":
    print("--- Starting AI-Powered HR Analytics Pipeline --- \n")

    # 1. Initialize SparkSession (Handling Databricks Connect setup)
    try:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.getOrCreate()
    except ImportError:
        spark = SparkSession.builder.getOrCreate()

    try:
        # 2. Use the exact table name from your Databricks Catalog
        table_name = "workspace.default.hr_employee_attrition"
        
        # Load Data using the new table name
        raw_df = load_hr_data(spark, table_name)

        # 3. Preprocessing
        clean_df = run_preprocessing_pipeline(raw_df)
        print("\nPreview of Cleaned Data:")
        clean_df.show(5)

        # 4. EDA (Exploratory Data Analysis)
        print("\nAttrition Distribution:")
        analyze_attrition_distribution(clean_df).show()

        print("\nSalary vs Performance:")
        analyze_salary_vs_performance(clean_df).show()

        print("\nDepartment Insights:")
        analyze_department_insights(clean_df).show()

        # 5. Feature Engineering
        feature_df = create_indicators(clean_df)
        print("\nPreview of Engineered Features:")
        feature_df.select("Department", "Engagement_Indicator", "Loyalty_Indicator", "Income_Experience_Ratio").show(5)

        # 6. AI Models Validation & Execution
        attrition_model, attrition_predictions = build_attrition_model(feature_df)
        performance_model, performance_predictions = build_performance_model(feature_df)
        cluster_predictions = segment_employees(feature_df)

        print("\nPreview of Employee Clusters:")
        cluster_predictions.select("Age", "MonthlyIncome", "Engagement_Indicator", "prediction").show(10)

        print("\n--- Pipeline Execution Completed Successfully! ---")

    finally:
        # Stop the session safely
        spark.stop()