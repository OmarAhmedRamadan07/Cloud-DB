# feature_engineering.py
from pyspark.sql import functions as F

def create_indicators(df):
    """Create new features for engagement, loyalty, and income ratio."""
    print("--- Creating Engineered Features ---")
    # 1. Engagement Indicator
    df_eng = df.withColumn("Engagement_Indicator",
        F.round((F.col("JobSatisfaction") + F.col("EnvironmentSatisfaction") + F.col("WorkLifeBalance")) / 3.0, 2))
    # 2. Loyalty Indicator
    df_loy = df_eng.withColumn("Loyalty_Indicator",
        F.round(F.col("YearsAtCompany") / (F.col("TotalWorkingYears") + 1), 2))
    # 3. Income to Experience Ratio
    return df_loy.withColumn("Income_Experience_Ratio",
        F.round(F.col("MonthlyIncome") / (F.col("TotalWorkingYears") + 1), 2))
    