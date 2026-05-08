# eda_analysis.py
from pyspark.sql import functions as F

def analyze_attrition_distribution(df):
    """Analyze the distribution of employee attrition."""
    return df.groupBy("Attrition").count().withColumnRenamed("count", "Total_Employees")

def analyze_salary_vs_performance(df):
    """Compare average monthly income based on performance ratings."""
    return df.groupBy("PerformanceRating") \
             .agg(F.round(F.avg("MonthlyIncome"), 2).alias("Average_Monthly_Income")) \
             .orderBy("PerformanceRating")

def analyze_department_insights(df):
    """Extract average income and attrition rate per department."""
    dept_insights = df.groupBy("Department") \
                      .agg(
                          F.count("EmployeeNumber").alias("Total_Employees"),
                          F.round(F.avg("MonthlyIncome"), 2).alias("Avg_Income"),
                          F.sum(F.when(F.col("Attrition") == "Yes", 1).otherwise(0)).alias("Attrition_Count")
                      )
    return dept_insights.withColumn("Attrition_Rate_%", 
                                   F.round((F.col("Attrition_Count") / F.col("Total_Employees")) * 100, 2))