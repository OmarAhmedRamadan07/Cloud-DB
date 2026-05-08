# models_classification.py
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

def prepare_features(df, feature_columns):
    """Assemble feature columns into a single vector column."""
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    return assembler.transform(df)

def build_attrition_model(df):
    """Build and evaluate a Random Forest model for attrition."""
    print("--- Building Attrition Classification Model ---")
    features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'EnvironmentSatisfaction', 
                'Engagement_Indicator', 'Loyalty_Indicator', 'OverTime_Index']
    model_df = prepare_features(df, features)
    train_data, test_data = model_df.randomSplit([0.8, 0.2], seed=42)
    rf = RandomForestClassifier(labelCol="Attrition_Index", featuresCol="features", numTrees=50)
    model = rf.fit(train_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="Attrition_Index", metricName="f1")
    print(f"Attrition Model F1-Score: {evaluator.evaluate(model.transform(test_data)):.4f}")
    return model, model.transform(test_data)

def build_performance_model(df):
    """Build and evaluate a Random Forest model for performance."""
    print("--- Building Performance Regression Model ---")
    features = ['Age', 'MonthlyIncome', 'Engagement_Indicator', 'YearsAtCompany', 'TrainingTimesLastYear']
    model_df = prepare_features(df, features)
    train_data, test_data = model_df.randomSplit([0.8, 0.2], seed=42)
    rf = RandomForestRegressor(labelCol="PerformanceRating", featuresCol="features", numTrees=50)
    model = rf.fit(train_data)
    evaluator = RegressionEvaluator(labelCol="PerformanceRating", metricName="rmse")
    print(f"Performance Model RMSE: {evaluator.evaluate(model.transform(test_data)):.4f}")
    return model, model.transform(test_data)