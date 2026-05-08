# models_clustering.py
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

def segment_employees(df):
    """Segment employees into clusters using K-Means."""
    print("--- Segmenting Employees (K-Means Clustering) ---")
    cluster_features = ['MonthlyIncome', 'Engagement_Indicator', 'PerformanceRating', 'Loyalty_Indicator']
    assembler = VectorAssembler(inputCols=cluster_features, outputCol="features")
    model_df = assembler.transform(df)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaled_df = scaler.fit(model_df).transform(model_df)
    kmeans = KMeans(featuresCol="scaledFeatures", k=3, seed=42)
    model = kmeans.fit(scaled_df)
    predictions = model.transform(scaled_df)
    evaluator = ClusteringEvaluator(featuresCol="scaledFeatures")
    print(f"Clustering Silhouette Score: {evaluator.evaluate(predictions):.4f}")
    return predictions