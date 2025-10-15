"""
Utility functions for E-commerce AI Platform
"""

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityUtils:
    """Data quality utilities"""

    @staticmethod
    def calculate_quality_score(df: DataFrame, key_columns: List[str]) -> float:
        """Calculate overall data quality score"""
        total_records = df.count()
        if total_records == 0:
            return 0.0

        # Check null values in key columns
        null_count = 0
        for col in key_columns:
            null_count += df.filter(F.col(col).isNull()).count()

        # Check duplicates
        duplicate_count = total_records - df.distinct().count()

        # Calculate score
        null_rate = null_count / (total_records * len(key_columns))
        duplicate_rate = duplicate_count / total_records
        quality_score = max(0, 1 - null_rate - duplicate_rate)

        return quality_score

    @staticmethod
    def add_quality_metadata(df: DataFrame, table_name: str, quality_score: float) -> DataFrame:
        """Add quality metadata columns to DataFrame"""
        return df.withColumn("data_quality_score", F.lit(quality_score)) \
                 .withColumn("ingestion_timestamp", F.current_timestamp()) \
                 .withColumn("source_table", F.lit(table_name)) \
                 .withColumn("ingestion_date", F.current_date())

class RecommendationUtils:
    """Recommendation system utilities"""

    @staticmethod
    def calculate_implicit_rating(event_type: str) -> float:
        """Convert event type to implicit rating"""
        rating_map = {
            "purchase": 5.0,
            "add_to_cart": 3.0, 
            "page_view": 1.0,
            "wishlist_add": 2.0,
            "remove_from_cart": -1.0
        }
        return rating_map.get(event_type, 0.0)

    @staticmethod
    def apply_segment_boost(base_score: float, segment: int) -> float:
        """Apply customer segment boost to recommendation score"""
        segment_multipliers = {
            0: 1.2,  # High-value customers
            1: 1.1,  # Regular customers
            2: 1.0,  # Average customers
            3: 0.9,  # Low-frequency customers
            4: 0.8   # At-risk customers
        }
        return base_score * segment_multipliers.get(segment, 1.0)

    @staticmethod
    def diversify_recommendations(recs_df: DataFrame, category_col: str = "category") -> DataFrame:
        """Ensure diversity in recommendations by category"""
        from pyspark.sql.window import Window

        # Add diversity score based on category distribution
        user_window = Window.partitionBy("user_id")
        recs_with_diversity = recs_df.withColumn(
            "category_count", 
            F.count("*").over(user_window.partitionBy(category_col))
        ).withColumn(
            "diversity_penalty",
            F.when(F.col("category_count") > 3, 0.9).otherwise(1.0)
        ).withColumn(
            "diversified_score",
            F.col("hybrid_score") * F.col("diversity_penalty")
        )

        return recs_with_diversity

class ModelUtils:
    """Machine learning model utilities"""

    @staticmethod
    def prepare_als_data(interactions_df: DataFrame) -> Tuple[DataFrame, object, object]:
        """Prepare data for ALS training with proper indexing"""
        from pyspark.ml.feature import StringIndexer

        # Create indexers
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
        item_indexer = StringIndexer(inputCol="product_id", outputCol="item_index")

        # Fit and transform
        user_indexer_model = user_indexer.fit(interactions_df)
        item_indexer_model = item_indexer.fit(interactions_df)

        indexed_df = user_indexer_model.transform(interactions_df)
        indexed_df = item_indexer_model.transform(indexed_df)

        return indexed_df, user_indexer_model, item_indexer_model

    @staticmethod
    def calculate_recommendation_metrics(predictions_df: DataFrame, actual_df: DataFrame, k: int = 10) -> Dict[str, float]:
        """Calculate recommendation system evaluation metrics"""
        # This is a simplified version - in practice, you'd implement proper precision@k, recall@k, etc.
        metrics = {
            "precision_at_k": 0.75,  # Placeholder
            "recall_at_k": 0.68,     # Placeholder
            "ndcg_at_k": 0.82,       # Placeholder
            "coverage": 0.85         # Placeholder
        }
        return metrics

class InventoryUtils:
    """Inventory management utilities"""

    @staticmethod
    def calculate_reorder_point(avg_demand: float, lead_time_days: int, safety_factor: float = 1.5) -> int:
        """Calculate optimal reorder point"""
        return int(avg_demand * lead_time_days * safety_factor)

    @staticmethod
    def calculate_economic_order_quantity(annual_demand: float, order_cost: float, holding_cost: float) -> int:
        """Calculate Economic Order Quantity (EOQ)"""
        import math
        eoq = math.sqrt((2 * annual_demand * order_cost) / holding_cost)
        return int(eoq)

    @staticmethod
    def detect_stockout_risk(current_stock: int, avg_daily_demand: float, lead_time_days: int) -> str:
        """Detect stockout risk level"""
        days_of_stock = current_stock / max(avg_daily_demand, 0.1)

        if days_of_stock <= lead_time_days:
            return "HIGH"
        elif days_of_stock <= lead_time_days * 1.5:
            return "MEDIUM"
        else:
            return "LOW"

class APIUtils:
    """API and serving utilities"""

    @staticmethod
    def format_recommendation_response(recs_df: DataFrame, user_id: str) -> Dict:
        """Format recommendations for API response"""
        recs_list = []
        for row in recs_df.collect():
            rec = {
                "product_id": row.product_id,
                "product_name": row.product_name,
                "category": row.category,
                "price": row.price,
                "rating": row.rating,
                "score": round(row.hybrid_score, 3),
                "reason": row.recommendation_type
            }
            recs_list.append(rec)

        return {
            "user_id": user_id,
            "recommendations": recs_list,
            "timestamp": datetime.now().isoformat(),
            "model_version": "v1.0"
        }

    @staticmethod
    def validate_recommendation_request(request_data: Dict) -> Tuple[bool, str]:
        """Validate recommendation API request"""
        required_fields = ["user_id"]

        for field in required_fields:
            if field not in request_data:
                return False, f"Missing required field: {field}"

        num_recs = request_data.get("num_recommendations", 10)
        if not isinstance(num_recs, int) or num_recs <= 0 or num_recs > 50:
            return False, "num_recommendations must be an integer between 1 and 50"

        return True, "Valid request"

class MonitoringUtils:
    """Monitoring and alerting utilities"""

    @staticmethod
    def calculate_model_drift(current_metrics: Dict, baseline_metrics: Dict, threshold: float = 0.05) -> Dict:
        """Calculate model drift indicators"""
        drift_indicators = {}

        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                drift = abs(current_value - baseline_value) / baseline_value
                drift_indicators[metric] = {
                    "drift_percentage": drift,
                    "is_drifted": drift > threshold,
                    "current_value": current_value,
                    "baseline_value": baseline_value
                }

        return drift_indicators

    @staticmethod
    def generate_health_report(quality_scores: Dict, performance_metrics: Dict) -> Dict:
        """Generate system health report"""
        overall_health = "HEALTHY"

        if any(score < 0.8 for score in quality_scores.values()):
            overall_health = "DEGRADED"

        if any(metric.get("is_drifted", False) for metric in performance_metrics.values()):
            overall_health = "AT_RISK"

        return {
            "overall_health": overall_health,
            "data_quality": quality_scores,
            "model_performance": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()
