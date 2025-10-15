"""
Configuration settings for E-commerce AI Platform
"""

import os
from typing import Dict, Any

# Azure Configuration
AZURE_CONFIG = {
    "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
    "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "ecommerce-ai-rg"),
    "location": os.getenv("AZURE_LOCATION", "eastus"),
    "databricks_workspace": os.getenv("DATABRICKS_WORKSPACE_URL"),
    "storage_account": os.getenv("AZURE_STORAGE_ACCOUNT"),
    "key_vault_name": os.getenv("AZURE_KEY_VAULT_NAME")
}

# Database Configuration
DATABASE_CONFIG = {
    "catalog_name": "ecommerce_platform",
    "bronze_schema": "bronze",
    "silver_schema": "silver", 
    "gold_schema": "gold",
    "checkpoint_location": "/mnt/ecommerce/checkpoints",
    "data_location": "/mnt/ecommerce/data"
}

# ML Configuration
ML_CONFIG = {
    "experiment_name": "/Shared/ecommerce-recommendations",
    "model_registry_name": "ecommerce_recommendation_models",
    "als_params": {
        "rank": 50,
        "maxIter": 20,
        "regParam": 0.1,
        "implicitPrefs": True
    },
    "kmeans_params": {
        "k": 5,
        "maxIter": 20,
        "seed": 42
    }
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 30,
    "max_recommendations": 50
}

# Data Quality Thresholds
DATA_QUALITY_CONFIG = {
    "min_quality_score": 0.8,
    "max_null_percentage": 0.05,
    "max_duplicate_percentage": 0.01,
    "freshness_threshold_hours": 24
}

# Recommendation System Configuration
RECOMMENDATION_CONFIG = {
    "default_recommendations": 10,
    "max_recommendations": 50,
    "min_interactions_for_cf": 5,
    "content_weight": 0.3,
    "collaborative_weight": 0.7,
    "popularity_fallback": True
}

# A/B Testing Configuration
AB_TEST_CONFIG = {
    "test_duration_days": 14,
    "min_sample_size": 1000,
    "statistical_significance": 0.05,
    "test_groups": ["A", "B"]
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "metrics_refresh_interval": "1 hour",
    "alert_thresholds": {
        "recommendation_latency_ms": 100,
        "model_accuracy_drop": 0.05,
        "data_quality_score": 0.8
    }
}

def get_config(config_type: str) -> Dict[str, Any]:
    """Get configuration by type"""
    configs = {
        "azure": AZURE_CONFIG,
        "database": DATABASE_CONFIG,
        "ml": ML_CONFIG,
        "api": API_CONFIG,
        "data_quality": DATA_QUALITY_CONFIG,
        "recommendation": RECOMMENDATION_CONFIG,
        "ab_test": AB_TEST_CONFIG,
        "monitoring": MONITORING_CONFIG
    }
    return configs.get(config_type, {})
