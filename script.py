import os
import json
from datetime import datetime

# Create the complete project structure for E-commerce Recommendation & Inventory System
project_name = "EcommercePlatform-AI-Recommendations"
current_date = datetime.now().strftime("%Y-%m-%d")

print(f"🚀 Creating {project_name} - Complete End-to-End Project")
print("=" * 60)

# Define the project structure
project_structure = {
    f"{project_name}/": {
        "README.md": "main_readme",
        "PROJECT_SUMMARY.md": "project_summary", 
        "requirements.txt": "requirements",
        ".gitignore": "gitignore",
        "setup.bat": "setup_script",
        
        "notebooks/": {
            "01_data_ingestion.py": "data_ingestion_nb",
            "02_data_transformation.py": "transformation_nb",
            "03_feature_engineering.py": "feature_engineering_nb",
            "04_recommendation_engine.py": "recommendation_nb",
            "05_inventory_forecasting.py": "inventory_nb",
            "06_model_deployment.py": "deployment_nb"
        },
        
        "src/": {
            "__init__.py": "init_file",
            "config.py": "config_file",
            "utils.py": "utils_file",
            "data_processing.py": "data_processing",
            "models.py": "models_file",
            "evaluation.py": "evaluation_file"
        },
        
        "data/": {
            "raw/": {},
            "processed/": {},
            "external/": {}
        },
        
        "models/": {
            "recommendation/": {},
            "forecasting/": {},
            "experiments/": {}
        },
        
        "dashboards/": {
            "power_bi/": {},
            "streamlit/": {}
        },
        
        "pipelines/": {
            "azure_data_factory/": {},
            "databricks_jobs/": {}
        },
        
        "infrastructure/": {
            "terraform/": {},
            "arm_templates/": {},
            "scripts/": {}
        },
        
        "docs/": {
            "architecture/": {},
            "api/": {},
            "user_guides/": {}
        }
    }
}

# Create the folder structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if name.endswith('/'):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            print(f"📁 Created directory: {path}")
            if isinstance(content, dict):
                create_structure(base_path, {os.path.join(name, k): v for k, v in content.items()})
        else:
            # It's a file - we'll create the content later
            pass

# Create the base directory
base_dir = project_name
os.makedirs(base_dir, exist_ok=True)
print(f"📁 Created main project directory: {base_dir}")

# Create subdirectories
for main_folder in ["notebooks", "src", "data", "models", "dashboards", "pipelines", "infrastructure", "docs"]:
    os.makedirs(os.path.join(base_dir, main_folder), exist_ok=True)
    print(f"📁 Created: {main_folder}/")

# Create sub-subdirectories
subdirs = [
    "data/raw", "data/processed", "data/external",
    "models/recommendation", "models/forecasting", "models/experiments",
    "dashboards/power_bi", "dashboards/streamlit",
    "pipelines/azure_data_factory", "pipelines/databricks_jobs",
    "infrastructure/terraform", "infrastructure/arm_templates", "infrastructure/scripts",
    "docs/architecture", "docs/api", "docs/user_guides"
]

for subdir in subdirs:
    os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    print(f"📁 Created: {subdir}/")

print("\n✅ Project structure created successfully!")
print(f"📊 Total directories: {len([d for d in subdirs]) + 9}")