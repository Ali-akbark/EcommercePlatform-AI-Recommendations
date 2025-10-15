# Create the .gitignore file
gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Azure Functions
.vscode/
local.settings.json
bin
obj

# Data files
*.csv
*.json
*.parquet
*.pkl
*.pickle
*.h5
*.hdf5

# Model files
*.model
*.pkl
models/
artifacts/
mlruns/

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Azure & Cloud
.azure/
*.publishsettings

# Databricks
.databricks/
databricks.cfg

# MLflow
mlruns/
mlartifacts/

# Power BI
*.pbix
*.pbit

# Terraform
*.tfstate
*.tfstate.*
*.tfplan
*.tfvars
.terraform/

# Docker
.dockerignore
Dockerfile.*

# Secrets and config
config.json
secrets.json
connection_strings.json
*.key
*.pem
*.p12
"""

# Write .gitignore
with open(f"{project_name}/.gitignore", "w") as f:
    f.write(gitignore_content)

print("✅ Created .gitignore")
print("📋 Covers: Python, Jupyter, Azure, MLflow, Databricks, Docker, Terraform, Data files, Models, Secrets")