# Create the setup.bat script
setup_script = """@echo off
echo 🚀 E-commerce AI Platform - Setup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo ✅ Python is installed
python --version

echo.
echo 📁 Creating folder structure...
mkdir data\\raw 2>nul
mkdir data\\processed 2>nul  
mkdir data\\external 2>nul
mkdir models\\recommendation 2>nul
mkdir models\\forecasting 2>nul
mkdir models\\experiments 2>nul
mkdir logs 2>nul

echo ✅ Folder structure created

echo.
echo 📊 Initializing Git repository...
git init
git add .
git commit -m "Initial commit: E-commerce AI Platform setup" 2>nul

echo.
echo 🐍 Creating Python virtual environment...
if exist "venv" (
    echo Virtual environment already exists
) else (
    python -m venv venv
)

echo.
echo ⚡ Activating virtual environment...
call venv\\Scripts\\activate

echo.
echo 📦 Upgrading pip and installing dependencies...
python -m pip install --upgrade pip setuptools wheel

echo.
echo 🔧 Installing core packages first...
pip install pandas==2.1.4 numpy==1.24.3 scikit-learn==1.3.2

echo.
echo 📊 Installing ML and cloud packages...
pip install -r requirements.txt

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📝 Next Steps:
echo 1. Create GitHub repository: https://github.com/new
echo 2. Add remote: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
echo 3. Push code: git push -u origin main
echo 4. Open Databricks and import notebooks
echo.
echo 🎯 Ready for your next interview!
echo.
pause
"""

# Write setup.bat
with open(f"{project_name}/setup.bat", "w") as f:
    f.write(setup_script)

print("✅ Created setup.bat script")
print("📋 Features: Python check, venv creation, dependency installation, Git setup, next steps guidance")