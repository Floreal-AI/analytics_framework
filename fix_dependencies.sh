#!/bin/bash
# Script to help resolve dependency issues in the analytics_framework project

echo "🛠️ Analytics Framework Dependency Fixer 🛠️"
echo "----------------------------------------"
echo "This script will help resolve dependency conflicts."
echo

# Check if pipenv is installed
if ! command -v pipenv &> /dev/null; then
    echo "❌ Pipenv is not installed. Installing now..."
    pip install pipenv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install pipenv. Please install it manually:"
        echo "   pip install pipenv"
        exit 1
    fi
fi

echo "✅ Pipenv is installed."
echo

# Clean existing environment
echo "🧹 Cleaning existing environment..."
pipenv --rm 2>/dev/null || true
echo "✅ Environment cleaned."
echo

# Create fresh environment with Python 3.10
echo "🔄 Creating fresh environment with Python 3.10..."
pipenv --python 3.10
echo "✅ Environment created."
echo

# Install dependencies one by one
echo "📦 Installing dependencies one by one..."
dependencies=(
    "bittensor>=6.0.0"
    "torch>=2.0.0"
    "pandas>=1.5.0"
    "faker>=18.0.0"
    "scipy>=1.9.0"
    "scikit-learn>=1.2.0"
    "numpy>=1.23.0"
    "loguru>=0.6.0"
    "password-strength==0.0.3.post2"
)

for dep in "${dependencies[@]}"; do
    echo "   Installing $dep..."
    pipenv install "$dep"
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: Failed to install $dep. Continuing with other dependencies."
    fi
done

echo "✅ Dependencies installed."
echo

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pipenv install -e .
if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Failed to install package in development mode."
else
    echo "✅ Package installed in development mode."
fi
echo

# Test imports
echo "🧪 Testing imports..."
pipenv run python test_imports.py
echo

echo "🎉 Setup completed!"
echo
echo "To activate the environment, run:"
echo "   pipenv shell"
echo
echo "If you encounter any issues, please refer to INSTALL.md for troubleshooting steps." 