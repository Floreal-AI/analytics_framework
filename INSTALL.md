# Installation Guide

This guide will help you install and set up the required dependencies for the Bittensor Conversion Subnet.

## Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/Floreal-AI/analytics_framework.git
cd analytics_framework

# Install dependencies using pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Option 2: Using pipenv

If you prefer using pipenv, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Floreal-AI/analytics_framework.git
cd analytics_framework

# Install pipenv if you don't have it
pip install pipenv

# Install dependencies
pipenv install

# Activate the virtual environment
pipenv shell
```

## Troubleshooting

### Dependency Conflicts

If you encounter dependency conflicts when using pipenv:

1. Try installing with pre-release versions:
   ```bash
   pipenv lock --pre
   pipenv install
   ```

2. If that doesn't work, install dependencies one by one:
   ```bash
   pipenv install bittensor>=6.0.0
   pipenv install torch>=2.0.0
   pipenv install pandas>=1.5.0
   # Continue with other dependencies
   ```

3. For persistent issues, use the manual approach:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies manually
   pip install -r requirements.txt
   pip install -e .
   ```

### Import Errors

If you experience import errors:

1. Verify the package is installed correctly:
   ```bash
   python test_imports.py
   ```

2. Check that your Python environment is correctly set up:
   ```bash
   # Verify the Python path includes your project
   python -c "import sys; print(sys.path)"
   ```

3. Make sure all requirements are installed:
   ```bash
   pip list | grep -E 'bittensor|torch|pandas|faker|scipy|scikit-learn|numpy'
   ```

## Testing Your Installation

After installation, run the test script to verify everything is working:

```bash
python test_imports.py
```

You should see several '✅ Success' messages indicating that all imports are working correctly. 