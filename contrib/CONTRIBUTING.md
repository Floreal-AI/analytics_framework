# Contributing to Bittensor Conversion Subnet

Thank you for your interest in contributing to the Bittensor Conversion Subnet project! This document provides guidelines and instructions for contributing to this open-source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Coding Standards](#coding-standards)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [Communication](#communication)

## Code of Conduct

This project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/analytics_framework.git
   cd analytics_framework
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```
4. Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to see if the problem has already been reported. When you are creating a bug report, please include as much detail as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the bug**
- **Provide specific examples** such as code snippets or configuration files
- **Describe the behavior you observed**
- **Explain the behavior you expected**
- **Include screenshots or terminal output** if applicable
- **Include details about your environment** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are welcome. When submitting an enhancement suggestion, include:

- **A clear and descriptive title**
- **A step-by-step description of the enhancement**
- **Any specific examples to demonstrate the enhancement**
- **A description of the current behavior and explanation of why it should be changed**
- **Explain why this enhancement would be useful to most users**

### Pull Requests

The process for submitting a pull request is as follows:

1. Ensure your fork is up-to-date with the main repository
2. Create a new branch for your feature or fix
3. Make your changes
4. Add or update tests as necessary
5. Run the test suite to ensure your changes don't break existing functionality
6. Update the documentation if necessary
7. Submit your pull request with a clear description of the changes

Pull requests should:

- Have a clear, focused scope
- Include tests for new functionality
- Follow the project's coding standards
- Update documentation as needed
- Include a description of the changes and their purpose

## Development Guidelines

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/Floreal-AI/analytics_framework.git
cd analytics_framework

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with testing dependencies
pip install -e ".[dev,test]"
```

### Coding Standards

This project follows standard Python coding conventions:

- Use 4 spaces for indentation
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use docstrings for modules, classes, and functions following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Limit line length to 100 characters

### Testing

All new features and bug fixes should include tests. To run the test suite:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=conversion_subnet tests/
```

## Project Structure

The project is organized as follows:

```
analytics_framework/
├── conversion_subnet/      # Core package code
│   ├── base/               # Base classes for miners and validators
│   ├── miner/              # Miner implementation
│   ├── validator/          # Validator implementation
│   └── utils/              # Shared utilities
├── neurons/                # Runnable neuron implementations
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Example code and notebooks
└── scripts/                # Utility scripts
```

## Specific Contribution Areas

Here are some specific areas where contributions are particularly welcome:

### Miners

- Improving prediction accuracy
- Adding new machine learning models
- Optimizing response time
- Implementing feature engineering techniques

### Validators

- Enhancing reward mechanisms
- Improving synthetic data generation
- Adding robust evaluation metrics
- Optimizing validator performance

### Documentation

- Improving examples and tutorials
- Adding explanations of the reward system
- Creating visualizations of the subnet architecture
- Writing guides for newcomers to Bittensor

### Infrastructure

- CI/CD improvements
- Docker containerization
- Monitoring tools
- Performance optimization

## Communication

- **Issues**: For bugs, enhancements, and discussions about specific features
- **Pull Requests**: For code contributions
- **Discussions**: For general questions and community discussion

Thank you again for contributing to the Bittensor Conversion Subnet project! 