# Test Suite Documentation

This directory contains comprehensive tests for the Bittensor Analytics Framework, including unit tests, integration tests, and end-to-end pipeline validation.

## 📁 Directory Structure

```
tests/
├── README.md                              # This documentation
├── conftest.py                           # pytest configuration and fixtures
├── pytest_plugins.py                    # Custom pytest plugins
├── test_utils.py                        # Utility functions for testing
├── test_data_api.py                     # Data API tests
├── unit/                                # Unit tests by component
├── integration/                         # Integration tests
└── [Pipeline Tests]                     # Comprehensive pipeline tests
    ├── test_challenge_generation.py     # Challenge generation testing
    ├── test_complete_reward_breakdown.py # Detailed reward calculation analysis
    ├── test_end_to_end_pipeline.py     # Complete pipeline with mock data
    ├── test_one_by_one_processing.py   # Sequential challenge processing
    ├── test_real_api_complete.py       # Real API integration tests
    ├── test_real_api_pipeline.py       # Real API pipeline testing
    ├── test_real_model_predictions.py  # Real model prediction testing
    ├── test_reward_calculation_detailed.py # Reward calculation deep dive
    └── test_xgboost_training.py        # XGBoost model training tests
```

## 🧪 Test Categories

### **1. Unit Tests** (`unit/`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Protocol, validators, miners, utilities
- **Execution**: Fast, no external dependencies

### **2. Integration Tests** (`integration/`)
- **Purpose**: Test component interactions
- **Coverage**: API clients, data flow, validator-miner communication
- **Execution**: Medium speed, may use mock external services

### **3. Pipeline Tests** (Root level `test_*.py`)
- **Purpose**: End-to-end workflow validation
- **Coverage**: Complete data flow from API to rewards
- **Execution**: Slower, may use real APIs

## 🎯 Key Test Files

### **Core Pipeline Tests**

#### `test_end_to_end_pipeline.py`
- **Purpose**: Complete pipeline testing with mock data
- **Features**:
  - 9-step pipeline validation
  - Detailed input/output tracing
  - Performance measurement
  - Comprehensive logging

#### `test_real_api_complete.py`
- **Purpose**: Real API integration testing
- **Features**:
  - Actual VoiceForm API calls
  - Real training data (100 samples)
  - Test data one-by-one fetching (`limit=1, offset=0,1,2...`)
  - Real model predictions
  - Performance analysis

#### `test_one_by_one_processing.py`
- **Purpose**: Sequential challenge processing validation
- **Features**:
  - Challenge generation → Miner prediction → Reward calculation
  - Step-by-step timing analysis
  - Multiple template testing
  - Configurable delays and validation

### **Specialized Analysis Tests**

#### `test_complete_reward_breakdown.py`
- **Purpose**: Deep dive into reward calculation mechanics
- **Features**:
  - Formula breakdown and constants analysis
  - Scenario-based testing
  - Component reward analysis (classification, prediction, timing)

#### `test_reward_calculation_detailed.py`
- **Purpose**: Reward system validation
- **Features**:
  - Perfect vs imperfect prediction scenarios
  - Time-based reward calculations
  - Edge case handling

#### `test_real_model_predictions.py`
- **Purpose**: Real model behavior analysis
- **Features**:
  - Actual XGBoost model inference
  - Feature preprocessing validation
  - Prediction format handling
  - Error recovery testing

### **Component Tests**

#### `test_challenge_generation.py`
- **Purpose**: Challenge generation system testing
- **Features**:
  - Template-based conversation generation
  - Feature validation
  - Realistic scenario creation

#### `test_xgboost_training.py`
- **Purpose**: Model training and loading validation
- **Features**:
  - Model initialization
  - Training data processing
  - Prediction accuracy validation

## 🚀 Running Tests

### **Run All Tests**
```bash
pytest tests/
```

### **Run Specific Test Categories**
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Pipeline tests only
pytest tests/test_*.py

# Specific test file
pytest tests/test_real_api_complete.py
```

### **Run Real API Tests** (Requires Environment Variables)
```bash
export VOICEFORM_API_BASE_URL="https://your-api-url"
export VOICEFORM_API_KEY="your-api-key"

pytest tests/test_real_api_complete.py
pytest tests/test_real_api_pipeline.py
```

### **Performance Testing**
```bash
# Quick performance test
pytest tests/test_one_by_one_processing.py -v

# Complete pipeline performance
pytest tests/test_end_to_end_pipeline.py -v
```

## 📊 Test Results

Test results are automatically saved to the `../test-results/` directory:

- **JSON Results**: Detailed execution traces and metrics
- **Summary Reports**: Human-readable analysis documents
- **Coverage Reports**: Code coverage analysis
- **Performance Logs**: Timing and performance metrics

### **Key Result Files**
- `complete_real_api_results.json` - Real API test detailed results
- `one_by_one_test_results.json` - Sequential processing results
- `pipeline_trace.json` - Complete pipeline execution trace
- `REAL_API_TESTING_SUMMARY.md` - Real API testing analysis
- `PIPELINE_TESTING_SUMMARY.md` - Overall pipeline testing summary

## 🔧 Test Configuration

### **Environment Variables**
```bash
# Required for real API tests
VOICEFORM_API_BASE_URL=https://your-api-endpoint
VOICEFORM_API_KEY=your-api-key

# Optional test configuration
CONVERSION_LOG_LEVEL=INFO
CONVERSION_TEST_MODE=true
```

### **Mock vs Real Testing**
- **Mock Mode**: Fast, no external dependencies, predictable results
- **Real Mode**: Slower, requires API access, real-world validation

Most tests automatically detect environment configuration and switch between mock and real modes.

## 📈 Test Metrics

### **Coverage Goals**
- **Unit Tests**: >90% code coverage
- **Integration Tests**: >80% component interaction coverage
- **Pipeline Tests**: 100% end-to-end workflow coverage

### **Performance Benchmarks**
- **Unit Tests**: <1s per test
- **Integration Tests**: <10s per test
- **Pipeline Tests**: <30s per complete pipeline

### **Reliability Standards**
- **Success Rate**: >95% for all test categories
- **Flakiness**: <5% random failures
- **Reproducibility**: Deterministic results with fixed seeds

## 🐛 Debugging Tests

### **Common Issues**
1. **Feature Count Mismatch**: Training data (42 features) vs test data (43 features)
2. **API Configuration**: Missing environment variables for real API tests
3. **Model Loading**: XGBoost model file path issues

### **Debug Commands**
```bash
# Verbose test output
pytest tests/test_name.py -v -s

# Debug specific failure
pytest tests/test_name.py::test_function_name --pdb

# Show test logs
pytest tests/test_name.py --log-cli-level=DEBUG
```

## 🔄 Continuous Integration

These tests are designed to run in CI/CD pipelines with different configurations:

- **PR Tests**: Unit + Integration tests (fast feedback)
- **Nightly Tests**: Full pipeline + Real API tests (comprehensive validation)
- **Release Tests**: All tests + Performance benchmarks (quality gate)

## 📚 Adding New Tests

### **Test Naming Convention**
- `test_component_functionality.py` for unit tests
- `test_integration_workflow.py` for integration tests
- `test_end_to_end_scenario.py` for pipeline tests

### **Test Structure Template**
```python
#!/usr/bin/env python3
"""
Test Description
===============

Brief description of what this test validates.
"""

import pytest
import asyncio
from conversion_subnet.component import Component

class TestComponent:
    """Test class for Component functionality."""
    
    def test_basic_functionality(self):
        """Test basic component operation."""
        # Arrange
        component = Component()
        
        # Act
        result = component.process()
        
        # Assert
        assert result is not None
        assert result.is_valid()
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async component operation."""
        # Implementation here
        pass

if __name__ == "__main__":
    pytest.main([__file__])
```

This comprehensive test suite ensures the reliability, performance, and correctness of the entire Bittensor Analytics Framework. 