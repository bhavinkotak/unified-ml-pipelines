# Unified ML Pipelines Code Project

**Author:** Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

Implementation of concepts from the research paper: *"Unified Machine Learning Pipeline Architecture for Enterprise-Scale Model Development and Deployment"*

## Overview

This code project demonstrates a complete, integrated ML pipeline architecture with:

- **Feature Store**: Centralized feature management with versioning
- **Model Registry**: Model versioning, lifecycle management, A/B testing support
- **Experiment Tracker**: MLflow-compatible experiment tracking
- **ML Pipeline**: End-to-end orchestration tying all components together

## Project Structure

```
unified-ml-pipelines/
├── src/
│   ├── __init__.py
│   ├── ml_pipeline.py        # Core pipeline components
│   └── data_generators.py    # Synthetic data generation
├── tests/
│   └── test_ml_pipeline.py   # Comprehensive test suite
├── data/                     # Generated data and artifacts
├── configs/                  # Configuration files
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.ml_pipeline import MLPipeline, MLPipelineConfig, TaskType
from src.data_generators import MLExperimentDataGenerator

# Generate synthetic data
data_gen = MLExperimentDataGenerator(random_seed=42)
df = data_gen.generate_classification_dataset(n_samples=2000)

# Configure pipeline
config = MLPipelineConfig(
    experiment_name="fraud_detection",
    task_type=TaskType.CLASSIFICATION,
    random_seed=42
)

# Initialize and run pipeline
pipeline = MLPipeline(config)
pipeline.prepare_data(df, target_column='target')
results = pipeline.train_and_evaluate('random_forest')

print(f"Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

### Feature Store Usage

```python
from src.ml_pipeline import FeatureStore

fs = FeatureStore("./my_features")

# Register features
fs.register_feature("customer_age", "int", description="Customer age in years")
fs.register_feature("total_spend", "float", description="Total spending amount")

# Ingest data
fs.ingest_features(df, ['customer_age', 'total_spend'])

# Retrieve features
features = fs.get_features(['customer_age', 'total_spend'], entity_ids=[1, 2, 3])
```

### Model Registry Usage

```python
from src.ml_pipeline import ModelRegistry, ModelStage

registry = ModelRegistry("./models")

# Register a trained model
model_id = registry.register_model(
    model=trained_model,
    model_name="fraud_detector",
    metrics={'accuracy': 0.95, 'f1': 0.92},
    parameters=model.get_params(),
    feature_names=['age', 'amount', 'distance']
)

# Promote to production
registry.transition_stage(model_id, ModelStage.PRODUCTION)

# Load production model
model, metadata = registry.load_model("fraud_detector", stage=ModelStage.PRODUCTION)
```

### Experiment Tracking

```python
from src.ml_pipeline import ExperimentTracker

tracker = ExperimentTracker("my_experiment")

run_id = tracker.start_run("baseline_model")
tracker.log_params({'learning_rate': 0.01, 'batch_size': 32})
tracker.log_metric('accuracy', 0.95)
tracker.log_metric('loss', 0.15)
tracker.end_run()

# Find best run
best = tracker.get_best_run('accuracy', maximize=True)
```

## Running the Demo

```bash
cd /path/to/unified-ml-pipelines
python src/ml_pipeline.py
```

This will:
1. Generate synthetic classification data
2. Configure the ML pipeline
3. Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
4. Compare model performance
5. Register the best model
6. Demonstrate feature store and model registry operations

## Running Tests

```bash
# Run all tests
pytest tests/test_ml_pipeline.py -v

# Run with coverage
pytest tests/test_ml_pipeline.py -v --cov=src

# Run specific test class
pytest tests/test_ml_pipeline.py::TestFeatureStore -v
```

## Key Components

### MLPipelineConfig

Central configuration for the entire pipeline:

```python
@dataclass
class MLPipelineConfig:
    experiment_name: str = "default_experiment"
    task_type: TaskType = TaskType.CLASSIFICATION
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    feature_store_path: str = "./feature_store"
    model_registry_path: str = "./model_registry"
    tracking_uri: str = "./mlruns"
```

### Supported Model Types

- `logistic_regression`: Logistic Regression with configurable regularization
- `random_forest`: Random Forest Classifier with 100 estimators
- `gradient_boosting`: Gradient Boosting Classifier

### Metrics Tracked

**Classification:**
- Accuracy, Precision, Recall, F1-Score, AUC-ROC

**Regression:**
- MSE, MAE, R²

## Data Generators

### MLExperimentDataGenerator

Generates synthetic ML datasets:

```python
gen = MLExperimentDataGenerator(random_seed=42)

# Classification
clf_data = gen.generate_classification_dataset(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    class_weight=0.3  # 30% positive class
)

# Regression
reg_data = gen.generate_regression_dataset(
    n_samples=1000,
    n_features=10
)
```

### FeatureDataGenerator

Generates realistic feature store data:

```python
gen = FeatureDataGenerator()
customers = gen.generate_customer_features(1000)
transactions = gen.generate_transaction_features(5000)
products = gen.generate_product_features(500)
```

## Architecture Alignment

This implementation aligns with the paper's proposed architecture:

| Paper Concept | Implementation |
|--------------|----------------|
| Feature Store | `FeatureStore` class with versioning, ingestion, retrieval |
| Model Registry | `ModelRegistry` class with stages, versioning, lifecycle |
| Experiment Tracking | `ExperimentTracker` class (MLflow-compatible) |
| Pipeline Orchestration | `MLPipeline` class integrating all components |

## License

MIT License - See research paper for full attribution.
