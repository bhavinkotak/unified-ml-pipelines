"""
Tests for Unified ML Pipeline
==============================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

Comprehensive test suite for ML pipeline components.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_pipeline import (
    MLPipelineConfig, FeatureConfig, ModelMetadata, ModelStage, TaskType,
    FeatureStore, ModelRegistry, ExperimentTracker, MLPipeline
)
from data_generators import MLExperimentDataGenerator, FeatureDataGenerator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing"""
    gen = MLExperimentDataGenerator(random_seed=42)
    return gen.generate_classification_dataset(n_samples=500, n_features=10)


@pytest.fixture
def feature_store(temp_dir):
    """Create FeatureStore instance"""
    return FeatureStore(store_path=f"{temp_dir}/features")


@pytest.fixture
def model_registry(temp_dir):
    """Create ModelRegistry instance"""
    return ModelRegistry(registry_path=f"{temp_dir}/models")


@pytest.fixture
def experiment_tracker(temp_dir):
    """Create ExperimentTracker instance"""
    return ExperimentTracker("test_experiment", tracking_uri=f"{temp_dir}/mlruns")


@pytest.fixture
def ml_pipeline(temp_dir, sample_dataframe):
    """Create configured MLPipeline"""
    config = MLPipelineConfig(
        experiment_name="test_pipeline",
        task_type=TaskType.CLASSIFICATION,
        random_seed=42,
        test_size=0.2,
        feature_store_path=f"{temp_dir}/features",
        model_registry_path=f"{temp_dir}/models",
        tracking_uri=f"{temp_dir}/mlruns"
    )
    pipeline = MLPipeline(config)
    pipeline.prepare_data(sample_dataframe, target_column='target')
    return pipeline


# ============================================================================
# Configuration Tests
# ============================================================================

class TestMLPipelineConfig:
    """Test configuration classes"""
    
    def test_default_config(self):
        config = MLPipelineConfig()
        assert config.experiment_name == "default_experiment"
        assert config.task_type == TaskType.CLASSIFICATION
        assert config.random_seed == 42
        assert config.test_size == 0.2
    
    def test_custom_config(self):
        config = MLPipelineConfig(
            experiment_name="custom",
            task_type=TaskType.REGRESSION,
            random_seed=123
        )
        assert config.experiment_name == "custom"
        assert config.task_type == TaskType.REGRESSION
        assert config.random_seed == 123
    
    def test_feature_config(self):
        config = FeatureConfig(
            name="age",
            dtype="int",
            description="Customer age",
            source="crm"
        )
        assert config.name == "age"
        assert config.dtype == "int"
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['name'] == "age"


# ============================================================================
# Feature Store Tests
# ============================================================================

class TestFeatureStore:
    """Test FeatureStore functionality"""
    
    def test_register_feature(self, feature_store):
        config = feature_store.register_feature(
            "age", "int", 
            description="Customer age",
            source="crm"
        )
        
        assert config.name == "age"
        assert "age" in feature_store.features
        assert "age" in feature_store.feature_versions
    
    def test_register_duplicate_raises(self, feature_store):
        feature_store.register_feature("age", "int")
        
        with pytest.raises(ValueError, match="already exists"):
            feature_store.register_feature("age", "int")
    
    def test_update_feature(self, feature_store):
        feature_store.register_feature("age", "int")
        config = feature_store.update_feature("age", "1.1", description="Updated")
        
        assert config.version == "1.1"
        assert config.description == "Updated"
        assert len(feature_store.feature_versions["age"]) == 2
    
    def test_ingest_features(self, feature_store):
        df = pd.DataFrame({
            'entity_id': [1, 2, 3],
            'age': [25, 35, 45],
            'income': [50000, 75000, 100000]
        })
        
        stats = feature_store.ingest_features(df, ['age', 'income'])
        
        assert stats['features_ingested'] == 2
        assert 'age' in feature_store.feature_data
        assert 'income' in feature_store.feature_data
    
    def test_get_features(self, feature_store):
        df = pd.DataFrame({
            'entity_id': [1, 2, 3],
            'age': [25, 35, 45],
            'income': [50000, 75000, 100000]
        })
        
        feature_store.ingest_features(df, ['age', 'income'])
        result = feature_store.get_features(['age', 'income'], entity_ids=[1, 2])
        
        assert len(result) == 2
        assert 'age' in result.columns
        assert 'income' in result.columns
    
    def test_list_features(self, feature_store):
        feature_store.register_feature("age", "int")
        feature_store.register_feature("income", "float")
        
        features = feature_store.list_features()
        
        assert len(features) == 2
        assert any(f['name'] == 'age' for f in features)
    
    def test_feature_stats(self, feature_store):
        df = pd.DataFrame({
            'entity_id': range(100),
            'value': np.random.randn(100) * 10 + 50
        })
        
        feature_store.ingest_features(df, ['value'])
        stats = feature_store.get_feature_stats('value')
        
        assert 'count' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['count'] == 100


# ============================================================================
# Model Registry Tests
# ============================================================================

class TestModelRegistry:
    """Test ModelRegistry functionality"""
    
    def test_register_model(self, model_registry):
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])  # Dummy fit
        
        model_id = model_registry.register_model(
            model=model,
            model_name="test_model",
            metrics={'accuracy': 0.95},
            parameters={'C': 1.0},
            feature_names=['feature_0']
        )
        
        assert model_id is not None
        assert model_id in model_registry.models
        assert model_registry.models[model_id].stage == ModelStage.DEVELOPMENT
    
    def test_load_model(self, model_registry):
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])
        
        model_id = model_registry.register_model(
            model=model,
            model_name="test_model",
            metrics={'accuracy': 0.95},
            parameters={'C': 1.0},
            feature_names=['feature_0']
        )
        
        loaded_model, metadata = model_registry.load_model("test_model")
        
        assert loaded_model is not None
        assert metadata.model_name == "test_model"
        assert metadata.metrics['accuracy'] == 0.95
    
    def test_transition_stage(self, model_registry):
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])
        
        model_id = model_registry.register_model(
            model=model,
            model_name="test_model",
            metrics={'accuracy': 0.95},
            parameters={},
            feature_names=[]
        )
        
        model_registry.transition_stage(model_id, ModelStage.STAGING)
        assert model_registry.models[model_id].stage == ModelStage.STAGING
        
        model_registry.transition_stage(model_id, ModelStage.PRODUCTION)
        assert model_registry.models[model_id].stage == ModelStage.PRODUCTION
    
    def test_version_increment(self, model_registry):
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])
        
        id1 = model_registry.register_model(
            model=model, model_name="versioned",
            metrics={}, parameters={}, feature_names=[]
        )
        
        id2 = model_registry.register_model(
            model=model, model_name="versioned",
            metrics={}, parameters={}, feature_names=[]
        )
        
        assert model_registry.models[id1].version == "1.0.0"
        assert model_registry.models[id2].version == "1.0.1"
    
    def test_list_models(self, model_registry):
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])
        
        model_registry.register_model(
            model=model, model_name="model_a",
            metrics={}, parameters={}, feature_names=[]
        )
        model_registry.register_model(
            model=model, model_name="model_b",
            metrics={}, parameters={}, feature_names=[]
        )
        
        all_models = model_registry.list_models()
        assert len(all_models) == 2
        
        filtered = model_registry.list_models(model_name="model_a")
        assert len(filtered) == 1
    
    def test_delete_model(self, model_registry):
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])
        
        model_id = model_registry.register_model(
            model=model, model_name="to_delete",
            metrics={}, parameters={}, feature_names=[]
        )
        
        model_registry.delete_model(model_id)
        assert model_id not in model_registry.models


# ============================================================================
# Experiment Tracker Tests
# ============================================================================

class TestExperimentTracker:
    """Test ExperimentTracker functionality"""
    
    def test_start_end_run(self, experiment_tracker):
        run_id = experiment_tracker.start_run("test_run")
        
        assert run_id is not None
        assert experiment_tracker.current_run == run_id
        assert experiment_tracker.runs[run_id]['status'] == 'running'
        
        experiment_tracker.end_run()
        
        assert experiment_tracker.current_run is None
        assert experiment_tracker.runs[run_id]['status'] == 'completed'
    
    def test_log_param(self, experiment_tracker):
        experiment_tracker.start_run()
        experiment_tracker.log_param("learning_rate", 0.01)
        
        run = experiment_tracker.runs[experiment_tracker.current_run]
        assert run['parameters']['learning_rate'] == 0.01
        
        experiment_tracker.end_run()
    
    def test_log_params(self, experiment_tracker):
        experiment_tracker.start_run()
        experiment_tracker.log_params({
            'learning_rate': 0.01,
            'batch_size': 32
        })
        
        run = experiment_tracker.runs[experiment_tracker.current_run]
        assert run['parameters']['learning_rate'] == 0.01
        assert run['parameters']['batch_size'] == 32
        
        experiment_tracker.end_run()
    
    def test_log_metric(self, experiment_tracker):
        experiment_tracker.start_run()
        experiment_tracker.log_metric("accuracy", 0.95)
        
        run = experiment_tracker.runs[experiment_tracker.current_run]
        assert run['metrics']['accuracy'] == 0.95
        
        experiment_tracker.end_run()
    
    def test_metric_history(self, experiment_tracker):
        experiment_tracker.start_run()
        experiment_tracker.log_metric("loss", 1.0, step=0)
        experiment_tracker.log_metric("loss", 0.5, step=1)
        experiment_tracker.log_metric("loss", 0.3, step=2)
        
        run = experiment_tracker.runs[experiment_tracker.current_run]
        assert len(run['metric_history']['loss']) == 3
        assert run['metrics']['loss'] == 0.3  # Latest value
        
        experiment_tracker.end_run()
    
    def test_set_tag(self, experiment_tracker):
        experiment_tracker.start_run()
        experiment_tracker.set_tag("model_type", "random_forest")
        
        run = experiment_tracker.runs[experiment_tracker.current_run]
        assert run['tags']['model_type'] == "random_forest"
        
        experiment_tracker.end_run()
    
    def test_search_runs(self, experiment_tracker):
        # Create multiple runs
        experiment_tracker.start_run("run_1")
        experiment_tracker.log_param("model", "rf")
        experiment_tracker.log_metric("accuracy", 0.85)
        experiment_tracker.end_run()
        
        experiment_tracker.start_run("run_2")
        experiment_tracker.log_param("model", "lr")
        experiment_tracker.log_metric("accuracy", 0.90)
        experiment_tracker.end_run()
        
        experiment_tracker.start_run("run_3")
        experiment_tracker.log_param("model", "rf")
        experiment_tracker.log_metric("accuracy", 0.95)
        experiment_tracker.end_run()
        
        # Search by param
        rf_runs = experiment_tracker.search_runs(filter_params={'model': 'rf'})
        assert len(rf_runs) == 2
        
        # Search by metric
        high_acc = experiment_tracker.search_runs(
            filter_metrics={'accuracy': ('>', 0.88)}
        )
        assert len(high_acc) == 2
    
    def test_get_best_run(self, experiment_tracker):
        experiment_tracker.start_run("run_1")
        experiment_tracker.log_metric("accuracy", 0.85)
        experiment_tracker.end_run()
        
        experiment_tracker.start_run("run_2")
        experiment_tracker.log_metric("accuracy", 0.95)
        experiment_tracker.end_run()
        
        best = experiment_tracker.get_best_run("accuracy", maximize=True)
        assert best['metrics']['accuracy'] == 0.95


# ============================================================================
# ML Pipeline Integration Tests
# ============================================================================

class TestMLPipeline:
    """Integration tests for MLPipeline"""
    
    def test_prepare_data(self, ml_pipeline):
        assert ml_pipeline.X_train is not None
        assert ml_pipeline.X_test is not None
        assert ml_pipeline.y_train is not None
        assert ml_pipeline.y_test is not None
        assert len(ml_pipeline.feature_names) > 0
    
    def test_train_logistic_regression(self, ml_pipeline):
        results = ml_pipeline.train_and_evaluate('logistic_regression')
        
        assert 'accuracy' in results['test_metrics']
        assert results['test_metrics']['accuracy'] > 0.5
        assert 'logistic_regression' in ml_pipeline.trained_models
    
    def test_train_random_forest(self, ml_pipeline):
        results = ml_pipeline.train_and_evaluate('random_forest')
        
        assert 'accuracy' in results['test_metrics']
        assert results['test_metrics']['accuracy'] > 0.5
        assert 'random_forest' in ml_pipeline.trained_models
    
    def test_train_gradient_boosting(self, ml_pipeline):
        results = ml_pipeline.train_and_evaluate('gradient_boosting')
        
        assert 'accuracy' in results['test_metrics']
        assert results['test_metrics']['accuracy'] > 0.5
        assert 'gradient_boosting' in ml_pipeline.trained_models
    
    def test_compare_models(self, ml_pipeline):
        ml_pipeline.train_and_evaluate('logistic_regression')
        ml_pipeline.train_and_evaluate('random_forest')
        
        comparison = ml_pipeline.compare_models()
        
        assert len(comparison) == 2
        assert 'model_type' in comparison.columns
        assert 'test_accuracy' in comparison.columns
    
    def test_get_best_model(self, ml_pipeline):
        ml_pipeline.train_and_evaluate('logistic_regression')
        ml_pipeline.train_and_evaluate('random_forest')
        
        best_type, best_model = ml_pipeline.get_best_model('accuracy')
        
        assert best_type in ['logistic_regression', 'random_forest']
        assert best_model is not None
    
    def test_register_best_model(self, ml_pipeline):
        ml_pipeline.train_and_evaluate('random_forest')
        
        model_id = ml_pipeline.register_best_model(
            "test_model",
            metric="accuracy"
        )
        
        assert model_id is not None
        
        # Verify in registry
        models = ml_pipeline.model_registry.list_models()
        assert len(models) == 1
        assert models[0]['model_name'] == "test_model"
    
    def test_predict(self, ml_pipeline, sample_dataframe):
        ml_pipeline.train_and_evaluate('logistic_regression')
        
        test_df = sample_dataframe.head(10).drop('target', axis=1)
        predictions = ml_pipeline.predict(test_df)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


# ============================================================================
# Data Generator Tests
# ============================================================================

class TestDataGenerators:
    """Test data generation utilities"""
    
    def test_classification_dataset(self):
        gen = MLExperimentDataGenerator(random_seed=42)
        df = gen.generate_classification_dataset(
            n_samples=100, 
            n_features=5
        )
        
        assert len(df) == 100
        assert 'target' in df.columns
        assert df['target'].nunique() == 2
    
    def test_regression_dataset(self):
        gen = MLExperimentDataGenerator(random_seed=42)
        df = gen.generate_regression_dataset(
            n_samples=100,
            n_features=5
        )
        
        assert len(df) == 100
        assert 'target' in df.columns
        assert df['target'].dtype == np.float64
    
    def test_reproducibility(self):
        gen1 = MLExperimentDataGenerator(random_seed=42)
        gen2 = MLExperimentDataGenerator(random_seed=42)
        
        df1 = gen1.generate_classification_dataset(n_samples=50)
        df2 = gen2.generate_classification_dataset(n_samples=50)
        
        assert df1['target'].sum() == df2['target'].sum()
    
    def test_feature_data_generator(self):
        gen = FeatureDataGenerator(random_seed=42)
        
        customers = gen.generate_customer_features(100)
        assert len(customers) == 100
        assert 'customer_age' in customers.columns
        
        transactions = gen.generate_transaction_features(500)
        assert len(transactions) == 500
        assert 'transaction_amount' in transactions.columns
        
        products = gen.generate_product_features(50)
        assert len(products) == 50
        assert 'category' in products.columns


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_feature_retrieval(self, feature_store):
        result = feature_store.get_features(['nonexistent'])
        assert len(result) == 0
    
    def test_model_not_found(self, model_registry):
        with pytest.raises(ValueError, match="No model found"):
            model_registry.load_model("nonexistent")
    
    def test_no_active_run(self, experiment_tracker):
        with pytest.raises(ValueError, match="No active run"):
            experiment_tracker.log_param("key", "value")
    
    def test_train_without_prepare(self, temp_dir):
        config = MLPipelineConfig(
            feature_store_path=f"{temp_dir}/features",
            model_registry_path=f"{temp_dir}/models",
            tracking_uri=f"{temp_dir}/mlruns"
        )
        pipeline = MLPipeline(config)
        
        with pytest.raises(ValueError, match="Call prepare_data"):
            pipeline.train_and_evaluate('logistic_regression')
    
    def test_unknown_model_type(self, ml_pipeline):
        with pytest.raises(ValueError, match="Unknown model type"):
            ml_pipeline.train_and_evaluate('unknown_model')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
