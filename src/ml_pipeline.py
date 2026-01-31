"""
Unified ML Pipeline Implementation
===================================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

This module implements core components of a unified ML pipeline architecture:
- Feature Store: Centralized feature management with versioning
- Model Registry: Model versioning, metadata tracking, lifecycle management  
- Experiment Tracker: MLflow-compatible experiment tracking
- ML Pipeline: End-to-end pipeline orchestration

Based on research paper: "Unified Machine Learning Pipeline Architecture for
Enterprise-Scale Model Development and Deployment"
"""

import json
import hashlib
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from enum import Enum
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)


# ============================================================================
# Configuration Classes
# ============================================================================

class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class TaskType(Enum):
    """ML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


@dataclass
class MLPipelineConfig:
    """Configuration for ML Pipeline"""
    experiment_name: str = "default_experiment"
    task_type: TaskType = TaskType.CLASSIFICATION
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Feature store settings
    feature_store_path: str = "./feature_store"
    feature_version: str = "v1.0"
    
    # Model registry settings
    model_registry_path: str = "./model_registry"
    
    # Experiment tracking
    tracking_uri: str = "./mlruns"
    
    # Training settings
    hyperparameter_tuning: bool = True
    early_stopping: bool = True
    max_iterations: int = 1000


@dataclass
class FeatureConfig:
    """Configuration for a feature"""
    name: str
    dtype: str
    description: str = ""
    source: str = "unknown"
    transformation: Optional[str] = None
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    model_id: str
    model_name: str
    version: str
    stage: ModelStage
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    feature_names: List[str]
    created_at: str
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['stage'] = self.stage.value
        return d


# ============================================================================
# Feature Store Implementation
# ============================================================================

class FeatureStore:
    """
    Centralized Feature Store for ML pipelines.
    
    Provides:
    - Feature versioning and lineage tracking
    - Point-in-time feature retrieval
    - Feature sharing across teams
    - Online/offline feature serving
    
    Example:
        >>> fs = FeatureStore("./features")
        >>> fs.register_feature("age", "int", description="Customer age")
        >>> fs.ingest_features(df, ["age", "income"])
        >>> features = fs.get_features(entity_ids=[1, 2, 3])
    """
    
    def __init__(self, store_path: str = "./feature_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.features: Dict[str, FeatureConfig] = {}
        self.feature_data: Dict[str, pd.DataFrame] = {}
        self.feature_versions: Dict[str, List[str]] = {}
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load existing feature metadata"""
        metadata_file = self.store_path / "feature_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                for name, config in data.get('features', {}).items():
                    self.features[name] = FeatureConfig(**config)
                self.feature_versions = data.get('versions', {})
    
    def _save_metadata(self) -> None:
        """Persist feature metadata"""
        metadata = {
            'features': {name: cfg.to_dict() for name, cfg in self.features.items()},
            'versions': self.feature_versions,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.store_path / "feature_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def register_feature(
        self, 
        name: str, 
        dtype: str, 
        description: str = "",
        source: str = "unknown",
        tags: Optional[List[str]] = None
    ) -> FeatureConfig:
        """
        Register a new feature in the store.
        
        Args:
            name: Feature name (must be unique)
            dtype: Data type (int, float, str, bool, datetime)
            description: Human-readable description
            source: Data source identifier
            tags: Optional tags for categorization
            
        Returns:
            FeatureConfig object
        """
        if name in self.features:
            raise ValueError(f"Feature '{name}' already exists. Use update_feature() instead.")
        
        config = FeatureConfig(
            name=name,
            dtype=dtype,
            description=description,
            source=source,
            tags=tags or []
        )
        
        self.features[name] = config
        self.feature_versions[name] = [config.version]
        self._save_metadata()
        
        return config
    
    def update_feature(
        self, 
        name: str, 
        new_version: str,
        transformation: Optional[str] = None,
        description: Optional[str] = None
    ) -> FeatureConfig:
        """Update an existing feature with a new version"""
        if name not in self.features:
            raise ValueError(f"Feature '{name}' not found")
        
        config = self.features[name]
        config.version = new_version
        config.transformation = transformation or config.transformation
        config.description = description or config.description
        
        self.feature_versions[name].append(new_version)
        self._save_metadata()
        
        return config
    
    def ingest_features(
        self, 
        df: pd.DataFrame, 
        feature_names: List[str],
        entity_column: str = "entity_id",
        timestamp_column: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Ingest feature data into the store.
        
        Args:
            df: DataFrame containing features
            feature_names: Columns to ingest as features
            entity_column: Column identifying entities
            timestamp_column: Optional timestamp for point-in-time
            
        Returns:
            Dict with ingestion statistics
        """
        stats = {'features_ingested': 0, 'rows_ingested': 0}
        
        for feature in feature_names:
            if feature not in df.columns:
                warnings.warn(f"Feature '{feature}' not in DataFrame, skipping")
                continue
            
            # Auto-register if not exists
            if feature not in self.features:
                dtype = str(df[feature].dtype)
                self.register_feature(feature, dtype, source="auto_ingest")
            
            # Store feature data
            cols = [entity_column, feature]
            if timestamp_column and timestamp_column in df.columns:
                cols.append(timestamp_column)
            
            self.feature_data[feature] = df[cols].copy()
            stats['features_ingested'] += 1
            stats['rows_ingested'] += len(df)
        
        return stats
    
    def get_features(
        self, 
        feature_names: List[str],
        entity_ids: Optional[List[Any]] = None,
        as_of_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve features from the store.
        
        Args:
            feature_names: Features to retrieve
            entity_ids: Optional filter by entity IDs
            as_of_time: Point-in-time retrieval (if timestamps available)
            
        Returns:
            DataFrame with requested features
        """
        result_frames = []
        
        for feature in feature_names:
            if feature not in self.feature_data:
                warnings.warn(f"Feature '{feature}' not found in store")
                continue
            
            df = self.feature_data[feature].copy()
            
            # Filter by entity if specified
            if entity_ids is not None and 'entity_id' in df.columns:
                df = df[df['entity_id'].isin(entity_ids)]
            
            result_frames.append(df)
        
        if not result_frames:
            return pd.DataFrame()
        
        # Merge all features
        result = result_frames[0]
        for df in result_frames[1:]:
            result = result.merge(df, on='entity_id', how='outer')
        
        return result
    
    def list_features(self) -> List[Dict[str, Any]]:
        """List all registered features"""
        return [f.to_dict() for f in self.features.values()]
    
    def get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get statistics for a feature"""
        if feature_name not in self.feature_data:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        df = self.feature_data[feature_name]
        feature_col = df[feature_name]
        
        stats = {
            'count': len(feature_col),
            'null_count': feature_col.isnull().sum(),
            'null_percentage': feature_col.isnull().mean() * 100,
            'unique_count': feature_col.nunique()
        }
        
        if np.issubdtype(feature_col.dtype, np.number):
            stats.update({
                'mean': feature_col.mean(),
                'std': feature_col.std(),
                'min': feature_col.min(),
                'max': feature_col.max(),
                'median': feature_col.median()
            })
        
        return stats


# ============================================================================
# Model Registry Implementation
# ============================================================================

class ModelRegistry:
    """
    Model Registry for versioning and lifecycle management.
    
    Provides:
    - Model versioning with semantic versioning
    - Stage transitions (dev -> staging -> production)
    - Model metadata and lineage tracking
    - Model comparison and A/B testing support
    
    Example:
        >>> registry = ModelRegistry("./models")
        >>> model_id = registry.register_model(model, "fraud_detector", metrics)
        >>> registry.transition_stage(model_id, ModelStage.PRODUCTION)
        >>> prod_model = registry.load_model("fraud_detector", stage=ModelStage.PRODUCTION)
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelMetadata] = {}
        self.model_versions: Dict[str, List[str]] = {}  # model_name -> [versions]
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry state"""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for model_id, metadata in data.get('models', {}).items():
                    metadata['stage'] = ModelStage(metadata['stage'])
                    self.models[model_id] = ModelMetadata(**metadata)
                self.model_versions = data.get('versions', {})
    
    def _save_registry(self) -> None:
        """Persist registry state"""
        registry = {
            'models': {mid: m.to_dict() for mid, m in self.models.items()},
            'versions': self.model_versions,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.registry_path / "registry.json", 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number"""
        if model_name not in self.model_versions:
            return "1.0.0"
        
        versions = self.model_versions[model_name]
        if not versions:
            return "1.0.0"
        
        latest = versions[-1]
        parts = latest.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        return '.'.join(parts)
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        feature_names: List[str],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: Trained model object (sklearn compatible)
            model_name: Name for the model
            metrics: Performance metrics
            parameters: Training parameters/hyperparameters
            feature_names: Features used in training
            description: Model description
            tags: Optional tags
            
        Returns:
            Unique model ID
        """
        version = self._generate_version(model_name)
        model_id = f"{model_name}_v{version}_{uuid.uuid4().hex[:8]}"
        
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            metrics=metrics,
            parameters=parameters,
            feature_names=feature_names,
            created_at=datetime.now().isoformat(),
            description=description,
            tags=tags or []
        )
        
        # Save model artifact
        model_path = self.registry_path / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.models[model_id] = metadata
        
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(version)
        
        self._save_registry()
        
        return model_id
    
    def load_model(
        self, 
        model_name: str, 
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Specific version (optional)
            stage: Load model at specific stage (optional)
            
        Returns:
            Tuple of (model, metadata)
        """
        # Find matching model
        candidates = [
            (mid, m) for mid, m in self.models.items()
            if m.model_name == model_name
        ]
        
        if version:
            candidates = [(mid, m) for mid, m in candidates if m.version == version]
        
        if stage:
            candidates = [(mid, m) for mid, m in candidates if m.stage == stage]
        
        if not candidates:
            raise ValueError(f"No model found matching criteria")
        
        # Get latest version
        model_id, metadata = max(candidates, key=lambda x: x[1].version)
        
        model_path = self.registry_path / f"{model_id}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata
    
    def transition_stage(
        self, 
        model_id: str, 
        target_stage: ModelStage,
        archive_previous: bool = True
    ) -> None:
        """
        Transition a model to a new stage.
        
        Args:
            model_id: Model to transition
            target_stage: Target stage
            archive_previous: Archive previous production model
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        metadata = self.models[model_id]
        model_name = metadata.model_name
        
        # Archive previous production model if promoting to production
        if target_stage == ModelStage.PRODUCTION and archive_previous:
            for mid, m in self.models.items():
                if m.model_name == model_name and m.stage == ModelStage.PRODUCTION:
                    m.stage = ModelStage.ARCHIVED
        
        metadata.stage = target_stage
        self._save_registry()
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a model"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        return self.models[model_id]
    
    def list_models(
        self, 
        model_name: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if model_name:
            models = [m for m in models if m.model_name == model_name]
        
        if stage:
            models = [m for m in models if m.stage == stage]
        
        return [m.to_dict() for m in models]
    
    def compare_models(
        self, 
        model_ids: List[str],
        metric: str = "accuracy"
    ) -> pd.DataFrame:
        """Compare multiple models by metrics"""
        data = []
        for model_id in model_ids:
            if model_id in self.models:
                m = self.models[model_id]
                data.append({
                    'model_id': model_id,
                    'version': m.version,
                    'stage': m.stage.value,
                    metric: m.metrics.get(metric, None),
                    **{k: v for k, v in m.metrics.items() if k != metric}
                })
        
        return pd.DataFrame(data)
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model from registry"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        # Remove model file
        model_path = self.registry_path / f"{model_id}.pkl"
        if model_path.exists():
            model_path.unlink()
        
        # Remove from registry
        metadata = self.models.pop(model_id)
        if metadata.version in self.model_versions.get(metadata.model_name, []):
            self.model_versions[metadata.model_name].remove(metadata.version)
        
        self._save_registry()


# ============================================================================
# Experiment Tracker Implementation
# ============================================================================

class ExperimentTracker:
    """
    MLflow-compatible experiment tracking.
    
    Tracks:
    - Parameters and hyperparameters
    - Metrics over time
    - Artifacts and models
    - Experiment comparisons
    
    Example:
        >>> tracker = ExperimentTracker("fraud_detection")
        >>> run_id = tracker.start_run("baseline_model")
        >>> tracker.log_param("learning_rate", 0.01)
        >>> tracker.log_metric("accuracy", 0.95)
        >>> tracker.end_run()
    """
    
    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_path = Path(tracking_uri) / experiment_name
        self.tracking_path.mkdir(parents=True, exist_ok=True)
        
        self.current_run: Optional[str] = None
        self.runs: Dict[str, Dict[str, Any]] = {}
        
        self._load_runs()
    
    def _load_runs(self) -> None:
        """Load existing runs"""
        runs_file = self.tracking_path / "runs.json"
        if runs_file.exists():
            with open(runs_file, 'r') as f:
                self.runs = json.load(f)
    
    def _save_runs(self) -> None:
        """Persist runs"""
        with open(self.tracking_path / "runs.json", 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def start_run(self, run_name: str = None) -> str:
        """Start a new experiment run"""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        
        self.runs[run_id] = {
            'run_id': run_id,
            'run_name': run_name or run_id,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'parameters': {},
            'metrics': {},
            'metric_history': {},
            'tags': {},
            'artifacts': []
        }
        
        self.current_run = run_id
        self._save_runs()
        
        return run_id
    
    def end_run(self, status: str = "completed") -> None:
        """End the current run"""
        if not self.current_run:
            raise ValueError("No active run")
        
        self.runs[self.current_run]['status'] = status
        self.runs[self.current_run]['end_time'] = datetime.now().isoformat()
        
        self._save_runs()
        self.current_run = None
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter"""
        if not self.current_run:
            raise ValueError("No active run")
        
        self.runs[self.current_run]['parameters'][key] = value
        self._save_runs()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters"""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric"""
        if not self.current_run:
            raise ValueError("No active run")
        
        run = self.runs[self.current_run]
        run['metrics'][key] = value
        
        # Track history
        if key not in run['metric_history']:
            run['metric_history'][key] = []
        
        run['metric_history'][key].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_runs()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run"""
        if not self.current_run:
            raise ValueError("No active run")
        
        self.runs[self.current_run]['tags'][key] = value
        self._save_runs()
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None) -> None:
        """Log an artifact file"""
        if not self.current_run:
            raise ValueError("No active run")
        
        self.runs[self.current_run]['artifacts'].append({
            'path': artifact_path,
            'name': artifact_name or Path(artifact_path).name,
            'logged_at': datetime.now().isoformat()
        })
        self._save_runs()
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run details"""
        if run_id not in self.runs:
            raise ValueError(f"Run '{run_id}' not found")
        return self.runs[run_id]
    
    def search_runs(
        self, 
        filter_params: Optional[Dict[str, Any]] = None,
        filter_metrics: Optional[Dict[str, Tuple[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search runs with filtering.
        
        Args:
            filter_params: Filter by parameter values
            filter_metrics: Filter by metric thresholds, e.g., {'accuracy': ('>', 0.9)}
        """
        results = list(self.runs.values())
        
        if filter_params:
            for key, value in filter_params.items():
                results = [
                    r for r in results 
                    if r['parameters'].get(key) == value
                ]
        
        if filter_metrics:
            for key, (op, threshold) in filter_metrics.items():
                if op == '>':
                    results = [r for r in results if r['metrics'].get(key, 0) > threshold]
                elif op == '<':
                    results = [r for r in results if r['metrics'].get(key, 0) < threshold]
                elif op == '>=':
                    results = [r for r in results if r['metrics'].get(key, 0) >= threshold]
                elif op == '<=':
                    results = [r for r in results if r['metrics'].get(key, 0) <= threshold]
                elif op == '==':
                    results = [r for r in results if r['metrics'].get(key) == threshold]
        
        return results
    
    def get_best_run(self, metric: str, maximize: bool = True) -> Dict[str, Any]:
        """Get the best run by a metric"""
        completed_runs = [r for r in self.runs.values() if r['status'] == 'completed']
        
        if not completed_runs:
            raise ValueError("No completed runs found")
        
        if maximize:
            return max(completed_runs, key=lambda r: r['metrics'].get(metric, float('-inf')))
        else:
            return min(completed_runs, key=lambda r: r['metrics'].get(metric, float('inf')))


# ============================================================================
# ML Pipeline Implementation
# ============================================================================

class MLPipeline:
    """
    End-to-end ML Pipeline orchestration.
    
    Integrates:
    - Feature Store for feature management
    - Model Registry for model versioning
    - Experiment Tracker for reproducibility
    
    Example:
        >>> pipeline = MLPipeline(config)
        >>> pipeline.prepare_data(df, "target")
        >>> results = pipeline.train_and_evaluate("random_forest")
        >>> pipeline.register_best_model("fraud_detector")
    """
    
    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self._rng = np.random.RandomState(config.random_seed)
        
        # Initialize components
        self.feature_store = FeatureStore(config.feature_store_path)
        self.model_registry = ModelRegistry(config.model_registry_path)
        self.tracker = ExperimentTracker(config.experiment_name, config.tracking_uri)
        
        # Pipeline state
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.scaler: Optional[StandardScaler] = None
        
        # Training results
        self.trained_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict[str, float]] = {}
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: Features to use (default: all except target)
            scale_features: Whether to standardize features
            
        Returns:
            Dict with preparation statistics
        """
        self.target_name = target_column
        
        # Select features
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]
        
        # Filter to numeric columns
        numeric_features = []
        for col in feature_columns:
            if np.issubdtype(df[col].dtype, np.number):
                numeric_features.append(col)
        
        self.feature_names = numeric_features
        
        X = df[numeric_features].values
        y = df[target_column].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y if self.config.task_type == TaskType.CLASSIFICATION else None
        )
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        # Ingest features into feature store
        df_with_ids = df.copy()
        df_with_ids['entity_id'] = range(len(df))
        self.feature_store.ingest_features(df_with_ids, numeric_features)
        
        return {
            'n_samples': len(df),
            'n_features': len(numeric_features),
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'features': numeric_features
        }
    
    def _get_model(self, model_type: str) -> Any:
        """Get model instance by type"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_seed
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_seed
            ),
            'logistic_regression': LogisticRegression(
                max_iter=self.config.max_iterations,
                random_state=self.config.random_seed
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        y_pred = model.predict(X)
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
            }
            
            # AUC only for binary classification
            if len(np.unique(y)) == 2:
                metrics['auc_roc'] = roc_auc_score(y, y_proba)
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return metrics
    
    def train_and_evaluate(
        self, 
        model_type: str,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train and evaluate a model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            run_name: Name for the experiment run
            
        Returns:
            Dict with training results and metrics
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first")
        
        # Start experiment run
        run_id = self.tracker.start_run(run_name or f"{model_type}_run")
        
        try:
            # Get and train model
            model = self._get_model(model_type)
            
            # Log parameters
            params = model.get_params()
            self.tracker.log_params({
                'model_type': model_type,
                'n_features': len(self.feature_names),
                'n_train_samples': len(self.X_train),
                **{k: str(v) for k, v in params.items()}
            })
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            train_metrics = self._evaluate_model(model, self.X_train, self.y_train)
            test_metrics = self._evaluate_model(model, self.X_test, self.y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=self.config.cv_folds,
                scoring='accuracy' if self.config.task_type == TaskType.CLASSIFICATION else 'r2'
            )
            
            # Log metrics
            for metric, value in test_metrics.items():
                self.tracker.log_metric(f"test_{metric}", value)
            for metric, value in train_metrics.items():
                self.tracker.log_metric(f"train_{metric}", value)
            
            self.tracker.log_metric("cv_mean", cv_scores.mean())
            self.tracker.log_metric("cv_std", cv_scores.std())
            
            # Store results
            self.trained_models[model_type] = model
            self.training_results[model_type] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'run_id': run_id
            }
            
            self.tracker.end_run()
            
            return {
                'model_type': model_type,
                'run_id': run_id,
                **self.training_results[model_type]
            }
            
        except Exception as e:
            self.tracker.end_run(status='failed')
            raise e
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        data = []
        for model_type, results in self.training_results.items():
            row = {'model_type': model_type}
            row.update({f"test_{k}": v for k, v in results['test_metrics'].items()})
            row['cv_mean'] = results['cv_mean']
            row['cv_std'] = results['cv_std']
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, Any]:
        """Get the best model by a metric"""
        if not self.training_results:
            raise ValueError("No models trained yet")
        
        best_type = max(
            self.training_results.keys(),
            key=lambda t: self.training_results[t]['test_metrics'].get(metric, 0)
        )
        
        return best_type, self.trained_models[best_type]
    
    def register_best_model(
        self, 
        model_name: str,
        metric: str = "accuracy",
        description: str = ""
    ) -> str:
        """Register the best model in the registry"""
        best_type, best_model = self.get_best_model(metric)
        results = self.training_results[best_type]
        
        model_id = self.model_registry.register_model(
            model=best_model,
            model_name=model_name,
            metrics=results['test_metrics'],
            parameters=best_model.get_params(),
            feature_names=self.feature_names,
            description=description or f"Best {best_type} model by {metric}",
            tags=[best_type, self.config.experiment_name]
        )
        
        return model_id
    
    def predict(self, df: pd.DataFrame, model_type: Optional[str] = None) -> np.ndarray:
        """Make predictions using trained model"""
        if model_type:
            model = self.trained_models.get(model_type)
        else:
            _, model = self.get_best_model()
        
        if model is None:
            raise ValueError("Model not found")
        
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return model.predict(X)


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Demonstrate the unified ML pipeline"""
    print("=" * 70)
    print("Unified ML Pipeline Demonstration")
    print("Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)")
    print("=" * 70)
    
    # Import data generator
    from data_generators import MLExperimentDataGenerator
    
    # Generate synthetic ML experiment data
    print("\n1. Generating Synthetic Dataset...")
    data_gen = MLExperimentDataGenerator(random_seed=42)
    df = data_gen.generate_classification_dataset(n_samples=2000, n_features=15)
    
    print(f"   Generated {len(df)} samples with {len(df.columns)-1} features")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Configure pipeline
    print("\n2. Configuring ML Pipeline...")
    config = MLPipelineConfig(
        experiment_name="fraud_detection_demo",
        task_type=TaskType.CLASSIFICATION,
        random_seed=42,
        test_size=0.2,
        cv_folds=5,
        feature_store_path="./data/feature_store",
        model_registry_path="./data/model_registry",
        tracking_uri="./data/mlruns"
    )
    
    pipeline = MLPipeline(config)
    
    # Prepare data
    print("\n3. Preparing Data...")
    prep_stats = pipeline.prepare_data(df, target_column='target')
    print(f"   Training samples: {prep_stats['n_train']}")
    print(f"   Test samples: {prep_stats['n_test']}")
    print(f"   Features used: {prep_stats['n_features']}")
    
    # Train multiple models
    print("\n4. Training Models...")
    models_to_train = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    for model_type in models_to_train:
        print(f"\n   Training {model_type}...")
        results = pipeline.train_and_evaluate(model_type)
        print(f"   Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"   CV Mean: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    
    # Compare models
    print("\n5. Model Comparison:")
    comparison = pipeline.compare_models()
    print(comparison.to_string(index=False))
    
    # Register best model
    print("\n6. Registering Best Model...")
    model_id = pipeline.register_best_model(
        "fraud_detector",
        metric="accuracy",
        description="Fraud detection model trained on synthetic data"
    )
    print(f"   Registered model: {model_id}")
    
    # Load and verify model
    print("\n7. Verifying Model Registry...")
    loaded_model, metadata = pipeline.model_registry.load_model("fraud_detector")
    print(f"   Model version: {metadata.version}")
    print(f"   Model stage: {metadata.stage.value}")
    print(f"   Accuracy: {metadata.metrics['accuracy']:.4f}")
    
    # Promote to production
    print("\n8. Promoting to Production...")
    pipeline.model_registry.transition_stage(model_id, ModelStage.PRODUCTION)
    print(f"   Model {model_id} promoted to production")
    
    # Feature store summary
    print("\n9. Feature Store Summary:")
    features = pipeline.feature_store.list_features()
    print(f"   Registered features: {len(features)}")
    
    # Save experiment summary
    print("\n10. Saving Results...")
    summary = {
        'experiment': config.experiment_name,
        'models_trained': models_to_train,
        'best_model': pipeline.get_best_model()[0],
        'best_accuracy': pipeline.get_best_model()[1].score(pipeline.X_test, pipeline.y_test),
        'model_id': model_id,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('./data/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary saved to ./data/experiment_summary.json")
    
    print("\n" + "=" * 70)
    print("Pipeline demonstration complete!")
    print("=" * 70)
    
    return pipeline, summary


if __name__ == "__main__":
    main()
