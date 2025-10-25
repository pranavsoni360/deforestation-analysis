#!/usr/bin/env python3
"""
Machine Learning Models for Deforestation Risk Prediction

This module contains ML models for predicting deforestation risk based on 
integrated environmental, satellite, biodiversity, and climate data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
# Optional gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. XGBoost model will be skipped.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. LightGBM model will be skipped.")

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be skipped.")

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available. Visualization features will be skipped.")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeforestationRiskPredictor:
    """Machine learning models for predicting deforestation risk."""
    
    def __init__(self, data_dir: str = "./processed_data", models_dir: str = "./trained_models"):
        """Initialize the ML predictor."""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Model parameters
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'random_state': 42
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            }
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model performance tracking
        self.model_performance = {}
    
    def load_and_prepare_data(self, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the integrated dataset for ML training."""
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load data
        df = pd.read_csv(dataset_path)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Define feature columns (excluding target and metadata)
        feature_columns = [
            'latitude', 'longitude',
            'mean_ndvi', 'ndvi_trend', 'mean_lst',
            'species_richness', 'threatened_species_count', 'endemic_species_count',
            'mean_carbon_stock', 'mean_canopy_height', 'mean_canopy_cover',
            'mean_temperature', 'annual_precipitation',
            'soil_ph', 'soil_organic_carbon', 'soil_fertility_index',
            'water_stress_index', 'soil_vulnerability_index', 'climate_stability_index'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        logger.info(f"Using {len(available_features)} features for training")
        self.feature_names = available_features
        
        # Prepare features
        X = df[available_features].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Prepare target variable
        if 'deforestation_risk_category' in df.columns:
            y = df['deforestation_risk_category'].copy()
        elif 'environmental_risk_score' in df.columns:
            # Convert continuous risk score to categories
            y = pd.cut(
                df['environmental_risk_score'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        else:
            raise ValueError("No suitable target variable found in dataset")
        
        # Remove rows with missing targets
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature matrix."""
        
        logger.info("Handling missing values")
        
        # Fill numeric columns with median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X[col].isnull().any():
                median_value = X[col].median()
                X[col].fillna(median_value, inplace=True)
        
        # Fill categorical columns with mode
        categorical_columns = X.select_dtypes(include=[object]).columns
        for col in categorical_columns:
            if X[col].isnull().any():
                mode_value = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown'
                X[col].fillna(mode_value, inplace=True)
        
        return X
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for ML training."""
        
        logger.info("Engineering features")
        
        X_engineered = X.copy()
        
        # Geographic features
        if 'latitude' in X.columns and 'longitude' in X.columns:
            # Distance from equator
            X_engineered['distance_from_equator'] = abs(X['latitude'])
            
            # Create regional clusters (simplified)
            X_engineered['region_cluster'] = self._create_region_clusters(X['latitude'], X['longitude'])
        
        # Vegetation health composite
        vegetation_features = ['mean_ndvi', 'ndvi_trend']
        available_veg_features = [f for f in vegetation_features if f in X.columns]
        if len(available_veg_features) >= 2:
            X_engineered['vegetation_health_index'] = X[available_veg_features].mean(axis=1)
        
        # Biodiversity pressure index
        biodiversity_features = ['species_richness', 'threatened_species_count']
        available_bio_features = [f for f in biodiversity_features if f in X.columns]
        if len(available_bio_features) >= 2:
            # Normalize and combine (higher diversity = lower pressure)
            species_norm = X['species_richness'] / (X['species_richness'].max() + 1e-6)
            threat_norm = X['threatened_species_count'] / (X['threatened_species_count'].max() + 1e-6)
            X_engineered['biodiversity_pressure'] = (1 - species_norm) + threat_norm
        
        # Climate stress composite
        climate_features = ['water_stress_index', 'soil_vulnerability_index']
        available_climate_features = [f for f in climate_features if f in X.columns]
        if len(available_climate_features) >= 2:
            X_engineered['climate_stress_composite'] = X[available_climate_features].mean(axis=1)
        
        # Carbon vulnerability index
        if 'mean_carbon_stock' in X.columns and 'mean_canopy_height' in X.columns:
            # Normalize carbon and height (lower values = higher vulnerability)
            carbon_norm = X['mean_carbon_stock'] / (X['mean_carbon_stock'].max() + 1e-6)
            height_norm = X['mean_canopy_height'] / (X['mean_canopy_height'].max() + 1e-6)
            X_engineered['carbon_vulnerability'] = 1 - ((carbon_norm + height_norm) / 2)
        
        # Interaction features
        if 'mean_temperature' in X.columns and 'annual_precipitation' in X.columns:
            # Climate interaction
            X_engineered['temp_precip_ratio'] = X['mean_temperature'] / (X['annual_precipitation'] + 1e-6)
        
        if 'soil_ph' in X.columns and 'soil_organic_carbon' in X.columns:
            # Soil health interaction
            X_engineered['soil_health_interaction'] = X['soil_ph'] * X['soil_organic_carbon']
        
        logger.info(f"Feature engineering complete. Features: {X_engineered.shape[1]}")
        
        return X_engineered
    
    def _create_region_clusters(self, lat: pd.Series, lon: pd.Series) -> pd.Series:
        """Create simple regional clusters based on coordinates."""
        
        # Simple clustering based on coordinate ranges
        clusters = []
        
        for i in range(len(lat)):
            if lat.iloc[i] > -10:  # Northern regions
                if lon.iloc[i] > -60:
                    clusters.append(0)  # Northeast
                else:
                    clusters.append(1)  # Northwest
            else:  # Southern regions
                if lon.iloc[i] > -55:
                    clusters.append(2)  # Southeast
                else:
                    clusters.append(3)  # Southwest
        
        return pd.Series(clusters, index=lat.index)
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train all configured ML models."""
        
        logger.info("Starting model training pipeline")
        
        # Prepare features
        X_engineered = self.prepare_features(X)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y_encoded, test_size=test_size, 
            stratify=y_encoded, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_scaled_standard = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled_standard = self.scalers['standard'].transform(X_test)
        
        X_train_scaled_robust = self.scalers['robust'].fit_transform(X_train)
        X_test_scaled_robust = self.scalers['robust'].transform(X_test)
        
        # Train models
        results = {}
        
        # Random Forest
        logger.info("Training Random Forest")
        rf_model = self._train_random_forest(X_train, y_train, X_test, y_test)
        results['random_forest'] = rf_model
        
        # Gradient Boosting
        logger.info("Training Gradient Boosting")
        gb_model = self._train_gradient_boosting(X_train, y_train, X_test, y_test)
        results['gradient_boosting'] = gb_model
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost")
            xgb_model = self._train_xgboost(X_train, y_train, X_test, y_test)
            results['xgboost'] = xgb_model
        else:
            logger.warning("Skipping XGBoost (package not installed)")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM")
            lgb_model = self._train_lightgbm(X_train, y_train, X_test, y_test)
            results['lightgbm'] = lgb_model
        else:
            logger.warning("Skipping LightGBM (package not installed)")
        
        # Logistic Regression (with scaling)
        logger.info("Training Logistic Regression")
        lr_model = self._train_logistic_regression(
            X_train_scaled_standard, y_train, X_test_scaled_standard, y_test
        )
        results['logistic_regression'] = lr_model
        
        # SVM (with scaling)
        logger.info("Training SVM")
        svm_model = self._train_svm(
            X_train_scaled_robust, y_train, X_test_scaled_robust, y_test
        )
        results['svm'] = svm_model
        
        # Neural Network (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            logger.info("Training Neural Network")
            nn_model = self._train_neural_network(
                X_train_scaled_standard, y_train, X_test_scaled_standard, y_test
            )
            results['neural_network'] = nn_model
        
        # Ensemble model
        logger.info("Creating Ensemble Model")
        ensemble_model = self._create_ensemble_model(X_train, y_train, X_test, y_test)
        results['ensemble'] = ensemble_model
        
        # Store feature names
        self.feature_names = list(X_engineered.columns)
        
        logger.info("Model training completed")
        
        return results
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest model."""
        
        model = RandomForestClassifier(**self.model_configs['random_forest'])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Store model and calculate metrics
        self.models['random_forest'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Feature importance
        self.feature_importance['random_forest'] = dict(
            zip(self.feature_names, model.feature_importances_)
        )
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': self.feature_importance['random_forest']
        }
    
    def _train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting model."""
        
        model = GradientBoostingClassifier(**self.model_configs['gradient_boosting'])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Store model and calculate metrics
        self.models['gradient_boosting'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Feature importance
        self.feature_importance['gradient_boosting'] = dict(
            zip(self.feature_names, model.feature_importances_)
        )
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': self.feature_importance['gradient_boosting']
        }
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost model."""
        
        model = xgb.XGBClassifier(**self.model_configs['xgboost'])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Store model and calculate metrics
        self.models['xgboost'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Feature importance
        importance_dict = model.get_booster().get_score(importance_type='weight')
        # Map feature indices to names
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_key = f'f{i}'
            feature_importance[feature_name] = importance_dict.get(feature_key, 0)
        
        self.feature_importance['xgboost'] = feature_importance
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': feature_importance
        }
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train LightGBM model."""
        
        model = lgb.LGBMClassifier(**self.model_configs['lightgbm'])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Store model and calculate metrics
        self.models['lightgbm'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Feature importance
        self.feature_importance['lightgbm'] = dict(
            zip(self.feature_names, model.feature_importances_)
        )
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': self.feature_importance['lightgbm']
        }
    
    def _train_logistic_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Logistic Regression model."""
        
        model = LogisticRegression(**self.model_configs['logistic_regression'])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Store model and calculate metrics
        self.models['logistic_regression'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Feature importance (absolute coefficients)
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
            self.feature_importance['logistic_regression'] = dict(
                zip(self.feature_names, importance)
            )
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': self.feature_importance.get('logistic_regression', {})
        }
    
    def _train_svm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train SVM model."""
        
        model = SVC(**self.model_configs['svm'])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Store model and calculate metrics
        self.models['svm'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Feature importance using permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42
        )
        
        self.feature_importance['svm'] = dict(
            zip(self.feature_names, perm_importance.importances_mean)
        )
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': self.feature_importance['svm']
        }
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Neural Network model."""
        
        # Convert target to categorical
        n_classes = len(np.unique(y_train))
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes=n_classes)
        
        # Build model
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train_cat,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test_cat),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predictions
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        # Store model and calculate metrics
        self.models['neural_network'] = model
        performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        performance['training_history'] = history.history
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': {}  # Neural networks don't have simple feature importance
        }
    
    def _create_ensemble_model(self, X_train, y_train, X_test, y_test) -> Dict:
        """Create ensemble model combining best performing models."""
        
        # Use top 3 performing models for ensemble
        # Include models that are available
        candidate_models = ['random_forest', 'gradient_boosting']
        if XGBOOST_AVAILABLE:
            candidate_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            candidate_models.append('lightgbm')
        best_models = candidate_models
        
        ensemble_predictions = []
        
        for model_name in best_models:
            if model_name in self.models:
                model = self.models[model_name]
                y_prob = model.predict_proba(X_test)
                ensemble_predictions.append(y_prob)
        
        if ensemble_predictions:
            # Average predictions
            ensemble_prob = np.mean(ensemble_predictions, axis=0)
            ensemble_pred = np.argmax(ensemble_prob, axis=1)
            
            # Calculate metrics
            performance = self._calculate_performance_metrics(y_test, ensemble_pred, ensemble_prob)
            
            return {
                'model': 'ensemble',
                'performance': performance,
                'component_models': best_models,
                'feature_importance': {}
            }
        
        return {}
    
    def _calculate_performance_metrics(self, y_true, y_pred, y_prob) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Multi-class ROC AUC
        try:
            if y_prob.shape[1] > 2:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:  # Binary
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.label_encoder.classes_, 
            output_dict=True
        )
        
        return metrics
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'random_forest') -> Dict:
        """Perform hyperparameter tuning for specified model."""
        
        logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        X_engineered = self.prepare_features(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"Hyperparameter tuning not configured for {model_name}")
        
        # Initialize model
        if model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(random_state=42)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_engineered, y_encoded)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters for {model_name}: {results['best_params']}")
        logger.info(f"Best cross-validation score: {results['best_score']:.4f}")
        
        return results
    
    def predict_deforestation_risk(self, X_new: pd.DataFrame, model_name: str = 'ensemble') -> Dict:
        """Predict deforestation risk for new data."""
        
        if model_name not in self.models and model_name != 'ensemble':
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Prepare features
        X_prepared = self.prepare_features(X_new)
        
        # Ensure same feature order as training
        X_prepared = X_prepared.reindex(columns=self.feature_names, fill_value=0)
        
        if model_name == 'ensemble':
            # Ensemble prediction
            predictions = []
            probabilities = []
            
            ensemble_models = ['random_forest', 'xgboost', 'gradient_boosting']
            for m_name in ensemble_models:
                if m_name in self.models:
                    model = self.models[m_name]
                    
                    # Scale if needed
                    if m_name in ['logistic_regression', 'svm', 'neural_network']:
                        X_scaled = self.scalers['standard'].transform(X_prepared)
                        prob = model.predict_proba(X_scaled)
                    else:
                        prob = model.predict_proba(X_prepared)
                    
                    probabilities.append(prob)
            
            if probabilities:
                # Average probabilities
                ensemble_prob = np.mean(probabilities, axis=0)
                ensemble_pred = np.argmax(ensemble_prob, axis=1)
                
                # Convert back to original labels
                predicted_labels = self.label_encoder.inverse_transform(ensemble_pred)
                
                return {
                    'predictions': predicted_labels,
                    'probabilities': ensemble_prob,
                    'risk_scores': ensemble_prob.max(axis=1),
                    'model_used': 'ensemble'
                }
        else:
            # Single model prediction
            model = self.models[model_name]
            
            # Scale if needed
            if model_name in ['logistic_regression', 'svm']:
                X_scaled = self.scalers['standard'].transform(X_prepared)
                pred = model.predict(X_scaled)
                prob = model.predict_proba(X_scaled)
            else:
                pred = model.predict(X_prepared)
                prob = model.predict_proba(X_prepared)
            
            # Convert back to original labels
            predicted_labels = self.label_encoder.inverse_transform(pred)
            
            return {
                'predictions': predicted_labels,
                'probabilities': prob,
                'risk_scores': prob.max(axis=1),
                'model_used': model_name
            }
    
    def save_models(self) -> Dict:
        """Save trained models to disk."""
        
        logger.info("Saving trained models")
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save sklearn models
        for model_name, model in self.models.items():
            if model_name != 'neural_network':
                filename = f"{model_name}_model_{timestamp}.pkl"
                filepath = self.models_dir / filename
                
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                
                saved_files[model_name] = str(filepath)
        
        # Save neural network model separately
        if 'neural_network' in self.models and TENSORFLOW_AVAILABLE:
            nn_filename = f"neural_network_model_{timestamp}.h5"
            nn_filepath = self.models_dir / nn_filename
            self.models['neural_network'].save(str(nn_filepath))
            saved_files['neural_network'] = str(nn_filepath)
        
        # Save scalers
        scalers_filename = f"scalers_{timestamp}.pkl"
        scalers_filepath = self.models_dir / scalers_filename
        
        with open(scalers_filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        saved_files['scalers'] = str(scalers_filepath)
        
        # Save label encoder
        encoder_filename = f"label_encoder_{timestamp}.pkl"
        encoder_filepath = self.models_dir / encoder_filename
        
        with open(encoder_filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        saved_files['label_encoder'] = str(encoder_filepath)
        
        # Save feature names
        features_filename = f"feature_names_{timestamp}.json"
        features_filepath = self.models_dir / features_filename
        
        with open(features_filepath, 'w') as f:
            json.dump(self.feature_names, f)
        
        saved_files['feature_names'] = str(features_filepath)
        
        # Save model performance
        performance_filename = f"model_performance_{timestamp}.json"
        performance_filepath = self.models_dir / performance_filename
        
        with open(performance_filepath, 'w') as f:
            json.dump(self.model_performance, f, default=str)
        
        saved_files['performance'] = str(performance_filepath)
        
        logger.info(f"Models saved to {self.models_dir}")
        
        return saved_files
    
    def load_models(self, model_files: Dict):
        """Load trained models from disk."""
        
        logger.info("Loading trained models")
        
        # Load sklearn models
        for model_name, filepath in model_files.items():
            if model_name in ['scalers', 'label_encoder', 'feature_names', 'performance']:
                continue
            
            if model_name == 'neural_network' and TENSORFLOW_AVAILABLE:
                self.models[model_name] = keras.models.load_model(filepath)
            else:
                with open(filepath, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        
        # Load scalers
        if 'scalers' in model_files:
            with open(model_files['scalers'], 'rb') as f:
                self.scalers = pickle.load(f)
        
        # Load label encoder
        if 'label_encoder' in model_files:
            with open(model_files['label_encoder'], 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        # Load feature names
        if 'feature_names' in model_files:
            with open(model_files['feature_names'], 'r') as f:
                self.feature_names = json.load(f)
        
        logger.info("Models loaded successfully")
    
    def generate_model_report(self) -> Dict:
        """Generate comprehensive model performance report."""
        
        logger.info("Generating model performance report")
        
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'features_used': self.feature_names,
            'model_performance': {},
            'feature_importance_summary': {},
            'best_model': None
        }
        
        # Summarize performance for each model
        best_f1_score = 0
        best_model_name = None
        
        for model_name in self.model_performance:
            performance = self.model_performance[model_name]
            
            report['model_performance'][model_name] = {
                'accuracy': performance.get('accuracy', 0),
                'precision': performance.get('precision', 0),
                'recall': performance.get('recall', 0),
                'f1_score': performance.get('f1_score', 0),
                'roc_auc': performance.get('roc_auc', 0)
            }
            
            # Track best model
            current_f1 = performance.get('f1_score', 0)
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_model_name = model_name
        
        report['best_model'] = best_model_name
        
        # Feature importance summary
        for model_name, importance_dict in self.feature_importance.items():
            if importance_dict:
                # Top 10 features
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                report['feature_importance_summary'][model_name] = sorted_features[:10]
        
        return report


def main():
    """Main function to demonstrate ML model training."""
    
    # Initialize ML predictor
    predictor = DeforestationRiskPredictor()
    
    # Look for latest ML dataset
    data_files = list(predictor.data_dir.glob("ml_dataset_*.csv"))
    
    if not data_files:
        logger.error("No ML dataset found. Please run the data integration pipeline first.")
        return
    
    # Use the most recent dataset
    latest_dataset = max(data_files, key=os.path.getctime)
    logger.info(f"Using dataset: {latest_dataset}")
    
    try:
        # Load and prepare data
        X, y = predictor.load_and_prepare_data(str(latest_dataset))
        
        # Train all models
        training_results = predictor.train_all_models(X, y)
        
        # Store performance results
        for model_name, results in training_results.items():
            predictor.model_performance[model_name] = results['performance']
        
        # Save models
        saved_files = predictor.save_models()
        
        # Generate report
        report = predictor.generate_model_report()
        
        # Display results
        print("\n" + "="*60)
        print("MACHINE LEARNING MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"Models trained: {len(training_results)}")
        print(f"Features used: {len(predictor.feature_names)}")
        print(f"Best model: {report['best_model']}")
        
        print(f"\nModel Performance Summary:")
        for model_name, performance in report['model_performance'].items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {performance['accuracy']:.4f}")
            print(f"    F1-Score: {performance['f1_score']:.4f}")
            print(f"    ROC-AUC: {performance['roc_auc']:.4f}")
        
        print(f"\nTop 5 Most Important Features (Random Forest):")
        if 'random_forest' in report['feature_importance_summary']:
            for feature, importance in report['feature_importance_summary']['random_forest'][:5]:
                print(f"  {feature}: {importance:.4f}")
        
        print(f"\nModel files saved:")
        for model_name, filepath in saved_files.items():
            print(f"  {model_name}: {Path(filepath).name}")
        
        # Example prediction
        print(f"\n" + "="*40)
        print("EXAMPLE PREDICTION")
        print("="*40)
        
        # Use first 5 samples for demonstration
        sample_data = X.head(5)
        predictions = predictor.predict_deforestation_risk(sample_data, 'ensemble')
        
        print(f"Sample predictions using ensemble model:")
        for i, (pred, prob) in enumerate(zip(predictions['predictions'], predictions['risk_scores'])):
            print(f"  Sample {i+1}: {pred} risk (confidence: {prob:.3f})")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
