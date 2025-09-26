"""
Enhanced Machine Learning Model Trainer for Fraud Detection

This module provides improved ML training with proper class balancing,
cross-validation, and comprehensive evaluation metrics to address
the poor precision issues in the original implementation.

Author: Shaurya Parshad
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """
    Enhanced ML model trainer with proper class balancing and evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        
        # Set up models with optimized hyperparameters
        self._initialize_models()
        
        logger.info("Fraud Model Trainer initialized successfully")
    
    def _initialize_models(self) -> None:
        """Initialize ML models with optimized parameters"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        }
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML training.
        
        Args:
            df: Input dataframe with fraud scores and labels
            target_col: Name of the target column
        
        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing features for ML training")
        
        # Create target variable if not exists
        if target_col not in df.columns:
            # Create fraud labels based on risk score threshold
            threshold = df['RiskScore'].quantile(0.95)  # Top 5% as fraud
            df[target_col] = (df['RiskScore'] > threshold).astype(int)
            logger.info(f"Created fraud labels with threshold: {threshold:.2f}")
        
        # Feature engineering
        features = pd.DataFrame()
        
        # Risk score features
        if 'RiskScore' in df.columns:
            features['risk_score'] = df['RiskScore']
            features['risk_score_log'] = np.log1p(df['RiskScore'])
            features['risk_score_sqrt'] = np.sqrt(df['RiskScore'])
        
        # Amount features
        if 'InitialApprovalAmount' in df.columns:
            amount = pd.to_numeric(df['InitialApprovalAmount'], errors='coerce').fillna(0)
            features['loan_amount'] = amount
            features['loan_amount_log'] = np.log1p(amount)
            features['loan_amount_binned'] = pd.cut(amount, bins=5, labels=False)
        
        # Employee features
        if 'JobsReported' in df.columns:
            jobs = pd.to_numeric(df['JobsReported'], errors='coerce').fillna(0)
            features['jobs_reported'] = jobs
            features['jobs_is_single'] = (jobs == 1).astype(int)
            features['jobs_is_small'] = (jobs <= 5).astype(int)
            
            # Loan per employee ratio
            if 'loan_amount' in features.columns:
                features['amount_per_employee'] = features['loan_amount'] / (jobs + 1)  # +1 to avoid division by zero
                features['amount_per_employee_log'] = np.log1p(features['amount_per_employee'])
        
        # Address features
        if 'BorrowerAddress' in df.columns:
            addr = df['BorrowerAddress'].fillna('').str.lower()
            features['addr_has_apt'] = addr.str.contains('apt|unit|#', na=False).astype(int)
            features['addr_has_po_box'] = addr.str.contains('po box|p.o.', na=False).astype(int)
            features['addr_length'] = addr.str.len()
        
        # State features
        if 'BorrowerState' in df.columns:
            state_encoder = LabelEncoder()
            features['borrower_state'] = state_encoder.fit_transform(df['BorrowerState'].fillna('Unknown'))
            
            # High-risk states (based on analysis)
            high_risk_states = ['CA', 'TX', 'FL', 'NY']
            features['state_is_high_risk'] = df['BorrowerState'].isin(high_risk_states).astype(int)
        
        # Business name features
        if 'BorrowerName' in df.columns:
            name = df['BorrowerName'].fillna('').str.lower()
            features['name_length'] = name.str.len()
            features['name_has_numbers'] = name.str.contains(r'\d', na=False).astype(int)
            features['name_has_llc'] = name.str.contains('llc', na=False).astype(int)
            features['name_has_inc'] = name.str.contains('inc', na=False).astype(int)
        
        # Date features (if available)
        if 'DateApproved' in df.columns:
            try:
                date_approved = pd.to_datetime(df['DateApproved'], errors='coerce')
                features['approval_month'] = date_approved.dt.month
                features['approval_day_of_week'] = date_approved.dt.dayofweek
                features['approval_is_weekend'] = (date_approved.dt.dayofweek >= 5).astype(int)
            except:
                logger.warning("Could not parse DateApproved column")
        
        # Handle missing values
        features = features.fillna(0)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Created {len(features.columns)} features for {len(features)} samples")
        
        return features, df[target_col]
    
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset to address class imbalance.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Balancing method ('smote', 'undersample', or 'combined')
        
        Returns:
            Balanced feature matrix and target vector
        """
        logger.info(f"Balancing dataset using {method} method")
        
        # Log original class distribution
        fraud_count = y.sum()
        total_count = len(y)
        logger.info(f"Original distribution - Fraud: {fraud_count}, Non-fraud: {total_count - fraud_count}")
        logger.info(f"Fraud rate: {fraud_count/total_count:.3f}")
        
        if method == 'smote':
            # Use SMOTE for oversampling minority class
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        elif method == 'undersample':
            # Use random undersampling for majority class
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            
        elif method == 'combined':
            # Combine SMOTE with undersampling
            # First oversample minority class, then undersample majority class
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            undersampler = RandomUnderSampler(random_state=self.random_state, sampling_strategy=0.5)
            
            # Create pipeline
            pipeline = ImbPipeline([
                ('smote', smote),
                ('undersample', undersampler)
            ])
            
            X_balanced, y_balanced = pipeline.fit_resample(X, y)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Log balanced class distribution
        fraud_balanced = y_balanced.sum()
        total_balanced = len(y_balanced)
        logger.info(f"Balanced distribution - Fraud: {fraud_balanced}, Non-fraud: {total_balanced - fraud_balanced}")
        logger.info(f"Balanced fraud rate: {fraud_balanced/total_balanced:.3f}")
        
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    balance_data: bool = True) -> Dict[str, Any]:
        """
        Train multiple ML models with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            balance_data: Whether to balance the training data
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training")
        
        # Balance training data if requested
        if balance_data:
            X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train, method='smote')
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_balanced),
            columns=X_train_balanced.columns
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns
        )
        self.scalers['standard'] = scaler
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Use scaled features for logistic regression, original for tree-based models
                if model_name == 'logistic_regression':
                    X_train_model = X_train_scaled
                    X_val_model = X_val_scaled
                else:
                    X_train_model = X_train_balanced
                    X_val_model = X_val
                
                # Train model
                model.fit(X_train_model, y_train_balanced)
                
                # Make predictions
                y_pred = model.predict(X_val_model)
                y_pred_proba = model.predict_proba(X_val_model)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_model, y_train_balanced, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc'
                )
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.3f}, "
                          f"Precision: {metrics['precision']:.3f}, "
                          f"Recall: {metrics['recall']:.3f}, "
                          f"F1: {metrics['f1']:.3f}")
                          
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                          y_pred_proba: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba)
        }
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       output_dir: str = 'outputs') -> None:
        """
        Evaluate trained models and generate visualizations.
        
        Args:
            X_test: Test features
            y_test: Test targets
            output_dir: Directory to save evaluation plots
        """
        logger.info("Evaluating models on test set")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Evaluate each model
        for model_name, result in self.results.items():
            model = result['model']
            
            # Use appropriate features (scaled or not)
            if model_name == 'logistic_regression':
                X_test_model = pd.DataFrame(
                    self.scalers['standard'].transform(X_test),
                    columns=X_test.columns
                )
            else:
                X_test_model = X_test
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Calculate test metrics
            test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            result['test_metrics'] = test_metrics
            
            logger.info(f"\n{model_name.upper()} Test Results:")
            logger.info(f"ROC-AUC: {test_metrics['roc_auc']:.3f}")
            logger.info(f"Precision: {test_metrics['precision']:.3f}")
            logger.info(f"Recall: {test_metrics['recall']:.3f}")
            logger.info(f"F1-Score: {test_metrics['f1']:.3f}")
            
            # Generate evaluation plots
            self._plot_evaluation(y_test, y_pred, y_pred_proba, model_name, output_dir)
    
    def _plot_evaluation(self, y_true: pd.Series, y_pred: pd.Series,
                        y_pred_proba: pd.Series, model_name: str,
                        output_dir: str) -> None:
        """
        Generate evaluation plots for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation: {model_name.title().replace("_", " ")}', fontsize=16)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        axes[1, 0].plot(recall, precision, color='red', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        
        # 4. Prediction Distribution
        axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, 
                        label='Non-fraud', color='blue', density=True)
        axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, 
                        label='Fraud', color='red', density=True)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_best_model(self, output_path: str, metric: str = 'roc_auc') -> str:
        """
        Save the best model based on specified metric.
        
        Args:
            output_path: Path to save the model
            metric: Metric to use for model selection
        
        Returns:
            Name of the best model
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        # Find best model
        best_model_name = max(
            self.results.keys(),
            key=lambda x: self.results[x]['metrics'][metric]
        )
        
        best_model = self.results[best_model_name]['model']
        
        # Save model and scaler
        model_data = {
            'model': best_model,
            'scaler': self.scalers.get('standard'),
            'feature_names': self.feature_names,
            'model_name': best_model_name,
            'metrics': self.results[best_model_name]['metrics']
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Best model ({best_model_name}) saved to {output_path}")
        logger.info(f"Best {metric}: {self.results[best_model_name]['metrics'][metric]:.3f}")
        
        return best_model_name


def main():
    """Command-line interface for model training"""
    parser = argparse.ArgumentParser(description='Fraud Detection Model Training')
    parser.add_argument('--train-data', required=True, help='Training data CSV file')
    parser.add_argument('--output', default='models/fraud_model.joblib', help='Output model path')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for plots')
    parser.add_argument('--balance-method', default='smote', 
                       choices=['smote', 'undersample', 'combined'],
                       help='Data balancing method')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load data
        logger.info(f"Loading training data from {args.train_data}")
        df = pd.read_csv(args.train_data)
        
        # Initialize trainer
        trainer = FraudModelTrainer()
        
        # Prepare features
        X, y = trainer.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, 
            random_state=42, stratify=y
        )
        
        # Further split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, 
            random_state=42, stratify=y_train
        )
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train models
        results = trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        trainer.evaluate_models(X_test, y_test, args.output_dir)
        
        # Save best model
        trainer.save_best_model(args.output)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())