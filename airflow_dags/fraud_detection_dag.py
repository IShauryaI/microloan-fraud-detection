"""
Apache Airflow DAG for Microloan Fraud Detection Pipeline

This DAG orchestrates the complete fraud detection workflow including:
- Data ingestion and preprocessing
- Fraud risk score calculation 
- Model training and evaluation
- Report generation and visualization

Author: Shaurya Parshad
Date: September 2025
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import pandas as pd
import logging
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# DAG configuration
default_args = {
    'owner': 'fraud-detection-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Initialize DAG
dag = DAG(
    'fraud_detection_pipeline',
    default_args=default_args,
    description='Complete fraud detection pipeline for microloan applications',
    schedule_interval='@daily',  # Run daily
    max_active_runs=1,
    tags=['fraud-detection', 'machine-learning', 'data-pipeline']
)


# Configuration
DATA_DIR = '/app/data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
OUTPUTS_DIR = '/app/outputs'
MODELS_DIR = '/app/models'
REPORTS_DIR = '/app/reports'


def validate_data_quality(**context):
    """
    Validate data quality and completeness.
    
    Returns:
        bool: True if data quality checks pass
    """
    logger.info("Starting data quality validation")
    
    input_file = f"{RAW_DATA_DIR}/ppp_loans.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input data file not found: {input_file}")
    
    # Load and validate data
    df = pd.read_csv(input_file, nrows=1000)  # Sample for validation
    
    # Required columns check
    required_cols = [
        'LoanNumber', 'BorrowerName', 'BorrowerAddress', 
        'BorrowerZip', 'InitialApprovalAmount', 'JobsReported'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Data completeness checks
    null_percentages = df[required_cols].isnull().mean()
    high_null_cols = null_percentages[null_percentages > 0.5].index.tolist()
    
    if high_null_cols:
        logger.warning(f"High null percentage in columns: {high_null_cols}")
    
    # Data type validation
    try:
        pd.to_numeric(df['InitialApprovalAmount'], errors='raise')
        pd.to_numeric(df['JobsReported'], errors='raise')
    except (ValueError, TypeError) as e:
        raise ValueError(f"Data type validation failed: {str(e)}")
    
    logger.info(f"Data quality validation passed for {len(df)} sample records")
    return True


def preprocess_data(**context):
    """
    Preprocess raw loan data for fraud detection.
    
    Returns:
        str: Path to processed data file
    """
    logger.info("Starting data preprocessing")
    
    input_file = f"{RAW_DATA_DIR}/ppp_loans.csv"
    output_file = f"{PROCESSED_DATA_DIR}/preprocessed_loans.csv"
    
    # Create output directory
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file, dtype=str, low_memory=False)
    logger.info(f"Loaded {len(df)} records for preprocessing")
    
    # Data cleaning
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['LoanNumber'], keep='first')
    logger.info(f"Removed {initial_count - len(df)} duplicate records")
    
    # Clean and validate loan numbers
    df = df[df['LoanNumber'].notna() & (df['LoanNumber'].str.len() > 0)]
    
    # Clean amounts and jobs
    df['InitialApprovalAmount'] = pd.to_numeric(
        df['InitialApprovalAmount'], errors='coerce'
    ).fillna(0)
    df['JobsReported'] = pd.to_numeric(
        df['JobsReported'], errors='coerce'
    ).fillna(0)
    
    # Filter for analysis range (e.g., $10K-$20K loans)
    df_filtered = df[
        (df['InitialApprovalAmount'] >= 10000) & 
        (df['InitialApprovalAmount'] <= 20000)
    ].copy()
    
    logger.info(f"Filtered to {len(df_filtered)} loans in $10K-$20K range")
    
    # Add preprocessing metadata
    df_filtered['ProcessedDate'] = datetime.now().isoformat()
    df_filtered['DataSource'] = 'PPP_Dataset'
    
    # Save processed data
    df_filtered.to_csv(output_file, index=False)
    logger.info(f"Preprocessed data saved to {output_file}")
    
    # Store metadata for downstream tasks
    context['task_instance'].xcom_push(
        key='processed_records_count', 
        value=len(df_filtered)
    )
    context['task_instance'].xcom_push(
        key='processed_file_path', 
        value=output_file
    )
    
    return output_file


def calculate_fraud_scores(**context):
    """
    Calculate fraud risk scores for loan applications.
    
    Returns:
        str: Path to fraud scores file
    """
    logger.info("Starting fraud score calculation")
    
    # Get processed data path from upstream task
    input_file = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_file_path'
    )
    
    output_file = f"{OUTPUTS_DIR}/fraud_scores.csv"
    
    # Create output directory
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Import and run fraud detection
    import sys
    sys.path.append('/app')
    from src.fraud_detector import calculate_fraud_scores
    
    # Calculate scores
    results = calculate_fraud_scores(input_file, output_file)
    
    # Log summary statistics
    high_risk_count = len(results[results['RiskScore'] > 100])
    avg_risk = results['RiskScore'].mean()
    max_risk = results['RiskScore'].max()
    
    logger.info(f"Fraud scoring complete:")
    logger.info(f"- Total applications: {len(results)}")
    logger.info(f"- High-risk applications (>100): {high_risk_count}")
    logger.info(f"- Average risk score: {avg_risk:.2f}")
    logger.info(f"- Maximum risk score: {max_risk:.2f}")
    
    # Store results for downstream tasks
    context['task_instance'].xcom_push(
        key='fraud_scores_count', 
        value=len(results)
    )
    context['task_instance'].xcom_push(
        key='high_risk_count', 
        value=high_risk_count
    )
    context['task_instance'].xcom_push(
        key='fraud_scores_file', 
        value=output_file
    )
    
    return output_file


def train_ml_model(**context):
    """
    Train machine learning model for fraud detection.
    
    Returns:
        str: Path to trained model file
    """
    logger.info("Starting ML model training")
    
    # Get fraud scores file from upstream task
    scores_file = context['task_instance'].xcom_pull(
        task_ids='calculate_fraud_scores',
        key='fraud_scores_file'
    )
    
    # Get processed data for features
    processed_file = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_file_path'
    )
    
    output_model = f"{MODELS_DIR}/fraud_model.joblib"
    
    # Create models directory
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data and merge
    df_scores = pd.read_csv(scores_file)
    df_processed = pd.read_csv(processed_file)
    
    # Merge datasets
    df_merged = df_processed.merge(df_scores, on='LoanNumber', how='inner')
    
    # Import and run model training
    import sys
    sys.path.append('/app')
    from src.model_trainer import FraudModelTrainer
    
    # Initialize trainer
    trainer = FraudModelTrainer(random_state=42)
    
    # Prepare features and split data
    X, y = trainer.prepare_features(df_merged)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Train models
    results = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    trainer.evaluate_models(X_test, y_test, REPORTS_DIR)
    
    # Save best model
    best_model = trainer.save_best_model(output_model)
    
    logger.info(f"Model training complete. Best model: {best_model}")
    
    # Store results
    context['task_instance'].xcom_push(
        key='best_model_name', 
        value=best_model
    )
    context['task_instance'].xcom_push(
        key='model_file_path', 
        value=output_model
    )
    
    return output_model


def generate_analytics_report(**context):
    """
    Generate comprehensive analytics report.
    
    Returns:
        str: Path to analytics report directory
    """
    logger.info("Generating analytics report")
    
    # Get file paths from upstream tasks
    scores_file = context['task_instance'].xcom_pull(
        task_ids='calculate_fraud_scores',
        key='fraud_scores_file'
    )
    
    processed_file = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_file_path'
    )
    
    output_dir = f"{REPORTS_DIR}/fraud_analytics"
    
    # Create reports directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Import and run analytics
    import sys
    sys.path.append('/app')
    from src.data_analyzer import FraudAnalyzer
    
    # Generate analytics
    analyzer = FraudAnalyzer()
    analyzer.create_fraud_analytics(scores_file, processed_file, output_dir)
    
    # Count generated files
    report_files = list(Path(output_dir).glob('*.png'))
    
    logger.info(f"Analytics report generated with {len(report_files)} visualizations")
    
    # Store results
    context['task_instance'].xcom_push(
        key='analytics_files_count', 
        value=len(report_files)
    )
    context['task_instance'].xcom_push(
        key='analytics_report_dir', 
        value=output_dir
    )
    
    return output_dir


def send_summary_notification(**context):
    """
    Send pipeline completion summary.
    
    Returns:
        dict: Summary of pipeline execution
    """
    logger.info("Preparing pipeline summary")
    
    # Collect metrics from all tasks
    processed_count = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_records_count'
    )
    
    fraud_scores_count = context['task_instance'].xcom_pull(
        task_ids='calculate_fraud_scores',
        key='fraud_scores_count'
    )
    
    high_risk_count = context['task_instance'].xcom_pull(
        task_ids='calculate_fraud_scores',
        key='high_risk_count'
    )
    
    best_model = context['task_instance'].xcom_pull(
        task_ids='train_ml_model',
        key='best_model_name'
    )
    
    analytics_files = context['task_instance'].xcom_pull(
        task_ids='generate_analytics_report',
        key='analytics_files_count'
    )
    
    # Create summary
    summary = {
        'execution_date': context['ds'],
        'processed_records': processed_count,
        'fraud_scores_generated': fraud_scores_count,
        'high_risk_applications': high_risk_count,
        'fraud_rate_percent': round((high_risk_count / fraud_scores_count) * 100, 2) if fraud_scores_count > 0 else 0,
        'best_model': best_model,
        'analytics_files_generated': analytics_files,
        'pipeline_status': 'SUCCESS'
    }
    
    logger.info(f"Pipeline Summary: {summary}")
    
    # In a real implementation, you might send this via email, Slack, etc.
    logger.info("Fraud Detection Pipeline completed successfully!")
    logger.info(f"Processed {processed_count} loan applications")
    logger.info(f"Identified {high_risk_count} high-risk applications ({summary['fraud_rate_percent']}% fraud rate)")
    logger.info(f"Best performing model: {best_model}")
    logger.info(f"Generated {analytics_files} analytical reports")
    
    return summary


# Define tasks

# 1. Data validation and preprocessing group
with TaskGroup("data_preparation", dag=dag) as data_prep_group:
    
    # Check for new data files
    wait_for_data = FileSensor(
        task_id='wait_for_data_file',
        filepath=f'{RAW_DATA_DIR}/ppp_loans.csv',
        timeout=300,  # 5 minutes
        poke_interval=30,
        dag=dag
    )
    
    # Validate data quality
    validate_data = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        dag=dag
    )
    
    # Preprocess data
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        dag=dag
    )
    
    wait_for_data >> validate_data >> preprocess

# 2. Fraud detection scoring
fraud_scoring = PythonOperator(
    task_id='calculate_fraud_scores',
    python_callable=calculate_fraud_scores,
    dag=dag
)

# 3. Model training and evaluation group
with TaskGroup("model_training", dag=dag) as model_training_group:
    
    # Train ML models
    train_model = PythonOperator(
        task_id='train_ml_model',
        python_callable=train_ml_model,
        dag=dag
    )
    
    # Archive previous models (optional)
    archive_models = BashOperator(
        task_id='archive_previous_models',
        bash_command=f'mkdir -p {MODELS_DIR}/archive && mv {MODELS_DIR}/*.joblib {MODELS_DIR}/archive/ 2>/dev/null || true',
        dag=dag
    )
    
    archive_models >> train_model

# 4. Analytics and reporting
analytics_report = PythonOperator(
    task_id='generate_analytics_report',
    python_callable=generate_analytics_report,
    dag=dag
)

# 5. Pipeline completion notification
summary_notification = PythonOperator(
    task_id='send_summary_notification',
    python_callable=send_summary_notification,
    dag=dag
)

# 6. Cleanup temporary files
cleanup = BashOperator(
    task_id='cleanup_temp_files',
    bash_command=f'find {OUTPUTS_DIR} -name "*.tmp" -delete 2>/dev/null || true',
    dag=dag
)

# Define task dependencies
data_prep_group >> fraud_scoring >> [model_training_group, analytics_report]
model_training_group >> summary_notification
analytics_report >> summary_notification
summary_notification >> cleanup


# Task failure handling
def task_failure_alert(context):
    """
    Send alert on task failure.
    
    Args:
        context: Airflow context dictionary
    """
    task_id = context['task_instance'].task_id
    dag_id = context['task_instance'].dag_id
    execution_date = context['execution_date']
    
    logger.error(f"Task {task_id} in DAG {dag_id} failed on {execution_date}")
    
    # In production, you would send actual notifications here
    # e.g., send_slack_notification, send_email_alert, etc.


# Apply failure callback to all tasks
for task in dag.tasks:
    task.on_failure_callback = task_failure_alert