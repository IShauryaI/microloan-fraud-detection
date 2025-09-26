# Microloan Fraud Detection System

A comprehensive fraud detection system for microloan applications using data warehousing techniques, machine learning, and automated ETL pipelines.

## Overview

This project analyzes over 11 million PPP (Paycheck Protection Program) loan records to develop and validate fraud detection methodologies applicable to microloan systems. Using a combination of rule-based scoring and machine learning approaches, the system identifies potentially fraudulent loan applications through pattern recognition and anomaly detection.

## Key Features

- **Automated ETL Pipeline**: Apache Airflow orchestrated data processing workflows
- **Rule-Based Fraud Scoring**: 15+ fraud indicators including geo-clustering, identity patterns, and application velocity
- **Machine Learning Classification**: XGBoost-based fraud detection with feature engineering
- **Interactive Analytics**: Comprehensive visualization dashboards for fraud pattern analysis
- **Scalable Architecture**: Modular design supporting batch processing of millions of records

## Architecture

```
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Raw Data        │────│ ETL Pipeline     │────│ Data Warehouse  │
│ (PPP Loans)     │    │ (Apache Airflow) │    │ (Processed)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                         │                       │
                         ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Fraud Scoring    │────│ ML Training     │
                       │ (Rule-Based)     │    │ (XGBoost)       │
                       └──────────────────┘    └─────────────────┘
                         │                       │
                         ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Risk Reports     │    │ Model Outputs   │
                       │ & Visualizations │    │ & Predictions   │
                       └──────────────────┘    └─────────────────┘
```

## Project Structure

```
microloan-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── fraud_detector.py      # Main fraud scoring engine
│   ├── model_trainer.py       # ML model training and evaluation
│   ├── data_analyzer.py       # Data analysis and visualization
│   └── utils/
│       └── data_processing.py
├── airflow_dags/
│   └── fraud_detection_dag.py # Airflow pipeline definition
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/                   # Original dataset files
│   ├── processed/             # Cleaned and transformed data
│   └── outputs/               # Risk scores and model results
├── notebooks/
│   └── exploratory_analysis.ipynb
├── docs/
│   ├── project_proposal.pdf
│   └── final_report.pdf
└── tests/
    └── test_fraud_detector.py
```

## Installation

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/IShauryaI/microloan-fraud-detection.git
cd microloan-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Docker Setup

```bash
cd docker/
docker-compose up --build
```

## Usage

### 1. Data Processing and Fraud Scoring

```bash
# Run fraud detection on dataset
python src/fraud_detector.py data/raw/ppp_data.csv data/outputs/fraud_scores.csv

# Or use the complete pipeline
python -m src.main --input data/raw/ --output data/outputs/
```

### 2. Model Training

```bash
# Train XGBoost classifier
python src/model_trainer.py --train-data data/processed/training_set.csv --output models/fraud_classifier.joblib
```

### 3. Generate Analytics Dashboard

```bash
# Create fraud analysis visualizations
python src/data_analyzer.py --scores data/outputs/fraud_scores.csv --output reports/
```

### 4. Run Complete Pipeline with Airflow

```bash
# Start Airflow scheduler
airflow scheduler

# Trigger fraud detection DAG
airflow dags trigger fraud_detection_pipeline
```

## Fraud Detection Methodology

### Rule-Based Scoring Components

The system evaluates multiple fraud indicators, each contributing to an overall risk score:

| **Category** | **Indicators** | **Risk Points** |
|--------------|---------------|----------------|
| **Identity Patterns** | Suspicious business names, fake identifiers | 5-15 points |
| **Address Analysis** | Residential vs commercial, address clustering | 8-18 points |
| **Financial Ratios** | Loan amount per employee, round number detection | 15-30 points |
| **Geographic Clustering** | Multiple businesses at same location | 18 points |
| **Temporal Patterns** | Application velocity, batch submissions | 15-25 points |
| **Data Quality** | Missing demographics, incomplete information | 10 points |

### Machine Learning Enhancement

- **Algorithm**: XGBoost Classifier with hyperparameter optimization
- **Features**: 50+ engineered features from rule-based analysis and raw data
- **Performance**: 96% ROC-AUC, 60.5% overall accuracy
- **Class Balancing**: Addresses severe class imbalance (fraud rate ~6.5%)

## Results and Performance

### Dataset Statistics

- **Total Records**: 11+ million PPP loan applications
- **Analysis Focus**: Loans between $10,000-$20,000 (2+ million records)
- **Fraud Cases Identified**: 785,365 high-risk applications
- **Processing Time**: ~45 minutes for complete pipeline

### Model Performance

```
ROC-AUC Score: 0.958
Average Precision: 0.577
Overall Accuracy: 60.5%
Fraud Class Precision: 0.127
Fraud Class Recall: 0.998
F1-Score (Fraud): 0.225
```

### Key Findings

- **Geographic Hotspots**: California, Texas, Florida show highest fraud concentrations
- **Temporal Patterns**: Fraud applications peaked during program launch periods
- **Risk Indicators**: Single-employee businesses and residential addresses strong predictors
- **Application Velocity**: Batch submissions and sequential loan numbers indicate organized fraud

## Known Limitations

### Technical Limitations

- **Memory Constraints**: Large datasets (>2M records) may cause OOM errors
- **Processing Speed**: Complex scoring rules require optimization for real-time use
- **Model Performance**: High false positive rate needs improvement for production use

### Data Quality Issues

- **Class Imbalance**: Severe fraud underrepresentation affects ML performance
- **Feature Engineering**: Limited to transaction data, missing behavioral signals
- **Temporal Validation**: No forward-looking validation on fraud evolution

### Scalability Concerns

- **Single-Machine Processing**: Current architecture doesn't support distributed computing
- **Database Integration**: File-based processing limits query performance
- **Real-Time Capabilities**: Batch processing unsuitable for live fraud detection

## Future Improvements

### Technical Enhancements

- Implement streaming data processing with Kafka/Spark
- Add database backend (PostgreSQL/MongoDB) for better data management
- Develop REST API for real-time fraud scoring
- Create web-based dashboard for fraud investigation workflow

### Model Improvements

- Address class imbalance with SMOTE/advanced sampling techniques
- Implement ensemble methods combining multiple algorithms
- Add unsupervised anomaly detection for unknown fraud patterns
- Integrate deep learning approaches for sequence analysis

### Operational Features

- Add model drift detection and automated retraining
- Implement A/B testing framework for rule optimization
- Create fraud investigator feedback loop
- Develop compliance reporting and audit trails

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Academic Context

This project was developed as part of advanced data mining coursework, focusing on practical applications of data warehousing techniques in financial fraud detection. The research demonstrates the integration of traditional business intelligence approaches with modern machine learning methodologies.

**Authors**: Timothy Eric Dare, Ahmet Furkan Eren, Shaurya Parshad, K. Sridharan Shrinicketh, Bala Kishore Mokhamatla, Ayeshkant Mallick

## Contact

For questions about this project or potential collaboration opportunities, please reach out via GitHub Issues.

**⚠️ Disclaimer**: This system is designed for research and educational purposes. Any production deployment requires additional validation, compliance review, and performance optimization.