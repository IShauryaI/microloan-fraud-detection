"""
Microloan Fraud Detection Engine

A comprehensive fraud detection system that analyzes loan applications
and calculates risk scores based on multiple fraud indicators including
identity patterns, address analysis, financial ratios, and geographic clustering.

Author: Shaurya Parshad
Date: September 2025
"""

import pandas as pd
import numpy as np
import re
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionConfig:
    """Configuration class for fraud detection parameters"""
    
    # Suspicious name patterns and weights
    SUSPICIOUS_PATTERNS = {
        r'\b(fake|test|sample|dummy)\b': 0.5,
        r'\b(llc|inc|corp)\s*\d+\b': 0.3,  # Generic business names with numbers
        r'^[a-z]+\s*\d+$': 0.4,  # Simple pattern like "business123"
    }
    PATTERN_MULTIPLIER = 3
    
    # Address analysis indicators
    RESIDENTIAL_INDICATORS = {
        'apt': 0.8, 'unit': 0.7, '#': 0.7, 'suite': 0.4, 'floor': 0.3,
        'po box': 0.9, 'p.o.': 0.9, 'box': 0.8, 'residence': 0.9,
        'residential': 0.9, 'apartment': 0.9, 'house': 0.8, 'condo': 0.8,
        'room': 0.9
    }
    COMMERCIAL_INDICATORS = {
        'plaza': -0.7, 'building': -0.5, 'tower': -0.6, 'office': -0.7,
        'complex': -0.5, 'center': -0.5, 'mall': -0.8, 'commercial': -0.8,
        'industrial': -0.8, 'park': -0.4, 'warehouse': -0.8, 'factory': -0.8,
        'store': -0.7, 'shop': -0.6
    }
    
    # Scoring thresholds and points
    RES_THRESHOLD = 0.7
    RES_POINTS = 10
    MULTI_BUS_RES_POINTS = 15
    MULTI_BUS_COMM_POINTS = 8
    GEO_CLUSTER_COUNT = 2
    GEO_CLUSTER_POINTS = 18
    EMP_RATIO_THRESH = 4000
    EMP_RATIO_POINTS = 15
    VHIGH_RATIO_POINTS = 15
    ONE_EMP_POINTS = 30
    TWO_EMP_POINTS = 20
    PROG_MAX_AMOUNT = 20000
    PROG_MAX_POINTS = 25
    BATCH_UNIFORM_MIN = 5
    BATCH_UNIFORM_PCT = 0.10
    BATCH_UNIFORM_POINTS = 15
    ZIP_SPIKE_MIN = 5
    ZIP_SPIKE_MULT = 5
    ZIP_SPIKE_POINTS = 20
    SBA_SPIKE_MULT = 15
    SBA_SPIKE_POINTS = 8
    DEMO_MISSING_POINTS = 10
    SEQ_POINTS = 25


class FraudRiskScorer:
    """
    Main fraud detection engine that analyzes loan applications
    and calculates comprehensive risk scores.
    """
    
    def __init__(self, config: Optional[FraudDetectionConfig] = None):
        """
        Initialize the fraud risk scorer with configuration parameters.
        
        Args:
            config: Configuration object with fraud detection parameters
        """
        self.config = config or FraudDetectionConfig()
        self._compile_patterns()
        logger.info("Fraud Risk Scorer initialized successfully")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient pattern matching"""
        self._name_pattern_regex = []
        for pattern, weight in self.config.SUSPICIOUS_PATTERNS.items():
            self._name_pattern_regex.append(
                (re.compile(pattern, re.IGNORECASE), weight)
            )
        logger.debug(f"Compiled {len(self._name_pattern_regex)} regex patterns")
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive fraud risk scores for loan applications.
        
        Args:
            df: DataFrame containing loan application data
        
        Returns:
            DataFrame with LoanNumber, BorrowerName, and RiskScore columns
        
        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        required_columns = [
            'LoanNumber', 'BorrowerName', 'BorrowerAddress', 
            'BorrowerZip', 'InitialApprovalAmount', 'JobsReported'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Calculating fraud scores for {len(df)} loan applications")
        
        # Create working copy and initialize score
        df_work = df.copy()
        df_work['RiskScore'] = 0.0
        
        # Prepare helper columns
        df_work = self._prepare_helper_columns(df_work)
        
        # Apply scoring rules
        df_work = self._score_name_patterns(df_work)
        df_work = self._score_address_analysis(df_work)
        df_work = self._score_multi_business_addresses(df_work)
        df_work = self._score_geographic_clustering(df_work)
        df_work = self._score_financial_ratios(df_work)
        df_work = self._score_employee_counts(df_work)
        df_work = self._score_program_maximums(df_work)
        df_work = self._score_sequential_patterns(df_work)
        df_work = self._score_batch_uniformity(df_work)
        df_work = self._score_geographic_spikes(df_work)
        df_work = self._score_sba_office_spikes(df_work)
        df_work = self._score_demographic_gaps(df_work)
        
        result_df = df_work[['LoanNumber', 'BorrowerName', 'RiskScore']].copy()
        
        # Log scoring summary
        high_risk_count = len(result_df[result_df['RiskScore'] > 100])
        logger.info(f"Scoring complete. {high_risk_count} applications flagged as high-risk")
        logger.info(f"Average risk score: {result_df['RiskScore'].mean():.2f}")
        logger.info(f"Max risk score: {result_df['RiskScore'].max():.2f}")
        
        return result_df
    
    def _prepare_helper_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare normalized helper columns for analysis"""
        df['__addr'] = df['BorrowerAddress'].str.lower().fillna('')
        df['__name'] = df['BorrowerName'].str.lower().fillna('')
        df['__zip5'] = df['BorrowerZip'].astype(str).str[:5]
        
        # Handle latitude/longitude if present
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df['__latr'] = df['Latitude'].round(3)
            df['__lonr'] = df['Longitude'].round(3)
        else:
            df['__latr'] = 0
            df['__lonr'] = 0
        
        df['__amt'] = pd.to_numeric(df['InitialApprovalAmount'], errors='coerce').fillna(0)
        df['__jobs'] = pd.to_numeric(df['JobsReported'], errors='coerce').fillna(0)
        
        return df
    
    def _score_name_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score based on suspicious name patterns"""
        def score_name(name: str) -> float:
            score = 0
            for regex, weight in self._name_pattern_regex:
                if regex.search(name):
                    score += self.config.PATTERN_MULTIPLIER * weight
            return score
        
        df['RiskScore'] += df['__name'].map(score_name)
        return df
    
    def _score_address_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score based on residential vs commercial address indicators"""
        def calculate_residential_score(addr: str) -> float:
            score = 0
            for indicator, weight in self.config.RESIDENTIAL_INDICATORS.items():
                if indicator in addr:
                    score += weight
            for indicator, weight in self.config.COMMERCIAL_INDICATORS.items():
                if indicator in addr:
                    score += weight
            return score
        
        df['__res_sum'] = df['__addr'].map(calculate_residential_score)
        mask_residential = df['__res_sum'] > self.config.RES_THRESHOLD
        df.loc[mask_residential, 'RiskScore'] += self.config.RES_POINTS
        
        return df
    
    def _score_multi_business_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score multiple businesses at the same address"""
        addr_counts = df.groupby('__addr').size()
        df['__addr_count'] = df['__addr'].map(addr_counts)
        
        mask_multi = df['__addr_count'] > 1
        mask_residential = df['__res_sum'] > self.config.RES_THRESHOLD
        
        # Penalize multiple businesses at residential addresses more heavily
        df.loc[mask_multi & mask_residential, 'RiskScore'] += \
            (df.loc[mask_multi & mask_residential, '__addr_count'] - 1) * self.config.MULTI_BUS_RES_POINTS
        
        df.loc[mask_multi & ~mask_residential, 'RiskScore'] += \
            (df.loc[mask_multi & ~mask_residential, '__addr_count'] - 1) * self.config.MULTI_BUS_COMM_POINTS
        
        return df
    
    def _score_geographic_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score geographic clustering of applications"""
        if '__latr' in df.columns and '__lonr' in df.columns:
            geo_counts = df.groupby(['__latr', '__lonr']).size()
            df['__geo_count'] = df.set_index(['__latr', '__lonr']).index.map(geo_counts)
            
            mask_cluster = df['__geo_count'] > self.config.GEO_CLUSTER_COUNT
            df.loc[mask_cluster, 'RiskScore'] += self.config.GEO_CLUSTER_POINTS
        
        return df
    
    def _score_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score suspicious loan amount to employee ratios"""
        df['__ratio'] = df['__amt'] / df['__jobs'].replace(0, np.nan)
        
        mask_high_ratio = df['__ratio'] > self.config.EMP_RATIO_THRESH
        df.loc[mask_high_ratio, 'RiskScore'] += \
            self.config.EMP_RATIO_POINTS + self.config.VHIGH_RATIO_POINTS
        
        return df
    
    def _score_employee_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score suspiciously low employee counts"""
        df.loc[df['__jobs'] == 1, 'RiskScore'] += self.config.ONE_EMP_POINTS
        df.loc[df['__jobs'] == 2, 'RiskScore'] += self.config.TWO_EMP_POINTS
        return df
    
    def _score_program_maximums(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score loans at program maximum amounts"""
        df.loc[df['__amt'] == self.config.PROG_MAX_AMOUNT, 'RiskScore'] += \
            self.config.PROG_MAX_POINTS
        return df
    
    def _score_sequential_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score sequential loan number patterns by lender"""
        if 'OriginatingLender' not in df.columns:
            return df
        
        sequential_lenders = defaultdict(bool)
        
        for lender, group in df.groupby('OriginatingLender'):
            # Extract numeric portions of loan numbers
            loan_nums = group['LoanNumber'].str.extract(r'(\d+)$', expand=False)
            loan_nums = loan_nums.dropna().astype(int)
            
            if len(loan_nums) > 1:
                sorted_nums = sorted(loan_nums)
                # Check for consecutive numbers
                consecutive_found = any(
                    b - a == 1 for a, b in zip(sorted_nums, sorted_nums[1:])
                )
                if consecutive_found:
                    sequential_lenders[lender] = True
        
        mask_sequential = df['OriginatingLender'].map(sequential_lenders)
        df.loc[mask_sequential, 'RiskScore'] += self.config.SEQ_POINTS
        
        return df
    
    def _score_batch_uniformity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score uniform batch submissions"""
        if 'OriginatingLender' not in df.columns or 'DateApproved' not in df.columns:
            return df
        
        for (lender, date, zip_code), group in df.groupby(['OriginatingLender', 'DateApproved', '__zip5']):
            if len(group) >= self.config.BATCH_UNIFORM_MIN:
                amounts = group['__amt']
                min_amt, max_amt = amounts.min(), amounts.max()
                
                if min_amt > 0 and (max_amt - min_amt) <= self.config.BATCH_UNIFORM_PCT * min_amt:
                    df.loc[group.index, 'RiskScore'] += self.config.BATCH_UNIFORM_POINTS
        
        return df
    
    def _score_geographic_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score unusual geographic spikes in applications"""
        if 'DateApproved' not in df.columns:
            return df
        
        zip_day_counts = df.groupby(['DateApproved', '__zip5']).size().rename('count')
        avg_zip_counts = zip_day_counts.groupby(level=1).mean()
        
        for (date, zip_code), count in zip_day_counts.items():
            if (count > self.config.ZIP_SPIKE_MIN and 
                zip_code in avg_zip_counts and
                avg_zip_counts[zip_code] > 0 and
                (count / avg_zip_counts[zip_code]) > self.config.ZIP_SPIKE_MULT):
                
                mask = (df['DateApproved'] == date) & (df['__zip5'] == zip_code)
                df.loc[mask, 'RiskScore'] += self.config.ZIP_SPIKE_POINTS
        
        return df
    
    def _score_sba_office_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score unusual SBA office processing spikes"""
        if 'SBAOfficeCode' not in df.columns or 'DateApproved' not in df.columns:
            return df
        
        office_day_counts = df.groupby(['DateApproved', 'SBAOfficeCode']).size().rename('count')
        avg_office_counts = office_day_counts.groupby(level=1).mean()
        
        for (date, office), count in office_day_counts.items():
            if (office and office in avg_office_counts and 
                avg_office_counts[office] > 0 and
                (count / avg_office_counts[office]) > self.config.SBA_SPIKE_MULT):
                
                mask = (df['DateApproved'] == date) & (df['SBAOfficeCode'] == office)
                df.loc[mask, 'RiskScore'] += self.config.SBA_SPIKE_POINTS
        
        return df
    
    def _score_demographic_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score missing or unanswered demographic information"""
        demo_cols = ['Race', 'Gender', 'Ethnicity']
        available_cols = [col for col in demo_cols if col in df.columns]
        
        if available_cols:
            mask_missing = df[available_cols].apply(
                lambda row: row.str.lower().isin(['unanswered', 'unknown', '']).all(), 
                axis=1
            )
            df.loc[mask_missing, 'RiskScore'] += self.config.DEMO_MISSING_POINTS
        
        return df


def calculate_fraud_scores(
    input_file: str,
    output_file: str,
    config: Optional[FraudDetectionConfig] = None
) -> pd.DataFrame:
    """
    Main function to calculate fraud scores for a dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        config: Optional configuration object
    
    Returns:
        DataFrame with fraud scores
    """
    try:
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, dtype=str, low_memory=False)
        logger.info(f"Loaded {len(df)} records")
        
        # Initialize scorer and calculate scores
        scorer = FraudRiskScorer(config)
        results = scorer.calculate_scores(df)
        
        # Save results
        results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing fraud scores: {str(e)}")
        raise


def main():
    """Command-line interface for fraud detection"""
    parser = argparse.ArgumentParser(description='Microloan Fraud Detection System')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        calculate_fraud_scores(args.input_file, args.output_file)
        logger.info("Fraud detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())