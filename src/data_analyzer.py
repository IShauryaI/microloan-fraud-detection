"""
Data Analysis and Visualization Module for Fraud Detection

This module provides comprehensive analytics and visualizations
for fraud detection results and patterns.

Author: Shaurya Parshad
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudAnalyzer:
    """
    Comprehensive fraud detection analytics and visualization.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the fraud analyzer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figures = []
        logger.info("Fraud Analyzer initialized successfully")
    
    def create_fraud_analytics(self, scores_file: str, processed_file: str, 
                              output_dir: str) -> None:
        """
        Create comprehensive fraud analytics dashboard.
        
        Args:
            scores_file: Path to fraud scores CSV
            processed_file: Path to processed data CSV 
            output_dir: Directory to save plots
        """
        logger.info("Creating comprehensive fraud analytics")
        
        # Load data
        try:
            df_scores = pd.read_csv(scores_file)
            logger.info(f"Loaded {len(df_scores)} fraud scores")
        except FileNotFoundError:
            logger.error(f"Fraud scores file not found: {scores_file}")
            return
        
        try:
            df_full = pd.read_csv(processed_file, parse_dates=['DateApproved'], errors='ignore')
            logger.info(f"Loaded {len(df_full)} processed records")
        except FileNotFoundError:
            logger.warning(f"Processed data file not found: {processed_file}")
            df_full = None
        
        # Merge data if both available
        if df_full is not None:
            df_flagged = df_full.merge(
                df_scores[['LoanNumber','RiskScore']], 
                on='LoanNumber', 
                how='inner'
            )
            logger.info(f"Merged data: {len(df_flagged)} records")
        else:
            df_flagged = df_scores
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self._plot_risk_score_distribution(df_scores, output_dir)
        
        if df_full is not None:
            self._plot_geographic_analysis(df_flagged, output_dir)
            self._plot_temporal_analysis(df_flagged, output_dir)
            self._plot_financial_analysis(df_flagged, output_dir)
            self._plot_business_analysis(df_flagged, output_dir)
        
        self._plot_risk_segmentation(df_scores, output_dir)
        
        logger.info(f"Analytics generated successfully in {output_dir}/")
        logger.info(f"Generated {len(list(Path(output_dir).glob('*.png')))} visualization files")
    
    def _plot_risk_score_distribution(self, df_scores: pd.DataFrame, 
                                     output_dir: str) -> None:
        """
        Plot risk score distribution analysis.
        
        Args:
            df_scores: DataFrame with risk scores
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Risk Score Distribution Analysis', fontsize=16, y=0.98)
        
        # 1. Overall distribution
        axes[0, 0].hist(df_scores['RiskScore'], bins=50, edgecolor='black', 
                       alpha=0.7, color='skyblue')
        axes[0, 0].axvline(df_scores['RiskScore'].mean(), color='red', 
                          linestyle='--', linewidth=2, 
                          label=f'Mean: {df_scores["RiskScore"].mean():.1f}')
        axes[0, 0].axvline(df_scores['RiskScore'].median(), color='orange', 
                          linestyle='--', linewidth=2, 
                          label=f'Median: {df_scores["RiskScore"].median():.1f}')
        axes[0, 0].set_title('Distribution of Fraud Risk Scores', fontsize=14)
        axes[0, 0].set_xlabel('Risk Score', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(df_scores['RiskScore'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 1].set_title('Risk Score Box Plot', fontsize=14)
        axes[0, 1].set_ylabel('Risk Score', fontsize=12)
        axes[0, 1].grid(True, linestyle='--', alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_scores = np.sort(df_scores['RiskScore'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 0].plot(sorted_scores, cumulative, linewidth=2, color='purple')
        axes[1, 0].set_title('Cumulative Distribution of Risk Scores', fontsize=14)
        axes[1, 0].set_xlabel('Risk Score', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1, 0].grid(True, linestyle='--', alpha=0.3)
        
        # 4. Percentile analysis
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = [np.percentile(df_scores['RiskScore'], p) for p in percentiles]
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        
        bars = axes[1, 1].bar(range(len(percentiles)), percentile_values, 
                             color=colors, alpha=0.7)
        axes[1, 1].set_title('Risk Score Percentiles', fontsize=14)
        axes[1, 1].set_xlabel('Percentile', fontsize=12)
        axes[1, 1].set_ylabel('Risk Score', fontsize=12)
        axes[1, 1].set_xticks(range(len(percentiles)))
        axes[1, 1].set_xticklabels([f'{p}th' for p in percentiles])
        
        # Add value labels on bars
        for bar, value in zip(bars, percentile_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_geographic_analysis(self, df_flagged: pd.DataFrame, 
                                output_dir: str) -> None:
        """
        Plot geographic analysis of fraud patterns.
        
        Args:
            df_flagged: DataFrame with fraud data
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Geographic Fraud Pattern Analysis', fontsize=16, y=0.98)
        
        # 1. Top states by high-risk loans
        if 'BorrowerState' in df_flagged.columns:
            state_counts = df_flagged['BorrowerState'].value_counts().head(15)
            bars = axes[0, 0].bar(range(len(state_counts)), state_counts.values, 
                                 color='steelblue', alpha=0.7)
            axes[0, 0].set_title('Top 15 States by High-Risk Loan Count', fontsize=14)
            axes[0, 0].set_xlabel('State Rank', fontsize=12)
            axes[0, 0].set_ylabel('Number of High-Risk Loans', fontsize=12)
            axes[0, 0].set_xticks(range(0, len(state_counts), 3))
            axes[0, 0].set_xticklabels([state_counts.index[i] for i in range(0, len(state_counts), 3)])
            
            # Add value labels on top bars
            for i, (bar, value) in enumerate(zip(bars[:5], state_counts.values[:5])):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                               f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Risk score by state (top 10)
        if 'BorrowerState' in df_flagged.columns:
            top_states = df_flagged['BorrowerState'].value_counts().head(10).index
            state_risk_data = []
            
            for state in top_states:
                state_data = df_flagged[df_flagged['BorrowerState'] == state]['RiskScore']
                state_risk_data.append(state_data)
            
            bp = axes[0, 1].boxplot(state_risk_data, patch_artist=True, labels=top_states)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
            
            axes[0, 1].set_title('Risk Score Distribution by Top 10 States', fontsize=14)
            axes[0, 1].set_xlabel('State', fontsize=12)
            axes[0, 1].set_ylabel('Risk Score', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. ZIP code analysis
        if 'BorrowerZip' in df_flagged.columns:
            zip_counts = df_flagged['BorrowerZip'].str[:5].value_counts().head(20)
            
            bars = axes[1, 0].bar(range(len(zip_counts)), zip_counts.values,
                                 color='mediumseagreen', alpha=0.7)
            axes[1, 0].set_title('Top 20 ZIP Codes by High-Risk Loan Count', fontsize=14)
            axes[1, 0].set_xlabel('ZIP Code Rank', fontsize=12)
            axes[1, 0].set_ylabel('Number of High-Risk Loans', fontsize=12)
            
            # Add some ZIP code labels
            for i in range(0, min(len(zip_counts), 20), 4):
                axes[1, 0].text(i, zip_counts.iloc[i] + 2, zip_counts.index[i], 
                               ha='center', va='bottom', rotation=45, fontsize=8)
        
        # 4. Geographic concentration analysis
        if 'BorrowerState' in df_flagged.columns:
            state_stats = df_flagged.groupby('BorrowerState').agg({
                'RiskScore': ['mean', 'count']
            }).round(2)
            state_stats.columns = ['avg_risk', 'loan_count']
            state_stats = state_stats[state_stats['loan_count'] >= 10]  # Filter for significant counts
            
            scatter = axes[1, 1].scatter(state_stats['loan_count'], state_stats['avg_risk'],
                                       s=100, alpha=0.6, c='orange', edgecolors='black')
            axes[1, 1].set_title('Average Risk Score vs Loan Count by State', fontsize=14)
            axes[1, 1].set_xlabel('Number of Loans', fontsize=12)
            axes[1, 1].set_ylabel('Average Risk Score', fontsize=12)
            axes[1, 1].grid(True, linestyle='--', alpha=0.3)
            
            # Annotate high-risk, high-volume states
            for state, row in state_stats.iterrows():
                if row['avg_risk'] > state_stats['avg_risk'].quantile(0.8) and \
                   row['loan_count'] > state_stats['loan_count'].quantile(0.8):
                    axes[1, 1].annotate(state, (row['loan_count'], row['avg_risk']),
                                       xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/geographic_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_analysis(self, df_flagged: pd.DataFrame, 
                              output_dir: str) -> None:
        """
        Plot temporal analysis of fraud patterns.
        
        Args:
            df_flagged: DataFrame with fraud data
            output_dir: Output directory
        """
        if 'DateApproved' not in df_flagged.columns:
            logger.warning("DateApproved column not found, skipping temporal analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Fraud Pattern Analysis', fontsize=16, y=0.98)
        
        try:
            # Convert date column
            df_flagged['DateApproved'] = pd.to_datetime(df_flagged['DateApproved'], errors='coerce')
            df_flagged = df_flagged.dropna(subset=['DateApproved'])
            
            # 1. Monthly trend
            df_flagged['Month'] = df_flagged['DateApproved'].dt.to_period('M')
            monthly = df_flagged.groupby('Month').agg({
                'RiskScore': ['mean', 'count']
            })
            monthly.columns = ['avg_risk', 'loan_count']
            
            # Plot loan count
            ax1 = axes[0, 0]
            monthly['loan_count'].plot(kind='line', ax=ax1, marker='o', linewidth=2, 
                                     markersize=6, color='steelblue')
            ax1.set_title('Monthly Trend of High-Risk Loan Approvals', fontsize=14)
            ax1.set_xlabel('Approval Month', fontsize=12)
            ax1.set_ylabel('Number of High-Risk Loans', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Average risk score by month
            monthly['avg_risk'].plot(kind='line', ax=axes[0, 1], marker='s', 
                                   linewidth=2, markersize=6, color='darkred')
            axes[0, 1].set_title('Average Risk Score by Month', fontsize=14)
            axes[0, 1].set_xlabel('Approval Month', fontsize=12)
            axes[0, 1].set_ylabel('Average Risk Score', fontsize=12)
            axes[0, 1].grid(True, linestyle='--', alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Day of week analysis
            df_flagged['DayOfWeek'] = df_flagged['DateApproved'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df_flagged['DayOfWeek'].value_counts().reindex(day_order)
            
            bars = axes[1, 0].bar(day_counts.index, day_counts.values, 
                                 color=['lightblue' if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] 
                                       else 'lightcoral' for day in day_counts.index],
                                 alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('High-Risk Loans by Day of Week', fontsize=14)
            axes[1, 0].set_xlabel('Day of Week', fontsize=12)
            axes[1, 0].set_ylabel('Number of High-Risk Loans', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 5,
                               f'{int(height):,}', ha='center', va='bottom')
            
            # 4. Hourly pattern (if time data available)
            if df_flagged['DateApproved'].dt.hour.nunique() > 1:
                hourly = df_flagged.groupby(df_flagged['DateApproved'].dt.hour).size()
                hourly.plot(kind='bar', ax=axes[1, 1], color='mediumorchid', alpha=0.7)
                axes[1, 1].set_title('High-Risk Loans by Hour of Day', fontsize=14)
                axes[1, 1].set_xlabel('Hour', fontsize=12)
                axes[1, 1].set_ylabel('Number of High-Risk Loans', fontsize=12)
                axes[1, 1].tick_params(axis='x', rotation=0)
            else:
                # If no hour data, show quarterly analysis
                df_flagged['Quarter'] = df_flagged['DateApproved'].dt.quarter
                quarterly = df_flagged.groupby('Quarter').size()
                quarterly.plot(kind='bar', ax=axes[1, 1], color='goldenrod', alpha=0.7)
                axes[1, 1].set_title('High-Risk Loans by Quarter', fontsize=14)
                axes[1, 1].set_xlabel('Quarter', fontsize=12)
                axes[1, 1].set_ylabel('Number of High-Risk Loans', fontsize=12)
                axes[1, 1].tick_params(axis='x', rotation=0)
            
        except Exception as e:
            logger.warning(f"Error in temporal analysis: {str(e)}")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_financial_analysis(self, df_flagged: pd.DataFrame, 
                               output_dir: str) -> None:
        """
        Plot financial analysis of fraud patterns.
        
        Args:
            df_flagged: DataFrame with fraud data
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial Pattern Analysis in Fraud Detection', fontsize=16, y=0.98)
        
        # 1. Risk score vs loan amount
        if 'InitialApprovalAmount' in df_flagged.columns:
            amount = pd.to_numeric(df_flagged['InitialApprovalAmount'], errors='coerce')
            scatter = axes[0, 0].scatter(amount, df_flagged['RiskScore'], 
                                       alpha=0.6, s=30, c='coral', edgecolors='black', linewidth=0.5)
            axes[0, 0].set_title('Risk Score vs. Initial Approval Amount', fontsize=14)
            axes[0, 0].set_xlabel('Initial Approval Amount ($)', fontsize=12)
            axes[0, 0].set_ylabel('Risk Score', fontsize=12)
            axes[0, 0].grid(True, linestyle='--', alpha=0.3)
            axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Loan amount distribution for high-risk loans
        if 'InitialApprovalAmount' in df_flagged.columns:
            amount = pd.to_numeric(df_flagged['InitialApprovalAmount'], errors='coerce').dropna()
            
            # Create risk categories
            high_risk = df_flagged[df_flagged['RiskScore'] > df_flagged['RiskScore'].quantile(0.9)]
            high_risk_amounts = pd.to_numeric(high_risk['InitialApprovalAmount'], errors='coerce').dropna()
            
            axes[0, 1].hist([amount, high_risk_amounts], bins=30, alpha=0.7, 
                           label=['All Loans', 'High Risk (>90th percentile)'],
                           color=['lightblue', 'red'], edgecolor='black')
            axes[0, 1].set_title('Loan Amount Distribution', fontsize=14)
            axes[0, 1].set_xlabel('Initial Approval Amount ($)', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].legend()
            axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 3. Jobs reported vs risk score
        if 'JobsReported' in df_flagged.columns:
            jobs = pd.to_numeric(df_flagged['JobsReported'], errors='coerce')
            
            # Filter out extreme outliers for better visualization
            jobs_filtered = jobs[jobs <= jobs.quantile(0.95)]
            risk_filtered = df_flagged.loc[jobs_filtered.index, 'RiskScore']
            
            scatter = axes[1, 0].scatter(jobs_filtered, risk_filtered, 
                                       alpha=0.6, s=30, c='mediumseagreen', 
                                       edgecolors='black', linewidth=0.5)
            axes[1, 0].set_title('Risk Score vs. Jobs Reported', fontsize=14)
            axes[1, 0].set_xlabel('Jobs Reported', fontsize=12)
            axes[1, 0].set_ylabel('Risk Score', fontsize=12)
            axes[1, 0].grid(True, linestyle='--', alpha=0.3)
        
        # 4. Amount per employee analysis
        if 'InitialApprovalAmount' in df_flagged.columns and 'JobsReported' in df_flagged.columns:
            amount = pd.to_numeric(df_flagged['InitialApprovalAmount'], errors='coerce')
            jobs = pd.to_numeric(df_flagged['JobsReported'], errors='coerce')
            
            # Calculate amount per employee (avoid division by zero)
            amount_per_emp = amount / (jobs.replace(0, np.nan))
            amount_per_emp = amount_per_emp.dropna()
            
            # Filter extreme outliers
            amount_per_emp_filtered = amount_per_emp[amount_per_emp <= amount_per_emp.quantile(0.95)]
            risk_filtered = df_flagged.loc[amount_per_emp_filtered.index, 'RiskScore']
            
            scatter = axes[1, 1].scatter(amount_per_emp_filtered, risk_filtered,
                                       alpha=0.6, s=30, c='gold', 
                                       edgecolors='black', linewidth=0.5)
            axes[1, 1].set_title('Risk Score vs. Amount per Employee', fontsize=14)
            axes[1, 1].set_xlabel('Amount per Employee ($)', fontsize=12)
            axes[1, 1].set_ylabel('Risk Score', fontsize=12)
            axes[1, 1].grid(True, linestyle='--', alpha=0.3)
            axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/financial_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_business_analysis(self, df_flagged: pd.DataFrame, 
                              output_dir: str) -> None:
        """
        Plot business pattern analysis.
        
        Args:
            df_flagged: DataFrame with fraud data
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Business Pattern Analysis in Fraud Detection', fontsize=16, y=0.98)
        
        # 1. Business type analysis (if available)
        if 'BusinessType' in df_flagged.columns:
            business_risk = df_flagged.groupby('BusinessType').agg({
                'RiskScore': ['mean', 'count']
            }).round(2)
            business_risk.columns = ['avg_risk', 'count']
            business_risk = business_risk[business_risk['count'] >= 10]
            business_risk = business_risk.sort_values('avg_risk', ascending=False).head(10)
            
            bars = axes[0, 0].bar(range(len(business_risk)), business_risk['avg_risk'],
                                 color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Average Risk Score by Business Type (Top 10)', fontsize=14)
            axes[0, 0].set_xlabel('Business Type', fontsize=12)
            axes[0, 0].set_ylabel('Average Risk Score', fontsize=12)
            axes[0, 0].set_xticks(range(len(business_risk)))
            axes[0, 0].set_xticklabels(business_risk.index, rotation=45, ha='right')
        else:
            # If no business type, analyze business name patterns
            if 'BorrowerName' in df_flagged.columns:
                name_stats = pd.DataFrame()
                names = df_flagged['BorrowerName'].fillna('').str.lower()
                
                name_stats['has_llc'] = names.str.contains('llc', na=False)
                name_stats['has_inc'] = names.str.contains('inc', na=False)
                name_stats['has_corp'] = names.str.contains('corp', na=False)
                name_stats['has_numbers'] = names.str.contains(r'\d', na=False)
                name_stats['RiskScore'] = df_flagged['RiskScore']
                
                patterns = ['has_llc', 'has_inc', 'has_corp', 'has_numbers']
                pattern_risks = []
                
                for pattern in patterns:
                    avg_risk = name_stats[name_stats[pattern]]['RiskScore'].mean()
                    pattern_risks.append(avg_risk)
                
                bars = axes[0, 0].bar(patterns, pattern_risks, 
                                     color=['red', 'orange', 'yellow', 'green'], 
                                     alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Average Risk Score by Name Pattern', fontsize=14)
                axes[0, 0].set_xlabel('Name Pattern', fontsize=12)
                axes[0, 0].set_ylabel('Average Risk Score', fontsize=12)
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Employee count distribution
        if 'JobsReported' in df_flagged.columns:
            jobs = pd.to_numeric(df_flagged['JobsReported'], errors='coerce')
            
            # Create employee count categories
            job_categories = pd.cut(jobs, bins=[0, 1, 2, 5, 10, 25, 50, float('inf')],
                                   labels=['1', '2', '3-5', '6-10', '11-25', '26-50', '50+'])
            
            job_risk = df_flagged.groupby(job_categories).agg({
                'RiskScore': 'mean'
            }).dropna()
            
            bars = axes[0, 1].bar(range(len(job_risk)), job_risk['RiskScore'],
                                 color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Average Risk Score by Employee Count', fontsize=14)
            axes[0, 1].set_xlabel('Number of Employees', fontsize=12)
            axes[0, 1].set_ylabel('Average Risk Score', fontsize=12)
            axes[0, 1].set_xticks(range(len(job_risk)))
            axes[0, 1].set_xticklabels(job_risk.index)
        
        # 3. Address pattern analysis
        if 'BorrowerAddress' in df_flagged.columns:
            addr_stats = pd.DataFrame()
            addresses = df_flagged['BorrowerAddress'].fillna('').str.lower()
            
            addr_stats['has_apt'] = addresses.str.contains('apt|unit|#', na=False)
            addr_stats['has_po_box'] = addresses.str.contains('po box|p.o.', na=False)
            addr_stats['has_suite'] = addresses.str.contains('suite', na=False)
            addr_stats['is_residential'] = addresses.str.contains('apt|unit|#|residence|house', na=False)
            addr_stats['RiskScore'] = df_flagged['RiskScore']
            
            patterns = ['has_apt', 'has_po_box', 'has_suite', 'is_residential']
            pattern_risks = []
            
            for pattern in patterns:
                avg_risk = addr_stats[addr_stats[pattern]]['RiskScore'].mean()
                pattern_risks.append(avg_risk)
            
            bars = axes[1, 0].bar(patterns, pattern_risks,
                                 color=['purple', 'orange', 'green', 'red'],
                                 alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Average Risk Score by Address Pattern', fontsize=14)
            axes[1, 0].set_xlabel('Address Pattern', fontsize=12)
            axes[1, 0].set_ylabel('Average Risk Score', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top high-risk characteristics summary
        high_risk_threshold = df_flagged['RiskScore'].quantile(0.95)
        high_risk_loans = df_flagged[df_flagged['RiskScore'] > high_risk_threshold]
        
        characteristics = []
        values = []
        
        if 'JobsReported' in df_flagged.columns:
            single_emp_pct = (pd.to_numeric(high_risk_loans['JobsReported'], errors='coerce') == 1).mean() * 100
            characteristics.append('Single Employee')
            values.append(single_emp_pct)
        
        if 'InitialApprovalAmount' in df_flagged.columns:
            high_amount_pct = (pd.to_numeric(high_risk_loans['InitialApprovalAmount'], errors='coerce') > 15000).mean() * 100
            characteristics.append('Amount >$15K')
            values.append(high_amount_pct)
        
        if 'BorrowerAddress' in df_flagged.columns:
            residential_pct = high_risk_loans['BorrowerAddress'].fillna('').str.lower().str.contains('apt|unit|#|house', na=False).mean() * 100
            characteristics.append('Residential Address')
            values.append(residential_pct)
        
        if 'BorrowerName' in df_flagged.columns:
            has_numbers_pct = high_risk_loans['BorrowerName'].fillna('').str.contains(r'\d', na=False).mean() * 100
            characteristics.append('Name Has Numbers')
            values.append(has_numbers_pct)
        
        if characteristics:
            bars = axes[1, 1].bar(characteristics, values,
                                 color=['red', 'orange', 'yellow', 'lightblue'],
                                 alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(f'High-Risk Loan Characteristics\n(Top 5% Risk Score)', fontsize=14)
            axes[1, 1].set_xlabel('Characteristic', fontsize=12)
            axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/business_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_segmentation(self, df_scores: pd.DataFrame, 
                              output_dir: str) -> None:
        """
        Plot risk segmentation analysis.
        
        Args:
            df_scores: DataFrame with risk scores
            output_dir: Output directory
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Risk Segmentation Analysis', fontsize=16, y=0.98)
        
        # Define risk categories
        risk_scores = df_scores['RiskScore']
        
        # 1. Risk categories
        risk_categories = pd.cut(risk_scores, 
                               bins=[0, 25, 50, 75, 100, float('inf')],
                               labels=['Low (0-25)', 'Medium (26-50)', 'High (51-75)', 
                                     'Very High (76-100)', 'Extreme (>100)'])
        
        category_counts = risk_categories.value_counts()
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        
        wedges, texts, autotexts = axes[0, 0].pie(category_counts.values, 
                                                 labels=category_counts.index,
                                                 colors=colors, autopct='%1.1f%%',
                                                 startangle=90)
        axes[0, 0].set_title('Distribution by Risk Category', fontsize=14)
        
        # 2. Cumulative risk distribution
        percentiles = np.arange(0, 101, 5)
        percentile_values = [np.percentile(risk_scores, p) for p in percentiles]
        
        axes[0, 1].plot(percentiles, percentile_values, marker='o', linewidth=2, 
                       markersize=4, color='darkblue')
        axes[0, 1].fill_between(percentiles, percentile_values, alpha=0.3, color='lightblue')
        axes[0, 1].set_title('Risk Score by Percentile', fontsize=14)
        axes[0, 1].set_xlabel('Percentile', fontsize=12)
        axes[0, 1].set_ylabel('Risk Score', fontsize=12)
        axes[0, 1].grid(True, linestyle='--', alpha=0.3)
        
        # 3. Risk concentration
        top_percentages = [1, 5, 10, 20, 50]
        concentration_data = []
        
        for pct in top_percentages:
            threshold = np.percentile(risk_scores, 100 - pct)
            high_risk_loans = risk_scores[risk_scores >= threshold]
            concentration = (high_risk_loans.sum() / risk_scores.sum()) * 100
            concentration_data.append(concentration)
        
        bars = axes[1, 0].bar([f'Top {p}%' for p in top_percentages], concentration_data,
                             color='crimson', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Risk Concentration Analysis', fontsize=14)
        axes[1, 0].set_xlabel('Top Percentage of Loans', fontsize=12)
        axes[1, 0].set_ylabel('% of Total Risk Score', fontsize=12)
        
        # Add value labels
        for bar, value in zip(bars, concentration_data):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Risk score summary statistics
        stats_data = {
            'Mean': risk_scores.mean(),
            'Median': risk_scores.median(),
            'Std Dev': risk_scores.std(),
            '95th Percentile': np.percentile(risk_scores, 95),
            '99th Percentile': np.percentile(risk_scores, 99),
            'Max': risk_scores.max()
        }
        
        y_pos = np.arange(len(stats_data))
        bars = axes[1, 1].barh(y_pos, list(stats_data.values()), 
                              color='mediumslateblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(list(stats_data.keys()))
        axes[1, 1].set_title('Risk Score Summary Statistics', fontsize=14)
        axes[1, 1].set_xlabel('Value', fontsize=12)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, stats_data.values())):
            axes[1, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                           f'{value:.1f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_segmentation.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Command-line interface for fraud analytics"""
    parser = argparse.ArgumentParser(description='Fraud Detection Analytics')
    parser.add_argument('--scores', required=True, help='Fraud scores CSV file')
    parser.add_argument('--data', help='Processed data CSV file (optional)')
    parser.add_argument('--output', default='reports', help='Output directory for plots')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize analyzer
        analyzer = FraudAnalyzer()
        
        # Create analytics
        analyzer.create_fraud_analytics(
            scores_file=args.scores,
            processed_file=args.data or args.scores,
            output_dir=args.output
        )
        
        logger.info("Fraud analytics completed successfully!")
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())