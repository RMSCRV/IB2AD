"""
Utility functions for ML Pipeline
Author: ML Pipeline Project
Date: 2025-12-02
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def set_plot_style():
    """Set consistent plotting style for all visualizations"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def print_section_header(title: str, char: str = '='):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in a DataFrame

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    ).reset_index(drop=True)

    return missing


def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """
    Visualize missing values in DataFrame

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    figsize : tuple
        Figure size
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found!")
        return

    plt.figure(figsize=figsize)
    missing.plot(kind='bar', color='coral')
    plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_distributions(df: pd.DataFrame, columns: List[str] = None,
                       ncols: int = 3, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot distributions for numerical columns

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of columns to plot (if None, plots all numerical columns)
    ncols : int
        Number of columns in subplot grid
    figsize : tuple
        Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]

    for idx, col in enumerate(columns):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)):
    """
    Plot correlation heatmap for numerical features

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    figsize : tuple
        Figure size
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[pd.Series, int]:
    """
    Detect outliers using IQR method

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name

    Returns:
    --------
    tuple
        Boolean mask of outliers and count
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    return outliers, outliers.sum()


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print comprehensive dataset information

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Dataset name
    """
    print_section_header(f"{name} Information")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing Values: {df.isnull().sum().sum()} total")


def compare_models(results: Dict[str, Dict[str, float]],
                   figsize: Tuple[int, int] = (14, 6)):
    """
    Compare multiple models with bar plots

    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and metrics dict as values
    figsize : tuple
        Figure size
    """
    df_results = pd.DataFrame(results).T

    fig, axes = plt.subplots(1, len(df_results.columns), figsize=figsize)
    if len(df_results.columns) == 1:
        axes = [axes]

    for idx, metric in enumerate(df_results.columns):
        df_results[metric].plot(kind='bar', ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'{metric}', fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_xlabel('Model')
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def save_figure(filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save current matplotlib figure

    Parameters:
    -----------
    filename : str
        Output filename (include path)
    dpi : int
        Resolution
    bbox_inches : str
        Bounding box setting
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved: {filename}")


def create_feature_importance_plot(feature_names: List[str],
                                   importances: np.ndarray,
                                   top_n: int = 10,
                                   figsize: Tuple[int, int] = (10, 6)):
    """
    Plot feature importances

    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : np.ndarray
        Feature importance values
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['Importance'], color='teal')
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("ML Pipeline Utilities Module")
    print("Import this module to use utility functions in your notebook")
