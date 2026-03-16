import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) == 0:
        return None
    
    fig, axes = plt.subplots(len(numerical_cols), 1, 
                              figsize=(10, 4 * len(numerical_cols)))
    
    if len(numerical_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], ax=axes[i], kde=True, color='steelblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    
    plt.tight_layout(pad=3.0)
    return fig

def plot_correlation(df):
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    if numerical_df.shape[1] < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=True, fmt='.2f', 
                cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout(pad=3.0)
    return fig

def plot_categorical(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        return None
    
    fig, axes = plt.subplots(len(categorical_cols), 1,
                              figsize=(10, 4 * len(categorical_cols)))
    
    if len(categorical_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts().head(10)
        sns.barplot(x=value_counts.index, y=value_counts.values, 
                    ax=axes[i], color='steelblue')
        axes[i].set_title(f'Top 10 values in {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(pad=3.0)
    return fig

def get_summary_stats(df):
    return df.describe()
