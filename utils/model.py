import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def detect_problem_type(df, target_col):
    if df[target_col].dtype == 'object':
        return 'classification'
    elif df[target_col].nunique() <= 10:
        return 'classification'
    else:
        return 'regression'

def preprocess(df, target_col):
    df = df.copy()
    le = LabelEncoder()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def train_models(df, target_col):
    problem_type = detect_problem_type(df, target_col)
    X, y = preprocess(df, target_col)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        metric = 'Accuracy'
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        metric = 'R2 Score'
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if problem_type == 'classification':
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)
        
        results[name] = round(score * 100, 2)
        trained_models[name] = model
    
    return results, trained_models, problem_type, metric, X.columns.tolist()

def plot_feature_importance(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, 
                color='steelblue', ax=ax)
    ax.set_title('Top 10 Most Important Features')
    plt.tight_layout()
    return fig

def plot_model_comparison(results, metric):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(results.keys())
    scores = list(results.values())
    
    bars = ax.bar(models, scores, color=['steelblue', 'coral', 'mediumseagreen'])
    ax.set_title(f'Model Comparison — {metric}')
    ax.set_ylabel(f'{metric} (%)')
    ax.set_ylim(0, 110)
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig