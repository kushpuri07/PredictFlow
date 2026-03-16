import streamlit as st
import pandas as pd
from utils.cleaner import load_data, clean_data, get_basic_info
from utils.eda import plot_distribution, plot_correlation, plot_categorical, get_summary_stats
from utils.model import train_models, plot_feature_importance, plot_model_comparison

# Page config
st.set_page_config(
    page_title="PredictFlow",
    page_icon="",
    layout="wide"
)

# Title
st.title("PredictFlow")
st.subheader("Upload any dataset — get EDA + ML predictions instantly")
st.divider()

# File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", 
                                   type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("Could not load file. Please upload a CSV or Excel file.")
    else:
        st.success("File uploaded successfully!")
        
        # SECTION 1: RAW DATA
        st.header("Raw Data")
        st.dataframe(df.head(10))
        
        info = get_basic_info(df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", info['rows'])
        col2.metric("Columns", info['columns'])
        col3.metric("Missing Values", info['missing_values'])
        col4.metric("Duplicates", info['duplicates'])
        
        st.divider()
        
        # SECTION 2: DATA CLEANING
        st.header("Data Cleaning")
        df_cleaned = clean_data(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before Cleaning")
            st.dataframe(df.head(5))
            st.caption(f"Missing values: {df.isnull().sum().sum()}")
        with col2:
            st.subheader("After Cleaning")
            st.dataframe(df_cleaned.head(5))
            st.caption(f"Missing values: {df_cleaned.isnull().sum().sum()}")

        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Data", csv, 
                           "cleaned_data.csv", "text/csv")
        
        st.divider()
        
        # SECTION 3: EDA
        st.header("Exploratory Data Analysis")
        
        st.subheader("Summary Statistics")
        st.dataframe(get_summary_stats(df_cleaned))
        
        st.subheader("Distributions")
        fig = plot_distribution(df_cleaned)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No numerical columns found for distribution plot.")
        
        st.subheader("Correlation Heatmap")
        fig = plot_correlation(df_cleaned)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numerical columns for correlation heatmap.")
        
        st.subheader("Categorical Columns")
        fig = plot_categorical(df_cleaned)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No categorical columns found.")
        
        st.divider()
        
        # SECTION 4: ML MODEL
        st.header("ML Predictions")
        
        target_col = st.selectbox("Select your target column (what you want to predict):",
                                   df_cleaned.columns.tolist())
        
        if st.button("Train Models"):
            with st.spinner("Training models... please wait"):
                results, trained_models, problem_type, metric, feature_names = train_models(
                    df_cleaned, target_col
                )
            
            st.success(f"Done! Problem type detected: **{problem_type.upper()}**")
            
            st.subheader("Model Comparison")
            fig = plot_model_comparison(results, metric)
            st.pyplot(fig)
            
            best_model_name = max(results, key=results.get)
            best_score = results[best_model_name]
            st.info(f"Best Model: **{best_model_name}** with **{best_score}%** {metric}")
            
            st.subheader("Feature Importance")
            best_model = trained_models[best_model_name]
            fig = plot_feature_importance(best_model, feature_names)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Feature importance not available for this model.")

else:
    st.info("Upload a CSV or Excel file to get started!")
    st.markdown("""
    ### What PredictFlow does:
    - **Loads** your CSV or Excel file
    - **Cleans** missing values and duplicates automatically
    - **Analyzes** your data with beautiful visualizations
    - **Trains** multiple ML models and picks the best one
    - **Shows** feature importance and model comparison
    """)