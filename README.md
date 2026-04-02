# PredictFlow

PredictFlow is an end-to-end automated machine learning web application that allows users to upload any CSV or Excel dataset and instantly receive cleaned data, exploratory data analysis, and machine learning model predictions all without writing a single line of code.

## Features

- Automatic data loading for CSV and Excel files
- Data cleaning with missing value imputation and duplicate removal
- Exploratory data analysis with distribution plots, correlation heatmaps, and categorical charts
- Automatic detection of classification vs regression problems
- Training and comparison of multiple ML models
- Feature importance visualization
- Download cleaned dataset

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

    PredictFlow/
    ├── app.py
    ├── utils/
    │   ├── __init__.py
    │   ├── cleaner.py
    │   ├── eda.py
    │   └── model.py
    ├── requirements.txt
    └── README.md

## Getting Started

### Prerequisites

Make sure you have Python 3.8 or above installed on your system.

### Installation

1. Clone the repository

        git clone https://github.com/yourusername/PredictFlow.git

2. Navigate to the project directory

        cd PredictFlow

3. Create a virtual environment

        python3 -m venv venv
        source venv/bin/activate

4. Install dependencies

        pip install -r requirements.txt

5. Run the application

        streamlit run app.py

6. Open your browser and go to http://localhost:8501

## Usage

1. Upload any CSV or Excel dataset using the file uploader
2. View the raw data preview and basic dataset information
3. Review the automatically cleaned data and download it if needed
4. Explore the EDA charts to understand your data
5. Select your target column from the dropdown
6. Click Train Models to train and compare ML models
7. View the best model, its accuracy and feature importance

## Models Used

### Classification Problems
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Regression Problems
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## How It Works

1. The user uploads a CSV or Excel file
2. The app loads and displays the raw data
3. Missing values are filled using median for numerical columns and mode for categorical columns
4. Duplicate rows are removed automatically
5. EDA charts are generated to visualize the data
6. The app automatically detects whether the problem is classification or regression based on the target column
7. Three models are trained on 80% of the data and tested on the remaining 20%
8. The best performing model is highlighted along with its feature importance
