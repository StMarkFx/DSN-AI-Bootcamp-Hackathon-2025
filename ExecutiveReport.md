# Executive Report: Hackathon Submission – Used Car Price Prediction

## Overview

This report summarizes the approach and results of our hackathon submission, which focused on building a robust machine learning pipeline to predict used car prices. The solution leverages advanced data preprocessing, feature engineering, and state-of-the-art modeling techniques to deliver accurate and reliable price predictions.

## Problem Statement

The objective was to develop a predictive model that estimates the selling price of used cars based on a variety of features, including vehicle specifications, condition, and historical data. Accurate price prediction is crucial for both buyers and sellers in the automotive market, enabling informed decision-making and fair transactions.

## Data Exploration and Analysis

- **Data Loading & Inspection:** The dataset was loaded and examined for structure, missing values, and feature types.
- **Target Variable:** The target variable, `price`, exhibited a right-skewed distribution. A log transformation was applied to normalize the distribution and improve model performance.
- **Exploratory Data Analysis (EDA):** Key statistics, missing value counts, and feature distributions were analyzed. The EDA revealed the need for comprehensive data cleaning and preprocessing, especially for handling missing values, high-cardinality categorical features, and outliers.

## Feature Engineering and Preprocessing

To enhance model performance, several feature engineering steps were implemented:

- **Engine Parsing:** Extracted numerical features such as horsepower, engine size (liters), and cylinder count from textual engine descriptions using regular expressions.
- **Transmission Normalization:** Standardized transmission types (Manual, Automatic, CVT, Other) and extracted the number of gears.
- **Accident and Title Status:** Mapped accident history and title status to standardized categories for better interpretability.
- **Color Normalization:** Grouped exterior and interior colors into common categories to reduce noise.
- **Car Age Calculation:** Derived car age from the model year and the current year.
- **Rare Category Bucketing:** High-cardinality features like brand and model were bucketed, grouping rare categories under "Other" to prevent overfitting.
- **Missing Value Handling:** Numeric features were coerced to appropriate types, and missing values were filled or marked as "Unknown" for categorical features.

## Modeling Approach

- **Model Selection:** The CatBoostRegressor, a gradient boosting algorithm well-suited for categorical data, was chosen for its performance and ease of handling mixed data types.
- **Cross-Validation:** A 5-fold Stratified K-Fold cross-validation strategy was employed, stratifying by price bins to ensure balanced folds and robust performance estimation.
- **Target Transformation:** The target variable was log-transformed to address skewness and stabilize variance.
- **Feature Importance:** Feature importances were computed to identify the most influential variables in price prediction.

## Results

- **Performance Metric:** The model was evaluated using Root Mean Squared Error (RMSE) on out-of-fold predictions, providing an unbiased estimate of generalization performance.
- **Feature Insights:** The most important features included car age, mileage, engine specifications, brand, and accident/title status.
- **Submission:** Final predictions were generated for the test set and saved for submission.

## Conclusion

The developed pipeline demonstrates a comprehensive and systematic approach to used car price prediction, combining thorough data analysis, thoughtful feature engineering, and advanced modeling techniques. The solution is robust, interpretable, and delivers strong predictive performance, making it well-suited for real-world deployment in the automotive industry.
