# Executive Report: DSN Bootcamp Hackathon – Used Car Price Prediction (Fast Baseline)

## Overview

This report details the methodology, implementation, and results of our submission to the DSN Bootcamp Hackathon, which addresses the challenge of predicting used car prices. The solution is based on the `hackathon_notebook.ipynb` notebook, which is designed as a fast, ready-to-run baseline. The approach prioritizes speed and simplicity, using minimal feature engineering and leveraging CatBoost for its native handling of categorical variables. The entire pipeline is optimized to run in under one hour on typical hardware, making it suitable for rapid prototyping and as a strong starting point for further improvements.

## Problem Statement

The objective is to develop a predictive model that estimates the selling price of used cars based on available features in the provided dataset. Accurate price prediction is essential for both buyers and sellers, enabling fair transactions and informed decision-making in the automotive market. The evaluation metric for the hackathon is Root Mean Squared Error (RMSE) on the predicted prices.

## Data and Submission Format

- **Training Data:** Provided in `train.csv`, containing features and the target variable (`price`).
- **Test Data:** Provided in `test.csv`, for which predictions are to be generated.
- **Submission:** A CSV file with columns `id` and `price`, where `price` is the predicted value for each test sample.

## Environment and Dependencies

The notebook uses the following Python libraries:
- `pandas`, `numpy` for data manipulation
- `scikit-learn` for model evaluation and cross-validation
- `catboost` for modeling
- `optuna` for hyperparameter tuning
- `matplotlib` for visualization

All dependencies are installed at the start of the notebook if not already present.

## Data Loading and Initial Inspection

- The training and test datasets are loaded using pandas.
- The code checks for the existence of `test.csv` to determine if inference and submission steps should be executed.
- Basic data inspection is performed using `head()` and `describe()` to understand the structure, feature types, and basic statistics.

## Preprocessing and Feature Engineering

The preprocessing pipeline is intentionally minimal to ensure fast execution:

- **Column Normalization:** All column names are converted to lowercase and spaces are replaced with underscores for consistency.
- **Missing Columns Handling:** The code ensures that all expected columns are present in the DataFrame, adding any missing ones with default values to prevent errors.
- **Feature: Car Age:** A new feature `car_age` is computed as the difference between a fixed current year (2025) and the `model_year` of the car.
- **Categorical Fills:** For key categorical features (`fuel_type`, `accident`, `clean_title`), missing values are filled with `'Unknown'`.
- **Engine Horsepower Parsing:** The `engine` column is parsed to extract a numeric `engine_hp` value, if present (e.g., from strings like "250 HP").
- **Color Normalization:** Both exterior and interior color columns are mapped into a small set of coarse color bins (black, white, silver, gray, blue, red, green, other) to reduce noise and cardinality.
- **Numeric and Categorical Features:** The pipeline defines sets of numeric (`model_year`, `milage`, `car_age`, `engine_hp`) and categorical features (`brand`, `model`, `fuel_type`, `transmission`, `ext_col_norm`, `int_col_norm`, `accident`, `clean_title`). Numeric columns are coerced to numeric types, and categoricals are ensured to be strings with missing values filled as `'Unknown'`.
- **Final Feature Set:** Only the relevant columns are retained for modeling.

This minimal preprocessing leverages CatBoost's ability to handle categorical features natively, reducing the need for extensive manual encoding or transformation.

## Modeling Approach

### Cross-Validation and Hyperparameter Tuning

- **Target Transformation:** The target variable (`price`) is transformed using `log1p` to reduce skewness and stabilize variance.
- **Stratified K-Fold:** The data is split into 3 folds using `StratifiedKFold`, stratifying on binned target values to ensure balanced splits.
- **Optuna Tuning:** A lightweight Optuna study is run with 5 trials to quickly search over a small hyperparameter space for CatBoost (learning rate, depth, L2 regularization). The search is kept minimal to ensure fast runtime.
- **Model Selection:** CatBoostRegressor is used for its efficiency and strong performance with categorical data. The best parameters from Optuna are used for final training.

### Final Training and Evaluation

- **Out-of-Fold (OOF) Predictions:** The model is retrained on each fold using the best hyperparameters, and OOF predictions are generated to estimate generalization performance.
- **RMSE Calculation:** The OOF predictions are inverse-transformed (`expm1`) and compared to the true prices using RMSE.
- **Feature Importance:** Feature importances are averaged across folds and visualized to identify the most influential features.

## Results

- **Performance:** The final OOF RMSE is reported, providing an unbiased estimate of model performance on unseen data.
- **Feature Insights:** The most important features typically include car age, mileage, engine horsepower, brand, and accident/title status, as determined by CatBoost's feature importance scores.
- **Visualization:** A horizontal bar plot of the top 15 features by importance is generated for interpretability.

## Inference and Submission

- If `test.csv` is available, the trained models are used to generate predictions for the test set.
- Predictions are inverse-transformed (`expm1`) to return to the original price scale and clipped to be non-negative.
- The results are saved in a timestamped CSV file in the required submission format.

## Limitations and Future Improvements

- **Minimal Feature Engineering:** The current pipeline uses only basic features and transformations. More sophisticated feature engineering (e.g., parsing additional engine specs, handling rare categories, or extracting more from text fields) could improve performance.
- **Limited Hyperparameter Search:** The Optuna search is intentionally shallow for speed. A more extensive search or ensembling could yield better results.
- **No Outlier Handling:** The pipeline does not explicitly handle outliers or rare categories, which may affect model robustness.
- **No External Data:** Only the provided data is used; incorporating external sources (e.g., car valuation guides) could further enhance predictions.

## Conclusion

The `hackathon_notebook.ipynb` notebook provides a fast, reliable baseline for used car price prediction in the DSN Bootcamp Hackathon. By focusing on minimal preprocessing, leveraging CatBoost's strengths, and using efficient cross-validation and tuning, the solution achieves strong performance with rapid turnaround. This pipeline serves as a solid foundation for further experimentation and improvement in future iterations.
