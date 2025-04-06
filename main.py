import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
import os
import statsmodels.api as sm
import data_loader
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from utils import backtest

# ---------- Modeling Functions ----------
def run_models(X_train, y_train, X_test, n_splits=5, use_polynomial=False):
    """
    Run models using time-series cross-validation on the training set, 
    then predict on the test set.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    steps_mlr = []
    steps_ann = []
    if use_polynomial:
        steps_mlr.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps_ann.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    
    steps_mlr.extend([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    mlr = Pipeline(steps_mlr)
    
    steps_ann.extend([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(50, 30),
            activation="tanh",
            max_iter=1000,
            random_state=42,
            solver="adam",
            learning_rate_init=0.07,
            alpha=0.1
        ))
    ])
    ann = Pipeline(steps_ann)
    
    # CV validation on training set
    cv_preds = pd.DataFrame(index=y_train.index, columns=["mlr", "ann", "combo"])
    
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        mlr.fit(X_cv_train, y_cv_train)
        ann.fit(X_cv_train, y_cv_train)
        
        y_mlr = mlr.predict(X_cv_val)
        y_ann = ann.predict(X_cv_val)
        
        # Inverse-error weighting based on training errors
        err_mlr = np.mean(np.abs(y_cv_train - mlr.predict(X_cv_train)))
        err_ann = np.mean(np.abs(y_cv_train - ann.predict(X_cv_train)))
        w_mlr = 1 / err_mlr
        w_ann = 1 / err_ann
        w_sum = w_mlr + w_ann
        
        y_combo = (w_mlr * y_mlr + w_ann * y_ann) / w_sum
        
        # Convert regression outputs to binary signal: buy=1 if prediction > 0.5.
        cv_preds.loc[X_cv_val.index, "mlr"] = (y_mlr > 0.5).astype(int)
        cv_preds.loc[X_cv_val.index, "ann"] = (y_ann > 0.5).astype(int)
        cv_preds.loc[X_cv_val.index, "combo"] = (y_combo > 0.5).astype(int)
    
    # Train on full training set and predict on test set
    mlr.fit(X_train, y_train)
    ann.fit(X_train, y_train)
    
    y_mlr_test = mlr.predict(X_test)
    y_ann_test = ann.predict(X_test)
    
    # Inverse-error weighting based on training errors
    err_mlr = np.mean(np.abs(y_train - mlr.predict(X_train)))
    err_ann = np.mean(np.abs(y_train - ann.predict(X_train)))
    w_mlr = 1 / err_mlr
    w_ann = 1 / err_ann
    w_sum = w_mlr + w_ann
    
    y_combo_test = (w_mlr * y_mlr_test + w_ann * y_ann_test) / w_sum
    
    # Convert regression outputs to binary signal
    test_preds = pd.DataFrame(index=X_test.index, columns=["mlr", "ann", "combo"])
    test_preds["mlr"] = (y_mlr_test > 0.5).astype(int)
    test_preds["ann"] = (y_ann_test > 0.5).astype(int)
    test_preds["combo"] = (y_combo_test > 0.5).astype(int)
    
    return cv_preds, test_preds, mlr, ann

def evaluate_classification(y_true, preds):
    """
    Evaluate classification accuracy only on rows where predictions are available.
    """
    for col in preds.columns:
        mask = preds[col].notna()
        if mask.sum() == 0:
            print(f"{col:7s} No predictions available.")
            continue
        y_pred = preds.loc[mask, col].astype(int)
        y_true_valid = y_true[mask]
        acc = np.mean(y_true_valid == y_pred)
        print(f"{col:7s} Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(pd.crosstab(y_true_valid, y_pred, rownames=["Actual"], colnames=["Predicted"]))
        print()

def backtest_portfolio(df, signals, initial_value=1.0):
    """
    Simulate a backtest with CORRECTED LOGIC:
      - Use today's signal to decide whether to invest in tomorrow's return.
    Returns a DataFrame with a 'portfolio' column tracking portfolio value over time.
    """
    df = df.sort_values("date").copy()
    signals = signals.sort_index()
    portfolio = [initial_value]
    
    # For each day except the last, use today's signal to decide whether to
    # invest in tomorrow's return.
    for i in range(len(df) - 1):
        if signals.iloc[i] == 1:  # Today's signal
            ret = df["target"].iloc[i]  # Tomorrow's return
            new_value = portfolio[-1] * (1 + ret)
        else:
            new_value = portfolio[-1]
        portfolio.append(new_value)
    
    df["portfolio"] = portfolio
    return df

def save_and_load_results(df, rel_path):
    """
    Save the DataFrame 'df' to CSV using the given relative path,
    then load the CSV back into a DataFrame.
    """
    directory = os.path.dirname(rel_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    df.to_csv(rel_path, index=False)
    loaded_df = pd.read_csv(rel_path, parse_dates=["date"])
    return loaded_df

def compute_backtest_metrics(df):
    """
    Compute backtest metrics from a DataFrame that contains a 'date' column and a 'portfolio' column.
    
    Returns:
      sharpe: Annualized Sharpe ratio (assumes 252 trading days)
      r2_value: R² of the regression of log(portfolio) vs. time index
      max_drawdown: Maximum drawdown (as a negative fraction)
    """
    df = df.sort_values("date").copy()
    df["daily_return"] = df["portfolio"].pct_change()
    daily_returns = df["daily_return"].dropna()
    
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    cum_max = df["portfolio"].cummax()
    drawdown = (df["portfolio"] - cum_max) / cum_max
    max_drawdown = drawdown.min()
    
    log_portfolio = np.log(df["portfolio"])
    time_index = np.arange(len(log_portfolio))
    X_time = sm.add_constant(time_index)
    model = sm.OLS(log_portfolio, X_time).fit()
    r2_value = model.rsquared
    
    return sharpe, r2_value, max_drawdown

def plot_profit_over_time(df, initial_value=1.0):
    """
    Plot the profit over time, where profit is defined as portfolio value minus the initial investment.
    """
    df = df.sort_values("date").copy()
    df["profit"] = df["portfolio"] - initial_value
    
    plt.figure(figsize=(12,6))
    plt.plot(df["date"], df["profit"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Profit")
    plt.title("Profit Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_feature_importance(X, y, n_splits=5, use_polynomial=False):
    """
    Iteratively drop each feature and compare the cross-validated accuracy (using the 'combo'
    signal) of the model with and without that feature.
    
    Returns a list of features that, when dropped, decrease accuracy compared to using all features.
    (i.e. these features have a positive impact on predictions)
    """
    from sklearn.metrics import accuracy_score

    # Evaluate baseline CV accuracy using the full feature set.
    cv_preds_full, _, _, _ = run_models(X, y, X, n_splits=n_splits, use_polynomial=use_polynomial)
    # We use the 'combo' predictions; ensure we compare the binary outputs.
    mask_full = cv_preds_full["combo"].notna()
    baseline_accuracy = np.mean((y[mask_full].astype(int) == cv_preds_full.loc[mask_full, "combo"].astype(int)))
    print(f"Baseline CV Accuracy (combo): {baseline_accuracy:.4f}")

    positive_features = []
    
    # Iterate over each feature in the dataset.
    for feature in X.columns:
        X_reduced = X.drop(columns=[feature])
        cv_preds_reduced, _, _, _ = run_models(X_reduced, y, X_reduced, n_splits=n_splits, use_polynomial=use_polynomial)
        mask_red = cv_preds_reduced["combo"].notna()
        reduced_accuracy = np.mean((y[mask_red].astype(int) == cv_preds_reduced.loc[mask_red, "combo"].astype(int)))
        print(f"Dropping {feature:30s} => CV Accuracy (combo): {reduced_accuracy:.4f}")
        
        # If the model performs worse without the feature, then the feature has a positive impact.
        if baseline_accuracy > reduced_accuracy:
            positive_features.append(feature)
    
    return positive_features

# ---------- Main Execution ----------
if __name__ == "__main__":
    # 1) Load and clean main SAP data.
    sap_df = data_loader.load_and_clean_sap_data("data/SAP_target.csv")
    sap_df = data_loader.engineer_target(sap_df)
    
    # 2) Merge temperature data.
    sap_df = data_loader.add_temperature_data(
        sap_df,
        fc_d_minus1_path="data/weather/TemperatureForecastD-1.csv",
        fc_d_path="data/weather/TemperatureForecastD.csv",
        ac_d_plus1_path="data/weather/TemperatureActualD+1.csv",
        sn_d_minus1_path="data/weather/TemperatureSeasonalNormalD-1.csv",
        sn_d_path="data/weather/TemperatureSeasonalNormalD.csv",
        sn_d_plus1_path="data/weather/TemperatureSeasonalNormalD+1.csv"
    )
    
    sap_df = data_loader.add_additional_weather_data(
        sap_df,
        forecast_ldz_nw_path="data/weather/CompositeWeatherVariableForecastLDZ(NW).csv",
        forecast_ldz_sc_path="data/weather/CompositeWeatherVariableForecastLDZ(SC).csv",
        forecast_ldz_se_path="data/weather/CompositeWeatherVariableForecastLDZ(SE).csv",
        correction_factor_nw_path="data/weather/WeatherCorrectionFactorForecast(NW).csv",
        correction_factor_sc_path="data/weather/WeatherCorrectionFactorForecast(SC).csv",
        correction_factor_se_path="data/weather/WeatherCorrectionFactorForecast(SE).csv"
    )
    
    # 3) Merge linepack data.
    sap_df = data_loader.add_linepack_data(
        sap_df,
        closing_lp_path="data/linepack/ClosingLinepackactual.csv",
        opening_lp_path="data/linepack/Openinglinepackactual.csv",
        hourly_agg_d_plus1_path="data/linepack/LinepackHourlyActualAggregateD+1.csv",
        predicted_closing_lp_path="data/linepack/PredictedClosingLinepack(PCLP1).csv"
    )
    
    # 4) Merge demand data.
    sap_df = data_loader.add_demand_data(
        sap_df,
        demand_cold_path="data/demand/Demand-Cold.csv",
        demand_warm_path="data/demand/Demand-Warm.csv",
        demand_actual_ntsd_plus1_path="data/demand/DemandActualNTSD+1.csv",
        demand_forecast_confidence_interval_path="data/demand/DemandForecastConfidenceInterval.csv",
        demand_forecast_nts_path="data/demand/DemandForecastNTS.csv",
        demand_ntssn_path="data/demand/DemandNTSSN.csv"
    )
    
    # 5) Add derived demand features.
    sap_df = data_loader.add_demand_features(sap_df)
    
    # 6) Add LDC features (including temperature, linepack, and now demand features).
    sap_df = data_loader.add_ldc_features(sap_df, t_ref=65)
    
    # 7) Build feature matrix; the target is the binary signal (1 = buy, 0 = sell/hold).
    X, y, final_df = data_loader.build_feature_matrix(sap_df)
    print(f"Feature matrix shape: {X.shape}")
    
    # Define the list of best features identified from your analysis.
    best_features = [
        "CDD",
        "demand_lag1",
        "is_weekend",
        "is_holiday",
        "predicted_closing_linepack",
        "lp_closing_pred_error",
        "temp_fc_d_minus1",
        "temp_sn_d_minus1",
        "temp_sn_d",
        "temp_sn_d_plus1",
        "temp_dev_fc_minus1",
        "temp_dev_fc",
        "CompositeWeatherVariableForecastLDZ(SE)",
        "WeatherCorrectionFactorForecast(NW)",
        "WeatherCorrectionFactorForecast(SC)",
        "demand_actual_ntsd_plus1",
        "demand_diff",
        "demand_avg",
        "demand_forecast_error",
        "demand_nts_spread",
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_4",
        "dow_6"
    ]

    # Filter the full feature matrix to keep only the best features.
    X = X[best_features]
    print("Filtered feature matrix shape:", X.shape)

    # Continue with time-based train-test split.
    final_df = final_df.sort_values("date").copy()
    train_size = int(len(final_df) * 0.75)
    train_df = final_df.iloc[:train_size]
    test_df = final_df.iloc[train_size:]

    X_train = X.loc[train_df.index]
    y_train = y.loc[train_df.index]
    X_test = X.loc[test_df.index]
    y_test = y.loc[test_df.index]

    # Run your modeling pipeline with the filtered features.
    cv_preds, test_preds, mlr_model, ann_model = run_models(
        X_train, y_train, X_test, n_splits=5, use_polynomial=False
    )

    print("\nCross-Validation Performance with Filtered Features:")
    evaluate_classification(y_train, cv_preds)

    print("\nTest Set Performance with Filtered Features:")
    evaluate_classification(y_test, test_preds)
    
    # 12) Backtest using the combined signal on test set.
    test_df = test_df.sort_values("date").copy()
    test_df["combo_signal"] = test_preds["combo"]

    backtest_df = backtest.backtest_portfolio(test_df, test_df["combo_signal"], initial_value=1.0)
    
    # 13) Save results and load them back.
    relative_path = "results/backtest_results.csv"
    loaded_backtest_df = save_and_load_results(backtest_df, relative_path)

    print("\nBacktest Results (first 5 rows):")
    print(loaded_backtest_df.head())
    
    sharpe, r2_value, max_drawdown = compute_backtest_metrics(loaded_backtest_df)
    
    # 14) Print the metrics.
    print("\nBacktest Metrics:")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"R²: {r2_value:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    
    # 15) Visualize backtest results.
    plot_profit_over_time(loaded_backtest_df, initial_value=1.0)# Example usage after building your feature matrix:
    
    # (Assuming X, y have already been created using data_loader.build_feature_matrix)
    #positive_feats = evaluate_feature_importance(X, y, n_splits=5, use_polynomial=False)
    #print("\nFeatures with positive outcomes on prediction:")
    #for feat in positive_feats:
    #    print(feat)