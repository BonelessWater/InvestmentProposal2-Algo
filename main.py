import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import statsmodels.api as sm
from scipy import stats
import backtest
import data_loader

# Function to calculate Sortino ratio
def calculate_sortino_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sortino ratio with annualization (252 trading days)
    """
    excess_returns = returns - risk_free_rate
    downside_returns = np.minimum(returns, 0)
    downside_deviation = np.std(downside_returns) * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.nan
    
    return (np.mean(excess_returns) * 252) / downside_deviation

# Function to calculate Calmar ratio
def calculate_calmar_ratio(returns, max_drawdown):
    """
    Calculate the Calmar ratio (annualized return / maximum drawdown)
    """
    annual_return = np.mean(returns) * 252
    
    # Avoid division by zero (max_drawdown is typically negative)
    if max_drawdown == 0:
        return np.nan
    
    return -annual_return / max_drawdown  # Negative because max_drawdown is negative

# Compute extended backtest metrics including Sortino and Calmar ratios
def compute_extended_backtest_metrics(df):
    """
    Compute extended backtest metrics including Sharpe, Sortino, Calmar ratios, R², and max drawdown.
    """
    df = df.sort_values("date").copy()
    df["daily_return"] = df["portfolio"].pct_change()
    daily_returns = df["daily_return"].dropna()
    

    # Daily risk free rate
    risk_free_rate = 0.045 / 252 
    print(daily_returns.mean(), daily_returns.std())

    # Sharpe ratio
    sharpe = ((daily_returns.mean() - risk_free_rate) / daily_returns.std()) * np.sqrt(252)

    # Sortino ratio
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate)
    
    # Maximum drawdown
    cum_max = df["portfolio"].cummax()
    drawdown = (df["portfolio"] - cum_max) / cum_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar = calculate_calmar_ratio(daily_returns, max_drawdown)
    
    # R-squared
    log_portfolio = np.log(df["portfolio"])
    time_index = np.arange(len(log_portfolio))
    X_time = sm.add_constant(time_index)
    model = sm.OLS(log_portfolio, X_time).fit()
    r2_value = model.rsquared
    
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "r2": r2_value,
        "max_drawdown": max_drawdown
    }

def run_models(X_train, y_train, X_test, y_test, n_splits=5, use_polynomial=False, random_state=37):
    """
    Run models using time-series cross-validation on the training set, 
    then predict on the test set. Uses the provided random_state for reproducibility.
    
    Additionally, computes and prints the R² for the MLR model, ANN model,
    and the inverse-error weighted combined model.
    
    Parameters:
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series or array-like
            Training target values.
        X_test : pandas.DataFrame
            Test features.
        y_test : pandas.Series or array-like
            True target values for the test set, used for computing R².
        n_splits : int, optional (default=5)
            Number of splits to use in the TimeSeriesSplit cross-validation.
        use_polynomial : bool, optional (default=False)
            Whether to include polynomial features.
        random_state : int, optional (default=37)
            Random state for reproducibility.
            
    Returns:
        cv_preds : pandas.DataFrame
            Cross-validation predictions (binary signals) on the training set.
        test_preds : pandas.DataFrame
            Test set predictions (continuous values) for each model.
        mlr : Pipeline object
            Trained Ridge regression pipeline.
        ann : Pipeline object
            Trained MLPRegressor pipeline.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    steps_mlr = []
    steps_ann = []
    if use_polynomial:
        steps_mlr.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps_ann.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    
    steps_mlr.extend([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=random_state))
    ])
    mlr = Pipeline(steps_mlr)
    
    steps_ann.extend([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(50, 30, 30, 10),
            activation="tanh",
            max_iter=1000,
            random_state=random_state,
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
        
        # Convert regression outputs to binary signal for CV predictions (if needed)
        # Here we only store the continuous prediction if you later want to evaluate R² on it.
        cv_preds.loc[X_cv_val.index, "mlr"] = y_mlr > 0.5
        cv_preds.loc[X_cv_val.index, "ann"] = y_mlr > 0.5
        cv_preds.loc[X_cv_val.index, "combo"] = y_combo > 0.5
    
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
    
    # Convert continuous predictions to binary
    y_mlr_binary = (y_mlr_test > 0.5).astype(int)
    y_ann_binary = (y_ann_test > 0.5).astype(int)
    y_combo_binary = (y_combo_test > 0.5).astype(int)

    acc_mlr = accuracy_score(y_test, y_mlr_binary)
    acc_ann = accuracy_score(y_test, y_ann_binary)
    acc_combo = accuracy_score(y_test, y_combo_binary)

    f1_mlr = f1_score(y_test, y_mlr_binary)
    f1_ann = f1_score(y_test, y_ann_binary)
    f1_combo = f1_score(y_test, y_combo_binary)

    roc_auc_mlr = roc_auc_score(y_test, y_mlr_test)
    roc_auc_ann = roc_auc_score(y_test, y_ann_test)
    roc_auc_combo = roc_auc_score(y_test, y_combo_test)

    print(f"MLR Accuracy: {acc_mlr:.4f}, F1: {f1_mlr:.4f}, ROC-AUC: {roc_auc_mlr:.4f}")
    print(f"ANN Accuracy: {acc_ann:.4f}, F1: {f1_ann:.4f}, ROC-AUC: {roc_auc_ann:.4f}")
    print(f"Combined Accuracy: {acc_combo:.4f}, F1: {f1_combo:.4f}, ROC-AUC: {roc_auc_combo:.4f}")

    r_mlr, _ = stats.pearsonr(y_test, y_mlr_test)
    r_ann, _ = stats.pearsonr(y_test, y_ann_test)
    r_combo, _ = stats.pearsonr(y_test, y_combo_test)

    print(f"MLR Correlation^2: {r_mlr**2:.4f}")
    print(f"ANN Correlation^2: {r_ann**2:.4f}")
    print(f"Combined Correlation^2: {r_combo**2:.4f}")

    # Prepare test predictions (continuous output)
    test_preds = pd.DataFrame(index=X_test.index, columns=["mlr", "ann", "combo"])
    test_preds["mlr"] = y_mlr_test
    test_preds["ann"] = y_ann_test
    test_preds["combo"] = y_combo_test
    
    return cv_preds, test_preds, mlr, ann

# Function to run multiple simulations with different random states
def run_multiple_simulations(X_train, y_train, X_test, y_test, test_df, n_runs=30, n_splits=5, use_polynomial=False):
    """
    Run the model multiple times with different random states and collect performance metrics.
    
    Returns:
        DataFrame with performance metrics for each run
        DataFrame with confidence intervals for each metric
    """
    # Create a list to store results from each run
    results = []
    
    # Run the model n_runs times with different random states
    for run in range(n_runs):
        random_state = run + 1  # Use run index + 1 as random state
        print(f"Running simulation {run+1}/{n_runs} with random_state={random_state}")
        
        # Run models with the current random state
        _, test_preds, _, _ = run_models(
            X_train, y_train, X_test, y_test, 
            n_splits=n_splits, 
            use_polynomial=use_polynomial,
            random_state=random_state
        )
        
        # Add the combo signal to the test dataframe
        test_df_run = test_df.sort_values("date").copy()
        test_df_run["combo_signal"] = test_preds["combo"]
        
        # Run backtest
        backtest_df = backtest.backtest_portfolio(test_df_run, test_df_run["combo_signal"], initial_value=1.0)
        
        # Compute metrics
        metrics = compute_extended_backtest_metrics(backtest_df)
        metrics["run"] = run + 1
        metrics["random_state"] = random_state
        metrics["final_value"] = backtest_df["portfolio"].iloc[-1]
        
        results.append(metrics)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate confidence intervals (95%)
    confidence_intervals = {}
    alpha = 0.05  # 95% confidence level
    
    for metric in ["sharpe", "sortino", "calmar", "r2", "max_drawdown", "final_value"]:
        values = results_df[metric].dropna()
        
        if len(values) >= 2:  # Need at least 2 values to calculate confidence interval
            mean = np.mean(values)
            std_err = stats.sem(values)
            conf_interval = stats.t.interval(1-alpha, len(values)-1, loc=mean, scale=std_err)
            
            confidence_intervals[metric] = {
                "mean": mean,
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "lower_bound": conf_interval[0],
                "upper_bound": conf_interval[1]
            }
        else:
            confidence_intervals[metric] = {
                "mean": np.mean(values) if len(values) > 0 else np.nan,
                "std": np.nan,
                "min": np.min(values) if len(values) > 0 else np.nan,
                "max": np.max(values) if len(values) > 0 else np.nan,
                "lower_bound": np.nan,
                "upper_bound": np.nan
            }
    
    # Convert confidence intervals to DataFrame
    ci_df = pd.DataFrame(confidence_intervals).T
    
    return results_df, ci_df

# Function to plot metric distributions
def plot_metric_distributions(results_df):
    """
    Create histograms for each performance metric.
    """
    metrics = ["sharpe", "sortino", "calmar", "r2", "max_drawdown", "final_value"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = results_df[metric].dropna()
        
        ax.hist(data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.4f}')
        
        # Add confidence interval
        mean = np.mean(data)
        std_err = stats.sem(data)
        conf_interval = stats.t.interval(0.95, len(data)-1, loc=mean, scale=std_err)
        
        ax.axvline(x=conf_interval[0], color='green', linestyle=':', linewidth=2, 
                   label=f'95% CI: [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]')
        ax.axvline(x=conf_interval[1], color='green', linestyle=':', linewidth=2)
        
        ax.set_title(f'Distribution of {metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/metric_distributions.png')
    plt.show()

# Main simulation function
def main_simulation():
    """
    Main function to run the full simulation pipeline.
    """
    # 1) Load and clean main SAP data
    sap_df = data_loader.load_and_clean_sap_data("data/SAP_target.csv")
    sap_df = data_loader.engineer_target(sap_df)
    
    # 2) Merge temperature data
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
    
    # 3) Merge linepack data
    sap_df = data_loader.add_linepack_data(
        sap_df,
        closing_lp_path="data/linepack/ClosingLinepackactual.csv",
        opening_lp_path="data/linepack/Openinglinepackactual.csv",
        hourly_agg_d_plus1_path="data/linepack/LinepackHourlyActualAggregateD+1.csv",
        predicted_closing_lp_path="data/linepack/PredictedClosingLinepack(PCLP1).csv"
    )
    
    # 4) Merge demand data
    sap_df = data_loader.add_demand_data(
        sap_df,
        demand_cold_path="data/demand/Demand-Cold.csv",
        demand_warm_path="data/demand/Demand-Warm.csv",
        demand_actual_ntsd_plus1_path="data/demand/DemandActualNTSD+1.csv",
        demand_forecast_confidence_interval_path="data/demand/DemandForecastConfidenceInterval.csv",
        demand_forecast_nts_path="data/demand/DemandForecastNTS.csv",
        demand_ntssn_path="data/demand/DemandNTSSN.csv"
    )
    
    # 5) Add derived demand features
    sap_df = data_loader.add_demand_features(sap_df)
    
    # 6) Add LDC features
    sap_df = data_loader.add_ldc_features(sap_df, t_ref=65)
    
    # 7) Build feature matrix
    X, y, final_df = data_loader.build_feature_matrix(sap_df)
    print(f"Original feature matrix shape: {X.shape}")
    
    # Define best features
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
    
    # Filter features
    X = X[best_features]
    print(f"Filtered feature matrix shape: {X.shape}")
    
    # Split data
    final_df = final_df.sort_values("date").copy()
    train_size = int(len(final_df) * 0.75)
    train_df = final_df.iloc[:train_size]
    test_df = final_df.iloc[train_size:]
    
    X_train = X.loc[train_df.index]
    y_train = y.loc[train_df.index]
    X_test = X.loc[test_df.index]
    y_test = y.loc[test_df.index]
    
    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Run simulations
    print("Starting multiple simulations...")
    results_df, ci_df = run_multiple_simulations(
        X_train, y_train, X_test, y_test, test_df,
        n_runs=2, 
        n_splits=5, 
        use_polynomial=False
    )
    
    # Save results
    results_df.to_csv("results/simulation_results.csv", index=False)
    ci_df.to_csv("results/confidence_intervals.csv")
    
    # Print confidence intervals
    print("\n95% Confidence Intervals:")
    print(ci_df[["mean", "lower_bound", "upper_bound"]])
    
    # Plot distributions
    plot_metric_distributions(results_df)
    
    return results_df, ci_df

if __name__ == "__main__":
    results_df, ci_df = main_simulation()