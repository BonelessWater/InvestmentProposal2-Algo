import pandas as pd
import numpy as np
import pandas.tseries.holiday as hol

# ---------- Main SAP Data Loader and Cleaner ----------
def load_and_clean_sap_data(path):
    """
    Load the main SAP data, parse dates, clean and interpolate close prices.
    """
    df = pd.read_csv(
        path,
        parse_dates=["Applicable At", "Applicable For", "Generated Time"],
        dayfirst=True
    )
    # Use "Applicable For" as date and extract symbol from "Data Item"
    df["date"] = df["Applicable For"]
    df["symbol"] = df["Data Item"].apply(lambda x: x.split(",")[0].strip())
    df["open"] = df["Value"]
    df["high"] = df["Value"]
    df["low"] = df["Value"]
    df["close"] = df["Value"]
    df = df[["date", "symbol", "open", "high", "low", "close"]]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values(by=["date", "symbol"])
    df.loc[df["close"] == 0, "close"] = np.nan
    symbols = df["symbol"].unique()
    for symbol in symbols:
        mask = df["symbol"] == symbol
        sub_df = df.loc[mask].copy().sort_values("date")
        sub_df = sub_df.set_index("date")
        sub_df["close"] = sub_df["close"].interpolate(method="time")
        sub_df = sub_df.reset_index()
        df.loc[mask, "close"] = sub_df["close"]
    return df

def engineer_target(df):
    """
    Compute next-day percent change of close as the target.
    Then create a binary signal: 1 if target > 0 (buy), 0 otherwise.
    
    REMOVED: log_target and target_deriv to prevent data leakage
    """
    symbols = df["symbol"].unique()
    df["target"] = np.nan
    for symbol in symbols:
        mask = df["symbol"] == symbol
        sub_df = df.loc[mask].sort_values("date")
        # Compute percent change of 'close' then shift(-1) for next-day value.
        df.loc[mask, "target"] = sub_df["close"].pct_change().shift(-1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["open", "high", "low", "close", "target"])
    
    # Clip extreme target values (e.g., to the 1st and 99th percentiles)
    q_low = df["target"].quantile(0.01)
    q_high = df["target"].quantile(0.99)
    df["target"] = df["target"].clip(q_low, q_high)
    
    # Create a binary signal for trading: 1 if target > 0, else 0.
    df["signal"] = (df["target"] > 0).astype(int)
    
    return df

# ---------- Temperature Data Loader ----------
def load_temperature_file(csv_path, new_col_name):
    """
    Load a temperature CSV file and rename columns:
      "Applicable For" -> "date", "Value" -> new_col_name.
    """
    tmp = pd.read_csv(csv_path, parse_dates=["Applicable For"], dayfirst=True)
    tmp.rename(columns={"Applicable For": "date", "Value": new_col_name}, inplace=True)
    tmp = tmp[["date", new_col_name]]
    return tmp

def add_temperature_data(df,
                         fc_d_minus1_path="../new_data/weather/TemperatureForecastD-1.csv",
                         fc_d_path="../new_data/weather/TemperatureForecastD.csv",
                         ac_d_plus1_path="../new_data/weather/TemperatureActualD+1.csv",
                         sn_d_minus1_path="../new_data/weather/TemperatureSeasonalNormalD-1.csv",
                         sn_d_path="../new_data/weather/TemperatureSeasonalNormalD.csv",
                         sn_d_plus1_path="../new_data/weather/TemperatureSeasonalNormalD+1.csv"):
    """
    Merge the six temperature CSV files into the main DataFrame.
    """
    df_fc_minus1 = load_temperature_file(fc_d_minus1_path, "temp_fc_d_minus1")
    df_fc = load_temperature_file(fc_d_path, "temp_fc_d")
    df_ac_plus1 = load_temperature_file(ac_d_plus1_path, "temp_ac_d_plus1")
    df_sn_minus1 = load_temperature_file(sn_d_minus1_path, "temp_sn_d_minus1")
    df_sn = load_temperature_file(sn_d_path, "temp_sn_d")
    df_sn_plus1 = load_temperature_file(sn_d_plus1_path, "temp_sn_d_plus1")
    
    df = pd.merge(df, df_fc_minus1, on="date", how="left")
    df = pd.merge(df, df_fc, on="date", how="left")
    df = pd.merge(df, df_ac_plus1, on="date", how="left")
    df = pd.merge(df, df_sn_minus1, on="date", how="left")
    df = pd.merge(df, df_sn, on="date", how="left")
    df = pd.merge(df, df_sn_plus1, on="date", how="left")
    return df

# ---------- Linepack Data Loader ----------
def load_linepack_file(csv_path, date_col="Applicable For", value_col="Value", new_name="linepack"):
    """
    Generic loader for a linepack CSV file.
    """
    lp = pd.read_csv(csv_path, parse_dates=[date_col], dayfirst=True)
    lp.rename(columns={date_col: "date", value_col: new_name}, inplace=True)
    lp = lp[["date", new_name]]
    return lp

def add_linepack_data(df,
                      closing_lp_path="../new_data/linepack/ClosingLinepackactual.csv",
                      opening_lp_path="../new_data/linepack/Openinglinepackactual.csv",
                      hourly_agg_d_plus1_path="../new_data/linepack/LinepackHourlyActualAggregateD+1.csv",
                      predicted_closing_lp_path="../new_data/linepack/PredictedClosingLinepack(PCLP1).csv"):
    """
    Merge linepack CSV files into the main DataFrame.
    """
    closing_lp = load_linepack_file(closing_lp_path, date_col="Applicable For", value_col="Value", new_name="closing_linepack_actual")
    df = pd.merge(df, closing_lp, on="date", how="left")
    
    opening_lp = load_linepack_file(opening_lp_path, date_col="Applicable For", value_col="Value", new_name="opening_linepack_actual")
    df = pd.merge(df, opening_lp, on="date", how="left")
    
    hourly_lp = load_linepack_file(hourly_agg_d_plus1_path, date_col="Applicable For", value_col="Value", new_name="linepack_hourly_agg_d_plus1")
    df = pd.merge(df, hourly_lp, on="date", how="left")
    
    pred_closing_lp = load_linepack_file(predicted_closing_lp_path, date_col="Applicable For", value_col="Value", new_name="predicted_closing_linepack")
    df = pd.merge(df, pred_closing_lp, on="date", how="left")
    
    return df

def add_additional_weather_data(df, 
                                forecast_ldz_nw_path, forecast_ldz_sc_path, forecast_ldz_se_path,
                                correction_factor_nw_path, correction_factor_sc_path, correction_factor_se_path):
    # Load forecast data with correct date parsing and rename columns.
    forecast_ldz_nw = pd.read_csv(forecast_ldz_nw_path, parse_dates=['Applicable For'], dayfirst=True)
    forecast_ldz_nw.rename(columns={
        'Applicable For': 'date',
        'Value': 'CompositeWeatherVariableForecastLDZ(NW)'
    }, inplace=True)

    forecast_ldz_sc = pd.read_csv(forecast_ldz_sc_path, parse_dates=['Applicable For'], dayfirst=True)
    forecast_ldz_sc.rename(columns={
        'Applicable For': 'date',
        'Value': 'CompositeWeatherVariableForecastLDZ(SC)'
    }, inplace=True)

    forecast_ldz_se = pd.read_csv(forecast_ldz_se_path, parse_dates=['Applicable For'], dayfirst=True)
    forecast_ldz_se.rename(columns={
        'Applicable For': 'date',
        'Value': 'CompositeWeatherVariableForecastLDZ(SE)'
    }, inplace=True)

    # Load correction factor data with date parsing and rename columns.
    correction_factor_nw = pd.read_csv(correction_factor_nw_path, parse_dates=['Applicable For'], dayfirst=True)
    correction_factor_nw.rename(columns={
        'Applicable For': 'date',
        'Value': 'WeatherCorrectionFactorForecast(NW)'
    }, inplace=True)

    correction_factor_sc = pd.read_csv(correction_factor_sc_path, parse_dates=['Applicable For'], dayfirst=True)
    correction_factor_sc.rename(columns={
        'Applicable For': 'date',
        'Value': 'WeatherCorrectionFactorForecast(SC)'
    }, inplace=True)

    correction_factor_se = pd.read_csv(correction_factor_se_path, parse_dates=['Applicable For'], dayfirst=True)
    correction_factor_se.rename(columns={
        'Applicable For': 'date',
        'Value': 'WeatherCorrectionFactorForecast(SE)'
    }, inplace=True)

    # Merge each additional weather dataset into the main DataFrame using the "date" column.
    df = df.merge(forecast_ldz_nw[['date', 'CompositeWeatherVariableForecastLDZ(NW)']],
                  on='date', how='left')
    df = df.merge(forecast_ldz_sc[['date', 'CompositeWeatherVariableForecastLDZ(SC)']],
                  on='date', how='left')
    df = df.merge(forecast_ldz_se[['date', 'CompositeWeatherVariableForecastLDZ(SE)']],
                  on='date', how='left')
    df = df.merge(correction_factor_nw[['date', 'WeatherCorrectionFactorForecast(NW)']],
                  on='date', how='left')
    df = df.merge(correction_factor_sc[['date', 'WeatherCorrectionFactorForecast(SC)']],
                  on='date', how='left')
    df = df.merge(correction_factor_se[['date', 'WeatherCorrectionFactorForecast(SE)']],
                  on='date', how='left')
    
    return df

# ---------- Demand Data Loader and Feature Engineering ----------
def load_demand_file(csv_path, new_col_name):
    """
    Load a demand CSV file, parse dates and rename:
      "Applicable For" -> "date", "Value" -> new_col_name.
    """
    df = pd.read_csv(csv_path, parse_dates=["Applicable For"], dayfirst=True)
    df.rename(columns={"Applicable For": "date", "Value": new_col_name}, inplace=True)
    df = df[["date", new_col_name]]
    return df

def add_demand_data(df,
                    demand_cold_path,
                    demand_warm_path,
                    demand_actual_ntsd_plus1_path,
                    demand_forecast_confidence_interval_path,
                    demand_forecast_nts_path,
                    demand_ntssn_path):
    """
    Merge the six demand datasets into the main DataFrame.
    """
    df_cold = load_demand_file(demand_cold_path, "demand_cold")
    df_warm = load_demand_file(demand_warm_path, "demand_warm")
    df_actual = load_demand_file(demand_actual_ntsd_plus1_path, "demand_actual_ntsd_plus1")
    df_conf_int = load_demand_file(demand_forecast_confidence_interval_path, "demand_forecast_confidence_interval")
    df_forecast_nts = load_demand_file(demand_forecast_nts_path, "demand_forecast_nts")
    df_ntssn = load_demand_file(demand_ntssn_path, "demand_ntssn")
    
    df = pd.merge(df, df_cold, on="date", how="left")
    df = pd.merge(df, df_warm, on="date", how="left")
    df = pd.merge(df, df_actual, on="date", how="left")
    df = pd.merge(df, df_conf_int, on="date", how="left")
    df = pd.merge(df, df_forecast_nts, on="date", how="left")
    df = pd.merge(df, df_ntssn, on="date", how="left")
    
    return df

def add_demand_features(df):
    """
    Create additional derived features from the demand datasets.
    """
    # Difference between warm and cold demand
    df["demand_diff"] = df["demand_warm"] - df["demand_cold"]
    # Average of warm and cold demand
    df["demand_avg"] = df[["demand_warm", "demand_cold"]].mean(axis=1)
    # Forecast error: actual demand minus forecast NTS
    df["demand_forecast_error"] = df["demand_actual_ntsd_plus1"] - df["demand_forecast_nts"]
    # Spread between forecast NTS and NTSSN
    df["demand_nts_spread"] = df["demand_forecast_nts"] - df["demand_ntssn"]
    
    return df

# ---------- LDC Feature Engineering ----------
def add_ldc_features(df, t_ref=65):
    """
    Add LDC features: lagged demand, day-of-week, Fourier terms, holiday indicator,
    and temperature-derived features, plus linepack features.
    """
    # Use one temperature column (e.g., temp_fc_d) for HDD/CDD calculations.
    df["HDD"] = np.clip(t_ref - df["temp_fc_d"], 0, None)
    df["CDD"] = np.clip(df["temp_fc_d"] - t_ref, 0, None)
    
    # Lagged demand features based on SAP close values (can be rethought)
    df["demand_lag1"] = df["close"].shift(1)
    df["demand_lag7"] = df["close"].shift(7)
    
    df["dow"] = df["date"].dt.weekday
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow")
    df = pd.concat([df, dow_dummies], axis=1)
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["friday_startup"] = (df["dow"] == 4).astype(float) * 0.5
    
    df["sin7"] = np.sin(2*np.pi * df["dow"] / 7)
    df["cos7"] = np.cos(2*np.pi * df["dow"] / 7)
    
    cal = hol.USFederalHolidayCalendar()
    holidays = cal.holidays(start=df["date"].min(), end=df["date"].max())
    df["is_holiday"] = df["date"].isin(holidays).astype(int)
    
    # Temperature deviation features
    df["temp_dev_fc_minus1"] = df["temp_fc_d_minus1"] - df["temp_sn_d_minus1"]
    df["temp_dev_fc"] = df["temp_fc_d"] - df["temp_sn_d"]
    df["temp_dev_fc_plus1"] = df["temp_ac_d_plus1"] - df["temp_sn_d_plus1"]
    
    # Linepack-based features
    df["lp_closing_diff"] = df["closing_linepack_actual"] - df["opening_linepack_actual"]
    df["lp_closing_pred_error"] = df["closing_linepack_actual"] - df["predicted_closing_linepack"]
    
    return df

def build_feature_matrix(df):
    """
    Build feature matrix X and target y from the merged and engineered DataFrame.
    REMOVED: log_target and target_deriv from features to prevent data leakage
    """
    df = df.dropna()
    feature_cols = [
        "HDD", "CDD", "demand_lag1", "demand_lag7",
        "is_weekend", "friday_startup", "sin7", "cos7", "is_holiday",
        "closing_linepack_actual", "opening_linepack_actual",
        "linepack_hourly_agg_d_plus1", "predicted_closing_linepack",
        "lp_closing_diff", "lp_closing_pred_error",
        "temp_fc_d_minus1", "temp_fc_d", "temp_ac_d_plus1",
        "temp_sn_d_minus1", "temp_sn_d", "temp_sn_d_plus1",
        "temp_dev_fc_minus1", "temp_dev_fc", "temp_dev_fc_plus1",
        "CompositeWeatherVariableForecastLDZ(NW)",
        "CompositeWeatherVariableForecastLDZ(SC)",
        "CompositeWeatherVariableForecastLDZ(SE)",
        "WeatherCorrectionFactorForecast(NW)",
        "WeatherCorrectionFactorForecast(SC)",
        "WeatherCorrectionFactorForecast(SE)",
        # Demand raw features
        "demand_cold", "demand_warm", "demand_actual_ntsd_plus1",
        "demand_forecast_confidence_interval", "demand_forecast_nts", "demand_ntssn",
        # Derived demand features
        "demand_diff", "demand_avg", "demand_forecast_error", "demand_nts_spread"
    ]
    dow_dummies = [col for col in df.columns if col.startswith("dow_")]
    feature_cols.extend(dow_dummies)
    X = df[feature_cols]
    y = df["signal"]  # Binary signal: 1 = buy, 0 = sell/hold
    return X, y, df
