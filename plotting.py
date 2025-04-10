import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Function to format percentage in plots
def percentage_formatter(x, pos):
    return f'{x:.1f}%'

# 1. Load the backtest results data
# Assuming the data has been generated and saved as in your original code
def load_and_prepare_data(backtest_path="results/backtest_results.csv", 
                          market_data_path="data/market_benchmark.csv"):
    """
    Load backtest results and market benchmark data.
    If market data doesn't exist, we'll simulate a market benchmark.
    """
    # Load backtest results
    try:
        backtest_df = pd.read_csv(backtest_path, parse_dates=["date"])
    except FileNotFoundError:
        # Create sample data for demonstration
        dates = pd.date_range(start='2022-01-01', periods=252, freq='B')
        backtest_df = pd.DataFrame({
            'date': dates,
            'portfolio': np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates)))
        })
        backtest_df['target'] = np.random.normal(0.0003, 0.015, len(dates))
        backtest_df['combo_signal'] = np.random.choice([0, 1], size=len(dates), p=[0.3, 0.7])
    
    # Calculate daily returns for strategy
    backtest_df['daily_return'] = backtest_df['portfolio'].pct_change() * 100
    
    # Create or load market data
    try:
        market_df = pd.read_csv(market_data_path, parse_dates=["date"])
    except FileNotFoundError:
        # Simulate market data with a slight negative correlation to our strategy
        market_df = pd.DataFrame({
            'date': backtest_df['date'],
            'market_value': np.cumprod(1 + np.random.normal(0.0003, 0.012, len(backtest_df)))
        })
    
    # Calculate market returns if they don't exist
    if 'market_return' not in market_df.columns:
        market_df['market_return'] = market_df['market_value'].pct_change() * 100
    
    # Merge data
    combined_df = pd.merge(backtest_df, market_df[['date', 'market_value', 'market_return']], 
                         on='date', how='left')
    
    # Add day of week information
    combined_df['day_of_week'] = combined_df['date'].dt.day_name()
    
    # Calculate cumulative returns - FIXED:
    # We divide by 100 because daily_return and market_return are already in percentage form
    combined_df['cum_strategy_return'] = (combined_df['daily_return'] / 100 + 1).cumprod() - 1
    combined_df['cum_market_return'] = (combined_df['market_return'] / 100 + 1).cumprod() - 1
    
    # Define market regimes (simple implementation based on market returns)
    combined_df['regime'] = pd.qcut(combined_df['market_return'].rolling(20).mean(), 3, 
                               labels=['Bear', 'Neutral', 'Bull'])
    
    return combined_df

def plot_risk_ratios(data, window=150):
    """
    Compute and plot the rolling Sharpe, Sortino, and Calmar ratios on a single graph.
    
    Parameters:
      data (DataFrame): Must contain a 'date' column (datetime) and a 'daily_return' column (in percentage).
      window (int): Rolling window in days to compute the risk metrics.
    """
    # Ensure the date column is in datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Convert daily returns from percentage to decimals.
    daily_ret = data['daily_return'] / 100.0  
    risk_free_rate = 0.045 / 252.0  # Daily risk-free rate
    
    # Excess returns relative to risk-free rate.
    rolling_excess_ret = daily_ret - risk_free_rate
    
    # Rolling Sharpe Ratio:
    rolling_sharpe = (rolling_excess_ret.rolling(window).mean() / 
                        rolling_excess_ret.rolling(window).std()) * np.sqrt(252)
    
    # Rolling Sortino Ratio:
    negative_returns = rolling_excess_ret.copy()
    negative_returns[negative_returns > 0] = 0  # only keep negative returns
    rolling_sortino = (rolling_excess_ret.rolling(window).mean() /
                         negative_returns.rolling(window).std().replace(0, np.nan)) * np.sqrt(252)
    
    # Rolling Calmar Ratio:
    # Define a helper function to compute Calmar over a window.
    def rolling_calmar_calc(x):
        cum_returns = np.cumprod(1 + x)
        dd = (cum_returns - np.maximum.accumulate(cum_returns)) / np.maximum.accumulate(cum_returns)
        max_dd = dd.min()  # maximum drawdown over the window (negative or zero)
        ann_ret = np.mean(x) * 252
        return -ann_ret / max_dd if max_dd < 0 else np.nan
    rolling_calmar = daily_ret.rolling(window).apply(rolling_calmar_calc, raw=True)
    
    # Create the risk ratios plot.
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], rolling_sharpe, label="Rolling Sharpe", linewidth=2)
    plt.plot(data['date'], rolling_sortino, label="Rolling Sortino", linewidth=2)
    plt.plot(data['date'], rolling_calmar, label="Rolling Calmar", linewidth=2)
    plt.title(f"Rolling Risk Ratios (window = {window} days)")
    plt.xlabel("Date")
    plt.ylabel("Risk Ratio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/risk_ratios.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Risk ratios plot saved as 'plots/risk_ratios.png'.")

def plot_risk_ratios_with_returns(data, window=150):
    """
    Compute and plot the rolling risk ratios (Sharpe, Sortino, Calmar) along with
    the portfolio’s cumulative return on a twin-axis plot.
    
    Parameters:
      data (DataFrame): Must contain 'date', 'daily_return', and (optionally) 'cum_strategy_return'.
      window (int): Rolling window in days for computing the risk metrics.
    """
    # Ensure the date column is in datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Compute risk metrics as before.
    daily_ret = data['daily_return'] / 100.0  
    risk_free_rate = 0.045 / 252.0  
    rolling_excess_ret = daily_ret - risk_free_rate
    rolling_sharpe = (rolling_excess_ret.rolling(window).mean() / 
                        rolling_excess_ret.rolling(window).std()) * np.sqrt(252)
    negative_returns = rolling_excess_ret.copy()
    negative_returns[negative_returns > 0] = 0
    rolling_sortino = (rolling_excess_ret.rolling(window).mean() /
                         negative_returns.rolling(window).std().replace(0, np.nan)) * np.sqrt(252)
    def rolling_calmar_calc(x):
        cum_returns = np.cumprod(1 + x)
        dd = (cum_returns - np.maximum.accumulate(cum_returns)) / np.maximum.accumulate(cum_returns)
        max_dd = dd.min()
        ann_ret = np.mean(x) * 252
        return -ann_ret / max_dd if max_dd < 0 else np.nan
    rolling_calmar = daily_ret.rolling(window).apply(rolling_calmar_calc, raw=True)
    
    # Prepare the portfolio cumulative return.
    # If 'cum_strategy_return' exists, use it; otherwise, compute cumulative return.
    if 'cum_strategy_return' in data.columns:
        cum_return = data['cum_strategy_return'] * 100  # Convert to percentage.
    else:
        cum_return = ((daily_ret + 1).cumprod() - 1) * 100
    
    # Create twin-axis plot.
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot risk ratios on the primary y-axis.
    ax1.plot(data['date'], rolling_sharpe, label="Sharpe", color='tab:blue', linewidth=2)
    ax1.plot(data['date'], rolling_sortino, label="Sortino", color='tab:green', linewidth=2)
    ax1.plot(data['date'], rolling_calmar, label="Calmar", color='tab:orange', linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Risk Ratio Value", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create a second y-axis for the portfolio cumulative return.
    ax2 = ax1.twinx()
    ax2.plot(data['date'], cum_return, label="Cumulative Return", color='tab:red', 
             linewidth=2, linestyle='--')
    ax2.set_ylabel("Cumulative Return (%)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Combine legends from both axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title(f"Risk Ratios with Portfolio Cumulative Return (window = {window} days)")
    fig.tight_layout()
    plt.savefig("plots/risk_ratios_with_returns.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Composite plot of risk ratios and portfolio cumulative return saved as 'plots/risk_ratios_with_returns.png'.")

# Update the main function to include the logarithmic plot
def create_trading_strategy_visualizations(data):
    """Create a comprehensive set of visualizations for the trading strategy"""
    # Make sure the directory exists
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Drawdown plot
    plot_drawdown(data)
    
    # 2. Strategy vs Market Returns Over Time
    plot_cumulative_returns(data)
    
    # 3. Logarithmic Cumulative Returns - New plot
    plot_log_cumulative_returns(data)
    
    # 4. Returns by Day of Week - Strategy vs Market
    plot_returns_by_weekday(data)
    
    # 5. Percentage Time in Each Market Regime
    plot_regime_distribution(data)
    
    # 6. Market vs Strategy Returns per Regime
    plot_regime_returns(data)
    
    # 7. Rolling Performance Metrics (Sharpe, Sortino, Win Rate)
    plot_rolling_metrics(data)

    # 8. Linear Regression of Log Cumulative Returns
    plot_log_portfolio_regression(data)
    
    # 9. Monthly Returns: Market vs Strategy
    plot_monthly_returns(data)

    # 10. Risk metrics on the same plot
    plot_risk_ratios(data)

    # 11. Risk metrics with cumulative returns
    plot_risk_ratios_with_returns(data)

    print("All visualizations have been saved to the 'plots/' directory.")
    
def plot_drawdown(data):
    """Plot the drawdown of the strategy"""
    # Calculate drawdown
    peak = data['portfolio'].cummax()
    drawdown = ((data['portfolio'] - peak) / peak) * 100
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(data['date'], drawdown, 0, color='crimson', alpha=0.3)
    plt.plot(data['date'], drawdown, color='crimson', linewidth=1)
    
    # Add horizontal lines at specific drawdown levels
    plt.axhline(y=-5, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=-10, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=-15, color='gray', linestyle='--', alpha=0.7)
    
    # Formatting
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.title('Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Mark the maximum drawdown
    max_dd_idx = drawdown.idxmin()
    plt.scatter(data.loc[max_dd_idx, 'date'], drawdown.min(), 
                color='darkred', s=100, zorder=5)
    plt.annotate(f'Max DD: {drawdown.min():.2f}%', 
                xy=(data.loc[max_dd_idx, 'date'], drawdown.min()),
                xytext=(15, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig('plots/drawdown.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_returns(data):
    """Plot cumulative returns of the strategy vs the market"""
    plt.figure(figsize=(12, 6))
    
    # Convert to percentage for better readability
    strategy_cum_ret = data['cum_strategy_return'] * 100
    market_cum_ret = data['cum_market_return'] * 100
    
    plt.plot(data['date'], strategy_cum_ret, label='Strategy', linewidth=2)
    plt.plot(data['date'], market_cum_ret, label='Market', linewidth=2, alpha=0.7)
    
    # Fill between the two series to highlight outperformance
    plt.fill_between(data['date'], 
                    strategy_cum_ret, 
                    market_cum_ret, 
                    where=(strategy_cum_ret > market_cum_ret),
                    color='green', alpha=0.3)
    plt.fill_between(data['date'], 
                    strategy_cum_ret, 
                    market_cum_ret, 
                    where=(strategy_cum_ret <= market_cum_ret),
                    color='red', alpha=0.3)
    
    # Formatting
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.title('Cumulative Returns: Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add final return values
    final_strategy = strategy_cum_ret.iloc[-1]
    final_market = market_cum_ret.iloc[-1]
    
    plt.annotate(f'Strategy: {final_strategy:.2f}%', 
                xy=(data['date'].iloc[-1], final_strategy),
                xytext=(-100, 10), textcoords='offset points')
    plt.annotate(f'Market: {final_market:.2f}%', 
                xy=(data['date'].iloc[-1], final_market),
                xytext=(-100, -20), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('plots/cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_log_cumulative_returns(data):
    """Plot logarithmic cumulative returns of the strategy vs the market"""
    plt.figure(figsize=(12, 6))
    
    # Calculate logarithmic returns (log of wealth relative)
    log_strategy = np.log((data['cum_strategy_return'] + 1))
    log_market = np.log((data['cum_market_return'] + 1))
    
    plt.plot(data['date'], log_strategy, label='Strategy', linewidth=2)
    plt.plot(data['date'], log_market, label='Market', linewidth=2, alpha=0.7)
    
    # Fill between the two series to highlight outperformance
    plt.fill_between(data['date'], 
                    log_strategy, 
                    log_market, 
                    where=(log_strategy > log_market),
                    color='green', alpha=0.3)
    plt.fill_between(data['date'], 
                    log_strategy, 
                    log_market, 
                    where=(log_strategy <= log_market),
                    color='red', alpha=0.3)
    
    # Formatting
    plt.title('Logarithmic Cumulative Returns: Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add final log return values
    final_log_strategy = log_strategy.iloc[-1]
    final_log_market = log_market.iloc[-1]
    
    # Convert log returns to percentage for annotation (for better interpretability)
    final_strategy_pct = (np.exp(final_log_strategy) - 1) * 100
    final_market_pct = (np.exp(final_log_market) - 1) * 100
    
    plt.annotate(f'Strategy: {final_strategy_pct:.2f}%', 
                xy=(data['date'].iloc[-1], final_log_strategy),
                xytext=(-100, 10), textcoords='offset points')
    plt.annotate(f'Market: {final_market_pct:.2f}%', 
                xy=(data['date'].iloc[-1], final_log_market),
                xytext=(-100, -20), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('plots/log_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_returns_by_weekday(data):
    """Plot average returns by day of week for both strategy and market"""
    # Calculate average returns by day of week
    weekday_returns = data.groupby('day_of_week')[['daily_return', 'market_return']].mean()
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_returns = weekday_returns.reindex(day_order)
    
    plt.figure(figsize=(10, 6))
    
    # Create bars
    x = np.arange(len(weekday_returns.index))
    width = 0.35
    
    plt.bar(x - width/2, weekday_returns['daily_return'], width, label='Strategy', color='#1f77b4')
    plt.bar(x + width/2, weekday_returns['market_return'], width, label='Market', color='#ff7f0e', alpha=0.7)
    
    # Add data labels
    for i, v in enumerate(weekday_returns['daily_return']):
        plt.text(i - width/2, v + (0.02 if v >= 0 else -0.1), 
                f'{v:.2f}%', ha='center', fontsize=10)
    
    for i, v in enumerate(weekday_returns['market_return']):
        plt.text(i + width/2, v + (0.02 if v >= 0 else -0.1), 
                f'{v:.2f}%', ha='center', fontsize=10)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Formatting
    plt.xticks(x, weekday_returns.index)
    plt.title('Average Returns by Day of Week')
    plt.ylabel('Average Return (%)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/returns_by_weekday.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_regime_distribution(data):
    """Plot the percentage of time spent in each market regime"""
    # Calculate the percentage of time in each regime
    regime_counts = data['regime'].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(10, 6))
    
    # Create pie chart with a slight explosion on all wedges
    colors = ['#d62728', '#1f77b4', '#2ca02c']  # red, blue, green
    explode = [0.05, 0.05, 0.05]  # Explode all wedges slightly
    
    wedges, texts, autotexts = plt.pie(regime_counts, 
                                     explode=explode,
                                     labels=regime_counts.index, 
                                     autopct='%1.1f%%',
                                     textprops={'fontsize': 12},
                                     colors=colors,
                                     startangle=90,
                                     shadow=True)
    
    # Make the percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Percentage of Time in Each Market Regime')
    
    plt.tight_layout()
    plt.savefig('plots/regime_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_regime_returns(data):
    """Plot the average returns for each market regime"""
    # Calculate average returns by regime
    regime_returns = data.groupby('regime')[['daily_return', 'market_return']].mean()
    
    # Order by bearish to bullish
    if 'Bear' in regime_returns.index and 'Neutral' in regime_returns.index and 'Bull' in regime_returns.index:
        regime_returns = regime_returns.reindex(['Bear', 'Neutral', 'Bull'])
    
    plt.figure(figsize=(10, 6))
    
    # Create bars
    x = np.arange(len(regime_returns.index))
    width = 0.35
    
    plt.bar(x - width/2, regime_returns['daily_return'], width, label='Strategy', color='#1f77b4')
    plt.bar(x + width/2, regime_returns['market_return'], width, label='Market', color='#ff7f0e', alpha=0.7)
    
    # Add data labels
    for i, v in enumerate(regime_returns['daily_return']):
        plt.text(i - width/2, v + (0.02 if v >= 0 else -0.08), 
                f'{v:.2f}%', ha='center', fontsize=10)
    
    for i, v in enumerate(regime_returns['market_return']):
        plt.text(i + width/2, v + (0.02 if v >= 0 else -0.08), 
                f'{v:.2f}%', ha='center', fontsize=10)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Formatting
    plt.xticks(x, regime_returns.index)
    plt.title('Average Returns by Market Regime')
    plt.ylabel('Average Return (%)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/regime_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rolling_metrics(data):
    """Plot rolling Sharpe, Sortino, and Win Rate and print the dates with highest/lowest values."""
    # Constants
    window = 150  # 60-day rolling window
    risk_free_rate = 0.045 / 252  # Daily risk-free rate (2% annually)
    
    # Calculate necessary rolling metrics
    # 1. Rolling Sharpe Ratio
    rolling_ret = data['daily_return'] / 100  # Convert to decimal
    rolling_excess_ret = rolling_ret - risk_free_rate
    rolling_sharpe = (rolling_excess_ret.rolling(window).mean() / 
                      rolling_excess_ret.rolling(window).std()) * np.sqrt(252)
    
    # 2. Rolling Sortino Ratio
    # Only consider negative deviation for downside risk
    negative_returns = rolling_excess_ret.copy()
    negative_returns[negative_returns > 0] = 0
    rolling_sortino = (rolling_excess_ret.rolling(window).mean() / 
                       negative_returns.rolling(window).std().replace(0, np.nan)) * np.sqrt(252)
    
    # 3. Rolling Win Rate (% of days with positive returns)
    rolling_win_rate = (rolling_ret > 0).rolling(window).mean() * 100
    
    # Print the dates where the rolling metrics reached highest and lowest values.
    # Note: idxmax()/idxmin() returns the index of the max/min value.
    # We then retrieve the date corresponding to that index in the data['date'] column.
    
    # Sharpe Ratio
    sharpe_max_idx = rolling_sharpe.idxmax()
    sharpe_min_idx = rolling_sharpe.idxmin()
    sharpe_max_date = data['date'].iloc[sharpe_max_idx] if pd.notnull(sharpe_max_idx) else "N/A"
    sharpe_min_date = data['date'].iloc[sharpe_min_idx] if pd.notnull(sharpe_min_idx) else "N/A"
    print("Sharpe Ratio - Highest on:", sharpe_max_date, "| Lowest on:", sharpe_min_date)
    
    # Sortino Ratio
    sortino_max_idx = rolling_sortino.idxmax()
    sortino_min_idx = rolling_sortino.idxmin()
    sortino_max_date = data['date'].iloc[sortino_max_idx] if pd.notnull(sortino_max_idx) else "N/A"
    sortino_min_date = data['date'].iloc[sortino_min_idx] if pd.notnull(sortino_min_idx) else "N/A"
    print("Sortino Ratio - Highest on:", sortino_max_date, "| Lowest on:", sortino_min_date)
    
    # Win Rate
    winrate_max_idx = rolling_win_rate.idxmax()
    winrate_min_idx = rolling_win_rate.idxmin()
    winrate_max_date = data['date'].iloc[winrate_max_idx] if pd.notnull(winrate_max_idx) else "N/A"
    winrate_min_date = data['date'].iloc[winrate_min_idx] if pd.notnull(winrate_min_idx) else "N/A"
    print("Win Rate - Highest on:", winrate_max_date, "| Lowest on:", winrate_min_date)
    
    # Create three separate plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot Rolling Sharpe Ratio
    axes[0].plot(data['date'], rolling_sharpe, color='green', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_title('Rolling 150-Day Sharpe Ratio')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Rolling Sortino Ratio
    axes[1].plot(data['date'], rolling_sortino, color='purple', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_title('Rolling 150-Day Sortino Ratio')
    axes[1].set_ylabel('Sortino Ratio')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Rolling Win Rate
    axes[2].plot(data['date'], rolling_win_rate, color='blue', linewidth=2)
    axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    axes[2].set_title('Rolling 150-Day Win Rate')
    axes[2].set_ylabel('Win Rate (%)')
    axes[2].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axes[2].grid(True, alpha=0.3)
    
    # X-axis label for the bottom plot
    axes[2].set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('plots/rolling_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_log_portfolio_regression(data):
    """
    Plot a linear regression of the log(portfolio) values over time and annotate with R².
    
    Instead of using raw date numbers (which can be large), the time variable is transformed
    into the number of days since the first date in the dataset.
    """
    # Ensure date is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Calculate log portfolio (assuming the portfolio starts at 1 or similar scale)
    log_portfolio = np.log(data["portfolio"])
    
    # Compute time as the number of days since the first date
    time_in_days = (data["date"] - data["date"].min()).dt.days
    # Add constant for the intercept
    X = sm.add_constant(time_in_days)
    
    # Fit the linear regression model with statsmodels
    model = sm.OLS(log_portfolio, X).fit()
    r2_value = model.rsquared
    
    # Generate predicted log portfolio values from the model
    log_portfolio_fit = model.predict(X)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(data['date'], log_portfolio, color="blue", alpha=0.6, label="Log(Portfolio)")
    plt.plot(data['date'], log_portfolio_fit, color="red", linewidth=2, 
             label=f"Linear Fit (R² = {r2_value:.4f})")
    
    # Formatting and labels
    plt.title("Linear Regression of Log Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Log(Portfolio)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save the plot
    plt.savefig("plots/log_portfolio_regression.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Linear regression plot saved as 'plots/log_portfolio_regression.png' with R² = {r2_value:.4f}")

def plot_monthly_returns(data):
    """Create a heatmap of monthly returns for both strategy and market"""
    # Prepare monthly return data
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    
    # Aggregate returns by month
    monthly_strategy = data.groupby(['year', 'month'])['daily_return'].sum().unstack()
    monthly_market = data.groupby(['year', 'month'])['market_return'].sum().unstack()
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Custom colormap from red to white to green
    colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffff", 
             "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]
    cmap = LinearSegmentedColormap.from_list("RdBu", colors, N=100)
    
    # Plot strategy monthly returns
    sns.heatmap(monthly_strategy, cmap=cmap, ax=axes[0], center=0,
               annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': '%'})
    axes[0].set_title('Strategy Monthly Returns (%)')
    axes[0].set_ylabel('Year')
    axes[0].set_xlabel('')
    
    # Replace month numbers with month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for ax in axes:
        ax.set_xticklabels(month_names)
    
    # Plot market monthly returns
    sns.heatmap(monthly_market, cmap=cmap, ax=axes[1], center=0,
               annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': '%'})
    axes[1].set_title('Market Monthly Returns (%)')
    axes[1].set_ylabel('Year')
    axes[1].set_xlabel('Month')
    
    plt.tight_layout()
    plt.savefig('plots/monthly_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

# Execute everything if run as a script
if __name__ == "__main__":
    # Load data
    data = load_and_prepare_data()
    
    # Create all visualizations
    create_trading_strategy_visualizations(data)