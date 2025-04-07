import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

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
    
    # 8. Monthly Returns: Market vs Strategy
    plot_monthly_returns(data)
    
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
                xytext=(15, -30), textcoords='offset points',
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
    """Plot rolling Sharpe, Sortino, and Win Rate"""
    # Constants
    window = 60  # 60-day rolling window
    risk_free_rate = 0.02 / 252  # Daily risk-free rate (2% annually)
    
    # Calculate necessary rolling metrics
    # 1. Rolling Sharpe Ratio
    rolling_ret = data['daily_return'] / 100  # Convert to decimal
    rolling_excess_ret = rolling_ret - risk_free_rate
    rolling_sharpe = (rolling_excess_ret.rolling(window).mean() / 
                     rolling_excess_ret.rolling(window).std()) * np.sqrt(252)
    
    # 2. Rolling Sortino Ratio
    negative_returns = rolling_excess_ret.copy()
    negative_returns[negative_returns > 0] = 0
    rolling_sortino = (rolling_excess_ret.rolling(window).mean() / 
                      negative_returns.rolling(window).std().replace(0, np.nan)) * np.sqrt(252)
    
    # 3. Rolling Win Rate
    rolling_win_rate = (rolling_ret > 0).rolling(window).mean() * 100
    
    # Create three separate plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot Rolling Sharpe Ratio
    axes[0].plot(data['date'], rolling_sharpe, color='green', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_title('Rolling 60-Day Sharpe Ratio')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Rolling Sortino Ratio
    axes[1].plot(data['date'], rolling_sortino, color='purple', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_title('Rolling 60-Day Sortino Ratio')
    axes[1].set_ylabel('Sortino Ratio')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Rolling Win Rate
    axes[2].plot(data['date'], rolling_win_rate, color='blue', linewidth=2)
    axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    axes[2].set_title('Rolling 60-Day Win Rate')
    axes[2].set_ylabel('Win Rate (%)')
    axes[2].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axes[2].grid(True, alpha=0.3)
    
    # X-axis label for the bottom plot
    axes[2].set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('plots/rolling_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

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