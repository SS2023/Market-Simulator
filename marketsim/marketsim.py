""" HW4 : Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def get_portfolio_stats( port_val,
                        daily_rf=0.0,
                        samples_per_year=252.0
                        ):
    """
    Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
   -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
   """

    fn = "[get_portfolio_stats]: "
    # Compute daily returns, leave out first row

    cum_ret = 0
    avg_daily_ret = 0
    std_daily_ret = 0
    sharpe_ratio = 0

    # STEP 1. Compute  DAILY RETURN - of whole portfolio
    # correct the below
    daily_rets = (port_val / port_val.shift(1)) - 1

    # STEP 2: Compute CUMULATIVE return statistics difference b/w first/last day
    cum_ret = (port_val.iloc[-1] / port_val.iloc[0]) - 1

    # STEP 4: Compute avg_daily_ret & std_daily_ret
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()

    # Compute annualized Sharpe ratio, given risk-free rate of return
    k = np.sqrt(252)
    sharpe_ratio = k * np.mean(avg_daily_ret - 0) / std_daily_ret

    return \
        cum_ret, \
        avg_daily_ret, \
        std_daily_ret, \
        sharpe_ratio

def get_SPX_stats( port_val,
                        daily_rf=0.0,
                        samples_per_year=252.0
                        ):
    """
    Calculate statistics on given SPX values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
   -------
        cum_ret_SPX: cumulative return
        avg_daily_ret_SPX: average of daily returns
        std_daily_ret_SPX: standard deviation of daily returns
        sharpe_ratio_SPX: annualized Sharpe ratio
   """

    fn = "[get_portfolio_stats]: "
    # Compute daily returns, leave out first row

    cum_ret = 0
    avg_daily_ret = 0
    std_daily_ret = 0
    sharpe_ratio = 0

    start_date = dt.datetime(2011,1,14)
    end_date = dt.datetime(2011,12,14)

    SPX_data = get_data(['$SPX'], dates=pd.date_range(start_date, end_date))
    SPX_data = SPX_data[SPX_data.columns[1]]

    # STEP 1. Compute  DAILY RETURN - of whole portfolio
    # correct the below
    daily_ret_SPX = (SPX_data / SPX_data.shift(1)) - 1

    # STEP 2: Compute CUMULATIVE return statistics difference b/w first/last day
    cum_ret_SPX = (SPX_data[-1] / SPX_data[0]) - 1

    # STEP 4: Compute avg_daily_ret & std_daily_ret
    avg_daily_ret_SPX = daily_ret_SPX.mean()
    std_daily_ret_SPX = daily_ret_SPX.std()

    # Compute annualized Sharpe ratio, given risk-free rate of return
    k = np.sqrt(252)
    sharpe_ratio_SPX = k * np.mean(daily_ret_SPX - 0) / std_daily_ret_SPX

    return \
        cum_ret_SPX, \
        avg_daily_ret_SPX, \
        std_daily_ret_SPX, \
        sharpe_ratio_SPX

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission = 9.95, impact = 0.005):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # reads the orders.csv file
    orders_df = pd.read_csv(orders_file, index_col= 'Date')
    orders_df = orders_df.sort_values(by=['Date'])

    # start and end dates
    dates = pd.date_range(orders_df.index[0], orders_df.index[-1])

    # place symbols into a list
    symbols = np.array(orders_df.Symbol.unique()).tolist()

    # data for the symbols
    cost = get_data(symbols, dates)
    cost = cost.sort_index()
    cost = cost.fillna(method='ffill')
    cost = cost.fillna(method='bfill')

    # stock prices for the symbols
    for x in symbols:
        cost['portval'] = pd.Series(start_val, index = cost.index)
        cost['Cash'] = pd.Series(start_val, index = cost.index)
        cost[x + ' Shares'] = pd.Series(0, index = cost.index)

    # impact cost when buying and selling
    buy_impact = 1 + impact
    sell_impact = 1 - impact

    # loop going through the file
    for i, order in orders_df.iterrows():
        symbol = order['Symbol']

        # shares are added and cash is subtracted
        if order['Order'] == 'BUY':
            cost.loc[i: , symbol + ' Shares'] += order['Shares']
            cost.loc[i: , 'Cash'] -= ((cost.loc[i , symbol] * buy_impact * order['Shares']) + commission)

        # shares are subtracted and cash is added
        if order['Order'] == 'SELL':
            cost.loc[i: , symbol + ' Shares'] -= order['Shares']
            cost.loc[i: , 'Cash'] += ((cost.loc[i , symbol] * sell_impact * order['Shares']) - commission)

    # gets final value
    for j, row in cost.iterrows():
        val = 0
        for symbol in symbols:
            val += cost.loc[j , symbol + ' Shares'] * row[symbol]
        cost.loc[j , 'portval'] = cost.loc[j , 'Cash'] + val

        portvals = cost.loc[:, 'portval']

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    #start_date  = dt.datetime(2011,1,14)
    #end_date    = dt.datetime(2011,12,14)
    #portvals    = get_data(['IBM'], pd.date_range(start_date, end_date))
    #portvals    = portvals[['IBM']]  # remove SPY

    return portvals


def RunCode():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2011,1,14)
    end_date = dt.datetime(2011,12,14)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # portval data
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # $SPX data
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_SPX_stats(portvals)

    # Compare portfolio against $SPX
    print( f"Start Date:  \t\t\t{start_date}" )
    print( f"End Date:    \t\t\t{end_date}" )

    print("")
    print( f"Sharpe Ratio of Fund: \t\t{sharpe_ratio}")
    print( f"Sharpe Ratio of $SPX:  \t\t{sharpe_ratio_SPX}\n")

    print( f"Cumulative Return of Fund: \t{cum_ret}")
    print( f"Cumulative Return of $SPX : \t{cum_ret_SPX}\n")

    print( f"Standard Deviation of Fund: \t{std_daily_ret}")
    print( f"Standard Deviation of $SPX : \t{std_daily_ret_SPX}\n")

    print( f"Average Daily Return of Fund: \t{avg_daily_ret}")
    print( f"Average Daily Return of $SPX : \t{avg_daily_ret_SPX}\n")

    print( f"Final Portfolio Value: \t\t{portvals[-1]}")

if __name__ == "__main__":
    RunCode()
