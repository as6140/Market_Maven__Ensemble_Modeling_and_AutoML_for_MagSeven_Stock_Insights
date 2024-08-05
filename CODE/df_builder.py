import pandas as pd
import requests
from datetime import datetime, timedelta
import Levenshtein as lev
import numpy as np
import CompoundIndicators as ci
import re


def FRED_data(ticker, start_date, end_date, fill_method='ffill'):
    if ticker == "DEV":
        filename = "SDJOBS.csv"  # Imports developer jobs
    elif ticker == "JOBS":
        filename = "ALLJOBS.csv"  # Imports all jobs
    elif ticker == "IAE":
        filename = "IAE.csv"  # Imports investor allocations to equity
    elif ticker == "TBILL":
        filename = "tbill_rates.csv"  # Imports t bill rates
    else:
        print("Error, no such FRED data.")  # Any other input -> error
        return None

    data = pd.read_csv('./data/' + filename)
    date_column = data.columns[0]
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.set_index(date_column).asfreq('D')

    if fill_method == 'ffill':
        data = data.ffill()
    elif fill_method == 'bfill':
        data = data.bfill()
    elif fill_method == 'interpolate':
        data = data.interpolate()
    else:
        print("Fill method not supported.")  # Any other input -> error
        return None

    filtered_data = data[start_date:end_date]

    # if ticker == "JOBS":
    #     filtered_data['ALLJOBS'] = filtered_data['ALLJOBS'] / filtered_data['ALLJOBS'].loc(0)

    return filtered_data
def extract_mean_value(amount_str):
    pattern = r"\$(\d{1,3}(?:,\d{3})*)"
    numbers = re.findall(pattern, amount_str)
    numbers_int = [int(n.replace(',', '')) for n in numbers]
    mean_value = sum(numbers_int) / len(numbers_int)
    return mean_value

def get_income_statement_data(symbol, start_date, end_date, period='annual'):
    """
    Fetches revenue and research & development expenses for the given stock ticker symbol
    within the specified date range and returns them as a pandas DataFrame.

    Parameters:
    - symbol (str): The stock ticker symbol.
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.
    - period (str): The period for the financial data ('annual' or 'quarter').

    Returns:
    - pandas.DataFrame: A DataFrame containing revenue and R&D expenses for each period within the date range.
    """
    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"  # Replace with your actual API key
    endpoint = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period={period}&apikey={api_key}"

    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.json()
        filtered_data = [
            {
                "date": item["date"],
                "RnDbyRev": (item["researchAndDevelopmentExpenses"]+1) / (item["revenue"]+20)
            }
            for item in data
            if start_date <= item["date"] <= end_date
        ]
        df = pd.DataFrame(filtered_data)

        if df.empty or 'date' not in df.columns:
            print(f"No data returned for {symbol} or 'date' column is missing. - get_financial_statement_data")
            return pd.DataFrame()

        return df.set_index('date')
    else:
        print(f"Failed to fetch income statement data for {symbol}. Status code: {response.status_code}")
        return pd.DataFrame()


def split_name(name):
    parts = name.split(" ")
    filtered_part = ''.join([char for char in parts[0] if char.isalpha()])
    return filtered_part if len(filtered_part) > 1 else parts[1]


def match_names(first, last, verbose=False):
    data = pd.read_csv('./data/merged_us_senators.csv')

    first_input = split_name(first).lower()
    last_input = split_name(last).lower()

    best_match = None
    highest_similarity = -1

    filtered_candidates = data[(data['First'].str.lower().str.startswith(first_input[0])) &
                               (data['Last'].str.lower().str.startswith(last_input[0]))]

    for index, row in filtered_candidates.iterrows():
        row_first = split_name(row['First'])
        row_last = split_name(row['Last'])

        first_name_similarity = lev.ratio(row_first, first_input) ** 2
        last_name_similarity = lev.ratio(row_last, last_input) ** 2

        combined_similarity = (first_name_similarity + last_name_similarity)

        if combined_similarity > highest_similarity:
            best_match = row
            highest_similarity = combined_similarity
    if verbose == True:
        print("INPUT:", first + " " + last)
        print("INPUT2:", split_name(first) + " " + split_name(last))
        if best_match is not None:
            print("OUTPUT:", best_match['First'] + " " + best_match['Last'])
        else:
            print("Senator Not Found")

    return best_match['Party'] if best_match is not None else 'I'


def get_senate_transactions(symbol, start_date=None, end_date=None):
    """
    Fetches Senate transactions data from the Financial Modeling Prep API for a specific stock symbol
    between the given start date and end date, and returns a DataFrame containing the transaction date,
    amount, and type of transaction.

    If no start_date is provided, the default will be the current date. If no end_date is provided,
    the default will be two years prior to the current date.

    Parameters:
    - symbol (str): The stock ticker symbol for which to fetch Senate transactions data.
    - start_date (str, optional): The start date of the date range in 'YYYY-MM-DD' format. Defaults to the current date.
    - end_date (str, optional): The end date of the date range in 'YYYY-MM-DD' format. Defaults to two years before the current date.

    Returns:
    - pandas.DataFrame: A DataFrame containing the transaction date, amount, and type of transaction
      for each relevant entry found between the start date and end date. Columns are 'Transaction Date',
      'Amount', and 'Type'.
    """
    if not start_date:
        start_date = datetime.now().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    api_endpoint = f"https://financialmodelingprep.com/api/v4/senate-trading?symbol={symbol}&apikey={api_key}"

    response = requests.get(api_endpoint)

    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    filtered_data = []

    for item in data:
        transaction_date = datetime.strptime(item['transactionDate'], '%Y-%m-%d')
        start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')

        # print(f"Transaction Date: {transaction_date}, Start: {start_date_parsed}, End: {end_date_parsed}")  # Debug date comparison
        # print(f"Amount: {item['amount']}, Type: {item['type']}")  # Debug amount and type

        if start_date_parsed <= transaction_date <= end_date_parsed:
            # if 1==1:
            filtered_data.append({
                "Transaction Date": item['transactionDate'],
                "Amount": item['amount'],
                "Type": item['type'],
                "First": item['firstName'],
                "Last": item['lastName'],
                "Party": match_names(item['firstName'], item['lastName'])
            })
            # print(f"Filtered data length: {len(filtered_data)}")  # Check the size
            # print(f"Last appended item: {filtered_data[-1]}")  # Check the last item appended

    df = pd.DataFrame(filtered_data)

    # Only convert 'Amount' and 'Type' to categorical if the DataFrame is not empty and the columns exist
    if not df.empty:
        if 'Amount' in df.columns:
            df['Amount'] = pd.Categorical(df['Amount'])
        if 'Type' in df.columns:
            df['Type'] = pd.Categorical(df['Type'])

    return df


def get_social_metric(symbol, start_date=None, end_date=None, metrics=['stocktwitsPosts', 'twitterPosts',
                                                                       'stocktwitsComments', 'twitterComments',
                                                                       'stocktwitsLikes', 'twitterLikes',
                                                                       'stocktwitsImpressions',
                                                                       'twitterImpressions', 'stocktwitsSentiment',
                                                                       'twitterSentiment']):
    """
    Fetches specified metric(s) data from the Financial Modeling Prep API for a specific stock symbol
    between the given start date and today. It collects data starting from the most recent
    entry back to the start date. If start_date is not provided, defaults to two years before today.

    Parameters:
    - symbol (str): The stock ticker symbol for which to fetch sentiment data.
    - start_date (datetime or str, optional): The start date of the date range as a datetime object
      or a string in 'YYYY-MM-DD' format. Defaults to two years before the current date.
    - end_date (datetime or str, optional): The end date of the date range as a datetime object
      or a string in 'YYYY-MM-DD' format. Defaults to the current date.
    - metrics (list of str): A list of specific metric keys from the API response to filter and return. Defaults to
      ['stocktwitsSentiment']. Each specified metric will represent a separate column in the output DataFrame.
      The metric can be any of the following: stocktwitsPosts, twitterPosts,
      stocktwitsComments, twitterComments, stocktwitsLikes, twitterLikes, stocktwitsImpressions,
      twitterImpressions, stocktwitsSentiment, twitterSentiment. Note that it appears most metrics
      are 60 days delayed, ending on January 31, 2024 currently.

    Returns:
    - pandas.DataFrame: A DataFrame containing the date and the specified sentiment metrics for each entry
      found between the start date and today. The DataFrame's first column is 'date', followed by columns for each specified metric.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=730)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    page = 0

    sentiment_records = []

    while True:
        api_endpoint = f"https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol={symbol}&page={page}&apikey={api_key}"
        response = requests.get(api_endpoint)

        if response.status_code != 200:
            print(f"Failed to fetch data for page {page}. Status code: {response.status_code}")
            break

        data = response.json()
        if not data:
            print("No more data available.")
            break

        # Convert start_date and end_date to datetime.date objects if they are datetime.datetime
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        for item in data:
            item_datetime = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')

            if item_datetime.strftime('%H:%M:%S') != "11:00:00":
                continue

            item_date_only = item_datetime.date()

            if item_date_only < start_date:
                return pd.DataFrame(sentiment_records)

            record = {"date": item_date_only}
            for metric in metrics:
                record[metric] = item.get(metric, 0)
            sentiment_records.append(record)

        page += 1

    return pd.DataFrame(sentiment_records)


def get_analyst_recommendations(symbol, start_date=None, end_date=None):
    """
    Fetches analyst recommendations data from the Financial Modeling Prep API for a specific stock symbol
    between the given start date and today. It collects data starting from the most recent entry back to the start date.
    If start_date is not provided, defaults to two years before today. If end_date is not provided, defaults to today's date.

    Parameters:
    - symbol (str): The stock ticker symbol for which to fetch analyst recommendations.
    - start_date (datetime or str, optional): The start date of the date range as a datetime object
      or a string in 'YYYY-MM-DD' format. Defaults to two years before the current date.
    - end_date (datetime or str, optional): The end date of the date range as a datetime object
      or a string in 'YYYY-MM-DD' format. Defaults to the current date.

    Returns:
    - pandas.DataFrame: A DataFrame containing the date and the analyst ratings for each entry
      found between the start date and today. Columns are 'date', 'analystRatingsBuy', 'analystRatingsHold',
      'analystRatingsSell', 'analystRatingsStrongSell', and 'analystRatingsStrongBuy'.
    """
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=730)  # Default to approximately two years before today

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    page = 0

    recommendations = []

    while True:
        api_endpoint = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{symbol}?apikey={api_key}&page={page}"
        response = requests.get(api_endpoint)

        if response.status_code != 200:
            print(f"Failed to fetch data for page {page}. Status code: {response.status_code}")
            break

        data = response.json()
        if not data:
            print("No more data available.")
            break

        for item in data:
            item_date = datetime.strptime(item['date'], '%Y-%m-%d')
            if item_date < start_date:
                # Once the data earlier than the start_date is encountered, return the collected data
                return pd.DataFrame(recommendations)

            recommendation_record = {
                "Date": item_date,
                "analystRatingsBuy": item.get("analystRatingsbuy", 0),
                "analystRatingsHold": item.get("analystRatingsHold", 0),
                "analystRatingsSell": item.get("analystRatingsSell", 0),
                "analystRatingsStrongSell": item.get("analystRatingsStrongSell", 0),
                "analystRatingsStrongBuy": item.get("analystRatingsStrongBuy", 0),
            }
            recommendations.append(recommendation_record)

        page += 1

    return pd.DataFrame(recommendations)


def get_house_trades(symbol, start_date=None, end_date=None):
    """
    Fetches House trades disclosure data from the Financial Modeling Prep API for a specific stock symbol
    between the given start date and end date, and returns a DataFrame containing the transaction date,
    amount, and type of transaction.

    If no start_date is provided, the default will be the current date. If no end_date is provided,
    the default will be two years prior to the current date.

    Parameters:
    - symbol (str): The stock ticker symbol for which to fetch House trades disclosure data.
    - start_date (str, optional): The start date of the date range in 'YYYY-MM-DD' format. Defaults to the current date.
    - end_date (str, optional): The end date of the date range in 'YYYY-MM-DD' format. Defaults to two years before the current date.

    Returns:
    - pandas.DataFrame: A DataFrame containing the transaction date, amount, and type of transaction
      for each relevant entry found between the start date and end date. Columns are 'Transaction Date',
      'Amount', and 'Type'.
    """
    if not start_date:
        start_date = datetime.now().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    api_endpoint = f"https://financialmodelingprep.com/api/v4/senate-disclosure?symbol={symbol}&apikey={api_key}"

    response = requests.get(api_endpoint)
    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    filtered_data = []

    for item in data:
        transaction_date = datetime.strptime(item['transactionDate'], '%Y-%m-%d')
        start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
        # print(f"Transaction Date: {transaction_date}, Start: {start_date_parsed}, End: {end_date_parsed}")  # Debug date comparison
        # print(f"Amount: {item['amount']}, Type: {item['type']}")  # Debug amount and type

        # if start_date_parsed <= transaction_date <= end_date_parsed:
        if 1 == 1:
            filtered_data.append({
                "Transaction Date": item['transactionDate'],
                "Amount": item['amount'],
                "Type": item['type']
            })

    df = pd.DataFrame(filtered_data)

    # Convert 'Amount' and 'Type' to categorical if the DataFrame is not empty and the columns exist
    if not df.empty:
        if 'Amount' in df.columns:
            df['Amount'] = pd.Categorical(df['Amount'])
        if 'Type' in df.columns:
            df['Type'] = pd.Categorical(df['Type'])

    return df


def get_esg_scores(symbol, start_date=None, end_date=None):
    """
    Fetches ESG (Environmental, Social, Governance) scores data from the Financial Modeling Prep API
    for a specific stock symbol between the given start date and end date, and returns a DataFrame
    containing the date, environmental score, social score, governance score, and ESG score.

    If no start_date is provided, the default will be the current date. If no end_date is provided,
    the default will be two years prior to the current date.

    Parameters:
    - symbol (str): The stock ticker symbol for which to fetch ESG scores.
    - start_date (str, optional): The start date of the date range in 'YYYY-MM-DD' format. Defaults to the current date.
    - end_date (str, optional): The end date of the date range in 'YYYY-MM-DD' format. Defaults to two years before the current date.

    Returns:
    - pandas.DataFrame: A DataFrame containing the date, environmental score, social score, governance score,
      and total ESG score for each relevant entry found between the start date and end date.
    """
    if not start_date:
        start_date = datetime.now().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    api_endpoint = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={symbol}&apikey={api_key}"

    response = requests.get(api_endpoint)
    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    filtered_data = []

    for item in data:
        esg_date = datetime.strptime(item['date'], '%Y-%m-%d')
        start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')

        # if start_date_parsed <= esg_date <= end_date_parsed:
        if 1 == 1:
            filtered_data.append({
                "Date": item['date'],
                "EnvironmentalScore": item['environmentalScore'],
                "SocialScore": item['socialScore'],
                "GovernanceScore": item['governanceScore'],
                "ESGScore": item['ESGScore']
            })

    return pd.DataFrame(filtered_data)


def handle_duplicate_dates(df, duplicate_dates_handling, custom_agg_func=None):
    """
    Handles duplicate dates in the DataFrame based on the specified handling strategy.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to process.
    - duplicate_dates_handling (str): The method for handling duplicate dates.
    - custom_agg_func (callable, optional): Custom aggregation function for 'custom' handling.
    """
    if duplicate_dates_handling == 'keep_first':
        return df[~df.index.duplicated(keep='first')]
    elif duplicate_dates_handling == 'average':
        return df.groupby(df.index).mean()
    elif duplicate_dates_handling == 'drop_all':
        return df.loc[~df.index.duplicated(keep=False)]
    elif duplicate_dates_handling in ['sum', 'max', 'min', 'median', 'last']:
        agg_func = np.sum if duplicate_dates_handling == 'sum' else getattr(np, duplicate_dates_handling)
        return df.groupby(df.index).agg(agg_func)
    elif duplicate_dates_handling == 'custom':
        if custom_agg_func is not None:
            return df.groupby(df.index).agg(custom_agg_func)
        else:
            raise ValueError("custom_agg_func must be provided when duplicate_dates_handling='custom'.")
    elif duplicate_dates_handling == 'no_handling':
        if not df.index.is_unique:
            raise ValueError("Duplicate dates found. Please choose a different duplicate_dates_handling option.")
    else:
        raise ValueError(f"Invalid duplicate_dates_handling option: {duplicate_dates_handling}")


def forward_fill_to_frequency(df, output_frequency='daily', duplicate_dates_handling='no_handling',
                              custom_agg_func=None):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'Date':
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'  # Renaming index to a consistent name

    # Apply duplicate handling
    df = handle_duplicate_dates(df, duplicate_dates_handling, custom_agg_func)

    # Determine the resample rule based on the desired output frequency
    frequency_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'quarterly': 'Q'}
    resample_rule = frequency_map.get(output_frequency, 'D')  # Default to 'daily'

    # Resample and forward fill
    df_resampled = df.resample(resample_rule).ffill()

    return df_resampled


def get_keymetrics(symbol, start_date=None, end_date=None):
    """
    Fetches key financial metrics for the given stock ticker symbol within the specified date range
    and returns them as a pandas DataFrame.

    Parameters:
    - symbol (str): The stock ticker symbol.
    - start_date (str, optional): The start date in 'YYYY-MM-DD' format. Defaults to two years before the current date.
    - end_date (str, optional): The end date in 'YYYY-MM-DD' format. Defaults to the current date.

    Returns:
    - pandas.DataFrame: A DataFrame containing the requested metrics for each period within the date range.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    endpoint = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period=quarter&apikey={api_key}"

    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.json()
        # Filter data by date range and select specific metrics
        filtered_data = [
            {
                "date": item["date"],
                "evToOperatingCashFlow": item["evToOperatingCashFlow"],
                "evToFreeCashFlow": item["evToFreeCashFlow"],
                "debtToEquity": item["debtToEquity"],
                "debtToAssets": item["debtToAssets"],
                "netDebtToEBITDA": item["netDebtToEBITDA"],
                "receivablesTurnover": item["receivablesTurnover"],
            }
            for item in data
            if start_date <= item["date"] <= end_date
        ]
        # Create and return a DataFrame from the filtered data
        return pd.DataFrame(filtered_data).set_index('date')
    else:
        print(f"Failed to fetch key metrics for {symbol}. Status code: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

def get_all_technical_indicators(ticker, start_date, end_date, n=7):
    api_key = 'tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4'
    api_indicators = ['sma', 'ema', 'wma', 'rsi', 'adx', 'standardDeviation']
    period = 20
    combined_df = pd.DataFrame()

    for indicator in api_indicators:
        url = f'https://financialmodelingprep.com/api/v3/technical_indicator/1day/{ticker}?type={indicator}&period={period}&apikey={api_key}'
        prices = requests.get(url).json()
        price_df = pd.DataFrame(prices)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df[(price_df['date'] >= start_date) & (price_df['date'] <= end_date)]

        if combined_df.empty:
            combined_df = price_df
        else:
            combined_df = pd.merge(combined_df, price_df[['date', indicator]], on='date', how='left')

    # Now add custom indicators
    combined_df.set_index('date', inplace=True)
    combined_df.sort_values('date', inplace=True)
    combined_df['BB_percentage'] = ci.bollinger_bands(combined_df['close'])
    ichimoku = ci.ichimoku_cloud(combined_df['high'], combined_df['low'])
    combined_df['Ichimoku'] = ichimoku
    combined_df['volume_oscillator'] = ci.volume_oscillator(combined_df['volume'])
    macd = ci.MACD(combined_df['close'])
    combined_df['MACD'] = macd

    return combined_df

def get_bipartisan_investments(symbol, start_date=None, end_date=None, alph=0.5, smoothing = True):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    house_endpoint = f"https://financialmodelingprep.com/api/v4/senate-disclosure?symbol={symbol}&apikey={api_key}"
    senate_endpoint = f"https://financialmodelingprep.com/api/v4/senate-trading?symbol={symbol}&apikey={api_key}"

    transactions = []

    response = requests.get(house_endpoint)
    if response.status_code == 200:
        data = response.json()
        for item in data:
            transaction_date = datetime.strptime(item['transactionDate'], '%Y-%m-%d')
            start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
            if start_date_parsed <= transaction_date <= end_date_parsed:
                transactions.append({
                    "Transaction Date": transaction_date,
                    "Buys": extract_mean_value(item['amount']) * (0 if 'sale' in item['type'].lower() else 1),
                    "Sells": extract_mean_value(item['amount']) * (1 if 'sale' in item['type'].lower() else 0),
                })

    response = requests.get(senate_endpoint)
    if response.status_code == 200:
        data = response.json()
        for item in data:
            transaction_date = datetime.strptime(item['transactionDate'], '%Y-%m-%d')
            start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
            if start_date_parsed <= transaction_date <= end_date_parsed:
                transactions.append({
                    "Transaction Date": transaction_date,
                    "Buys": extract_mean_value(item['amount']) * (0 if 'sale' in item['type'].lower() else 1),
                    "Sells": extract_mean_value(item['amount']) * (1 if 'sale' in item['type'].lower() else 0),
                })

    df = pd.DataFrame(transactions)
    df_grouped = df.groupby(['Transaction Date']).agg({'Buys': 'sum', 'Sells': 'sum'}).reset_index()

    start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start=start_date_parsed, end=end_date_parsed)

    df_grouped.set_index('Transaction Date', inplace=True)
    df_grouped = df_grouped.reindex(date_range, fill_value=0).rename_axis('Transaction Date')

    df_grouped['BP Buys'] = df_grouped['Buys'].ewm(alpha=alph, adjust=False).mean()
    df_grouped['BP Sells'] = df_grouped['Sells'].ewm(alpha=alph, adjust=False).mean()
    total_ewm = df_grouped['BP Buys'] + df_grouped['BP Sells']

    df_grouped['Bipartisan Buy Ratio'] = np.where(total_ewm == 0, 0.5, df_grouped['BP Buys'] / total_ewm)
    df_grouped.drop(['Buys', 'Sells'], axis=1, inplace=True)
    if smoothing == True:
        df_grouped['Bipartisan Buy Ratio'] = df_grouped['Bipartisan Buy Ratio'].ewm(alpha=alph, adjust=False).mean()
    # df_grouped['BP Sells'] = -df_grouped['BP Sells']
    return df_grouped
def get_partisan_investments(symbol, start_date=None, end_date=None, alph=0.7, smoothing = True):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    api_endpoint = f"https://financialmodelingprep.com/api/v4/senate-trading?symbol={symbol}&apikey={api_key}"
    response = requests.get(api_endpoint)
    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    filtered_data = []
    for item in data:
        transaction_date = datetime.strptime(item['transactionDate'], '%Y-%m-%d')
        start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')

        if start_date_parsed <= transaction_date <= end_date_parsed:
            filtered_data.append({
                "Transaction Date": transaction_date,
                "RBuys DSells": (extract_mean_value(item['amount']) * (0 if 'sale' in item['type'].lower() else 1) * (1 if match_names(item['firstName'], item['lastName']) == 'R' else 0)) + (extract_mean_value(item['amount']) * (1 if 'sale' in item['type'].lower() else 0) * (1 if match_names(item['firstName'], item['lastName']) == 'D' else 0)),
                "RSells DBuys": (extract_mean_value(item['amount']) * (1 if 'sale' in item['type'].lower() else 0) * (1 if match_names(item['firstName'], item['lastName']) == 'R' else 0)) + (extract_mean_value(item['amount']) * (0 if 'sale' in item['type'].lower() else 1) * (1 if match_names(item['firstName'], item['lastName']) == 'D' else 0)),
            })

    df = pd.DataFrame(filtered_data)
    df_grouped = df.groupby(['Transaction Date']).agg({'RBuys DSells': 'sum', 'RSells DBuys': 'sum'}).reset_index()

    start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start=start_date_parsed, end=end_date_parsed)

    df_grouped.set_index('Transaction Date', inplace=True)
    df_grouped = df_grouped.reindex(date_range, fill_value=0).rename_axis('Transaction Date')

    buys_ewm = df_grouped['RBuys DSells'].ewm(alpha=alph, adjust=False).mean()
    df_grouped['RBuys DSells'] = buys_ewm
    sells_ewm = df_grouped['RSells DBuys'].ewm(alpha=alph, adjust=False).mean()
    df_grouped['RSells DBuys'] = sells_ewm
    total_ewm = buys_ewm + sells_ewm

    df_grouped['Partisan Investing'] = np.where(total_ewm == 0, 0.5, buys_ewm / total_ewm) #1 is republic, 0 is democrat
    df_grouped.drop(['RBuys DSells', 'RSells DBuys'], axis=1, inplace=True)
    if smoothing == True:
        df_grouped['Partisan Investing'] = df_grouped['Partisan Investing'].ewm(alpha=alph, adjust=False).mean()
    return df_grouped

def process_df(df, type='visual'):
    df_new = pd.DataFrame()
    if type == 'visual':
        df_new['Price'] = df['close'] # (filtered_data / filtered_data.dropna().iloc[0])
        df_new['Partisan Investing'] = df['Partisan Investing']
        df_new['Job Postings'] = (df['ALLJOBS'] / df['ALLJOBS'].dropna().iloc[0])
        df_new['Equity Investment'] = (df['IAE'] / df['IAE'].dropna().iloc[0])
        df_new['Risk Free Rate'] = (df['4 WEEKS BANK DISCOUNT'] / df['4 WEEKS BANK DISCOUNT'].dropna().iloc[-181]) #
        df_new['Twitter Posts'] = (df['stocktwitsPosts'] + df['twitterPosts'])
        # df_new['Social 2'] = (df['stocktwitsComments'] + df['twitterComments'])
        # df_new['Social 1'] = (df['stocktwitsLikes'] + df['twitterLikes'])
        twitter_impressions = (df['stocktwitsImpressions'] + df['twitterImpressions'])
        impressions_max = max(twitter_impressions.max(), df['Social 2 Peers'].max())
        impressions_min = min(twitter_impressions.min(), df['Social 2 Peers'].min())
        impressions_dif = impressions_max - impressions_min

        df_new['Twitter Impressions'] = twitter_impressions / 1000  # ((twitter_impressions - impressions_min) / impressions_dif) * 5
        df_new['Twitter Sentiment'] = (df['stocktwitsSentiment'] + df['twitterSentiment']) * 10
        df_new['Peer Twitter Posts'] = df['Social 1 Peers']
        df_new['Peer Twitter Impressions'] = df['Social 2 Peers'] / 1000  # ((df['Social 2 Peers'] - impressions_min) / impressions_dif) * 5
        df_new['Peer Twitter Sentiment'] = df['Social 3 Peers'] * 10
        df_new['Investment in R&D'] = df['RnDbyRev']
        df_new['Environmental Score'] = df['EnvironmentalScore']
        df_new['Social Score'] = df['SocialScore']
        df_new['Governance Score'] = df['GovernanceScore']
        df_new['Peer Investments in R&D'] = df['Technological Peers']
        df_new['Peer Environmental Score'] = df['Environmental 1 Peers']
        df_new['Peer Social Score'] = df['Environmental 2 Peers']
        df_new['Peer Governance Score'] = df['Environmental 3 Peers']
        df_new['Bipartisan Buying'] = df['BP Buys']
        df_new['Bipartisan Selling'] = -df['BP Sells']
    elif type == 'peers':
        # df_new['Social 1 Peers'] = (df['stocktwitsLikes'] + df['twitterLikes'])
        df_new['Social 1 Peers'] = (df['stocktwitsPosts'] + df['twitterPosts'])
        # df_new['Social 1 Alt 2'] = (df['stocktwitsComments'] + df['twitterComments'])
        df_new['Social 2 Peers'] = (df['stocktwitsImpressions'] + df['twitterImpressions'])
        df_new['Social 3 Peers'] = (df['stocktwitsSentiment'] + df['twitterSentiment'])
        df_new['Technological Peers'] = df['RnDbyRev']
        df_new['Environmental 1 Peers'] = df['EnvironmentalScore']
        df_new['Environmental 2 Peers'] = df['SocialScore']
        df_new['Environmental 3 Peers'] = df['GovernanceScore']

    return df_new




def combine_stock_data_peers(symbol, start_date=None, end_date=None, frequency='daily', duplicate_handling='keep_first', lookback_days=90):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Extend the start date to fetch extra data
    extended_start = (pd.to_datetime(start_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    # Define your data-fetching functions
    data_fetch_functions = [
        (get_income_statement_data, 'date'),
        (get_social_metric, 'date'),
        (get_esg_scores, 'date'),
    ]

    data_frames = []

    for fetch_func, date_col in data_fetch_functions:
        df = fetch_func(symbol, extended_start, end_date)

        if df.empty:
            continue

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'

        # Resampling and handling duplicates
        df_resampled = forward_fill_to_frequency(df, output_frequency=frequency,
                                                 duplicate_dates_handling=duplicate_handling)
        data_frames.append(df_resampled)

    # Concatenate all data frames
    combined_df = pd.concat(data_frames, axis=1, join='outer')
    combined_df = combined_df.ffill()

    # Trim the combined DataFrame to the originally specified date range
    combined_df = combined_df.loc[start_date:end_date]

    return combined_df

def get_stock_peers(symbol):
    """
    Fetches a list of peer stock symbols for the given stock ticker symbol.

    Parameters:
    - symbol (str): The stock ticker symbol.

    Returns:
    - list: A list of peer stock symbols.
    """
    api_key = "tLHD5sBz6eXnv6yv4nUnk1SXgqYjPHr4"
    endpoint = f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={symbol}&apikey={api_key}"

    # Make the API call
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        # Extract and return the list of peers if the response contains the expected data
        if data and isinstance(data, list) and 'peersList' in data[0]:
            holder = data[0]['peersList']
            # holder.append(symbol)
            return holder
        else:
            print("Peers list not found in the response.")
            return []
    else:
        print(f"Failed to fetch peers for {symbol}. Status code: {response.status_code}")
        return []

def average_dataframes(dfs, start_date, end_date):
    if not dfs:
        return pd.DataFrame()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    date_range = pd.date_range(start=start_date, end=end_date)

    summed_df = dfs[0].reindex(date_range).fillna(0)
    counts_df = dfs[0].reindex(date_range).notna().astype(int)

    for df in dfs[1:]:
        df_reindexed = df.reindex(date_range)

        for col in df_reindexed.columns:
            if col not in summed_df.columns:
                summed_df[col] = 0
                counts_df[col] = 0

            summed_df[col] += df_reindexed[col].fillna(0)
            counts_df[col] += df_reindexed[col].notna().astype(int)

    averaged_df = summed_df.div(counts_df.where(counts_df != 0, 1))

    return averaged_df

def add_peers_df(symbol, start_date=None, end_date=None, frequency='daily', duplicate_handling='keep_first', lookback_days=90):
    peers = get_stock_peers(symbol)
    # peers = ["AAPL", "MSFT"]
    # print(peers)
    holder = []
    for peer in peers:
        holder.append(combine_stock_data_peers(peer, start_date, end_date))

    averaged = average_dataframes(holder, start_date, end_date)
    # print(averaged)
    return process_df(averaged, "peers").fillna(0)

def combine_stock_data(symbol, start_date=None, end_date=None, frequency='daily', duplicate_handling='keep_first', lookback_days=90):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Extend the start date to fetch extra data
    extended_start = (pd.to_datetime(start_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    # Define your data-fetching functions
    data_fetch_functions = [
        # (lambda s, sd, ed: FRED_data("DEV", sd, ed), 'Date'),
        (lambda s, sd, ed: FRED_data("JOBS", sd, ed), 'Date'),
        (lambda s, sd, ed: FRED_data("IAE", sd, ed), 'Date'),
        (lambda s, sd, ed: FRED_data("TBILL", sd, ed), 'Date'),
        (get_income_statement_data, 'date'),
        (get_social_metric, 'date'),
        (get_analyst_recommendations, 'Date'),
        # (get_senate_transactions, 'Transaction Date'),
        # (get_house_trades, 'Transaction Date'),
        (get_bipartisan_investments, 'Transaction Date'),
        (get_partisan_investments, 'Transaction Date'),
        (get_esg_scores, 'date'),
        (get_keymetrics, 'date'),
        (get_all_technical_indicators, 'Date'),
        (add_peers_df,'Date')
    ]

    data_frames = []

    for fetch_func, date_col in data_fetch_functions:
        df = fetch_func(symbol, extended_start, end_date)

        if df.empty:
            continue

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'

        # Resampling and handling duplicates
        df_resampled = forward_fill_to_frequency(df, output_frequency=frequency,
                                                 duplicate_dates_handling=duplicate_handling)
        data_frames.append(df_resampled)

    # Concatenate all data frames
    combined_df = pd.concat(data_frames, axis=1, join='outer')
    combined_df = combined_df.ffill()

    combined_df = combined_df.loc[start_date:end_date]

    return combined_df

if __name__ == "__main__":
    stocks = combine_stock_data("AAPL", '2022-01-01', '2024-01-03', frequency='daily', duplicate_handling='keep_first')
    # stocks = get_all_technical_indicators("AAPL", '2022-01-01', '2024-01-01', n=2)
    pd.set_option('display.max_columns', None)
    print(stocks.tail(5))