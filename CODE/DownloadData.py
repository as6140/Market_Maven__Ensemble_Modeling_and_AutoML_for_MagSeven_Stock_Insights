from df_builder import combine_stock_data, process_df
import pandas as pd
import time

stock_list = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "META", "NVDA"]
sd = '2022-1-1'
ed = '2023-12-31'
pd.set_option('display.max_columns', None)

for tickr in stock_list:
    stock = combine_stock_data(tickr, sd, ed, frequency='daily', duplicate_handling='keep_first')
    stock.to_csv('./data/stocks/'+tickr+'.csv')
    # stock = pd.read_csv('./data/stocks/'+tickr+'.csv', index_col=0, parse_dates=True)
    stock_vis = process_df(stock, 'visual')
    stock_vis.to_csv('./stock-dashboard/csv/' + tickr + '.csv', index_label='Date')
    print(tickr," is finished.")
    time.sleep(15)