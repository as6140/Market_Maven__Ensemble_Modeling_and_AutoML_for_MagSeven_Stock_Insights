# pylint: disable=unspecified-encoding
from django.shortcuts import render
from django.http import HttpResponse
import json
import csv
import os
import pandas as pd
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


# Create your views here.
def index(request):
    csv_files = ['Select', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    day_range = ['2', '3', '4', '5', '6', '7']
    if request.method == 'POST':
        selected_csv = request.POST.get('csv_file')
        algo_model = request.POST.get('algoModel')
        prediction_days = request.POST.get('days')
        selected_day = [str(prediction_days)]
        filepath = os.getcwd() + '/csv/' + selected_csv + '.csv'
        df = pd.read_csv(filepath)
        last_180_rows = df.iloc[516:]
        last_180_rows_date = last_180_rows['Date'].tolist()

        # ------------------Prediction-------------------
        csv_options = {
            'AAPL': os.getcwd() + '/results' + '/AAPL_results' + '/AAPL_7day_predictions.csv',
            'AMZN': os.getcwd() + '/results' + '/AMZN_results' + '/AMZN_7day_predictions.csv',
            'GOOGL': os.getcwd() + '/results' + '/GOOGL_results' + '/GOOGL_7day_predictions.csv',
            'META': os.getcwd() + '/results' + '/META_results' + '/META_7day_predictions.csv',
            'MSFT': os.getcwd() + '/results' + '/MSFT_results' + '/MSFT_7day_predictions.csv',
            'NVDA': os.getcwd() + '/results' + '/NVDA_results' + '/NVDA_7day_predictions.csv',
            'TSLA': os.getcwd() + '/results' + '/TSLA_results' + '/TSLA_7day_predictions.csv',
            # Add more mappings as needed
        }
        apple_file_path = os.getcwd() + '/AAPL_results' + '/AAPL_7day_predictions.csv'
        pred_file_path = csv_options.get(selected_csv, apple_file_path)
        pred_df = pd.read_csv(pred_file_path)

        print('algo_model-----------', algo_model)
        updated_pred_df = pd.DataFrame()
        if selected_csv == 'AAPL':
            if algo_model == 'Tuned Blended':
                print('----------------------------  AAPL called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Light Gradient Boosting':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Gradient Boosting':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Random Forest':
                indices = [0, 4] + list(range(23, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]
        elif selected_csv == 'AMZN':
            if algo_model == 'Tuned Blended':
                print('---------------------------- AMZN called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Bayesian Ridge':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Elastic Net':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'ARIMA':
                indices = [0, 4] + list(range(24, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]
        elif selected_csv == 'GOOGL':
            if algo_model == 'Tuned Blended':
                print('---------------------------- GOOGL called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'ARIMA':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Elastic Net':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Bayesian Ridge':
                indices = [0, 4] + list(range(23, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]
        elif selected_csv == 'META':
            if algo_model == 'Tuned Blended':
                print('---------------------------- META called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'ARIMA':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Bayesian Ridge':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Elastic Net':
                indices = [0, 4] + list(range(23, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]
        elif selected_csv == 'MSFT':
            if algo_model == 'Tuned Blended':
                print('---------------------------- MSFT called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'ARIMA':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Light Gradient Boosting':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Random Forest':
                indices = [0, 4] + list(range(23, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]
        elif selected_csv == 'NVDA':
            if algo_model == 'Tuned Blended':
                print('---------------------------- NVDA called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'ARIMA':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Light Gradient Boosting':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Gradient Boosting':
                indices = [0, 4] + list(range(23, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]
        elif selected_csv == 'TSLA':
            if algo_model == 'Tuned Blended':
                print('---------------------------- TSLA called')
                indices = [0, 1] + list(range(5, 11))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'ARIMA':
                indices = [0, 2] + list(range(11, 17))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Bayesian Ridge':
                indices = [0, 3] + list(range(17, 23))
                updated_pred_df = pred_df.iloc[:, indices]
            elif algo_model == 'Gradient Boosting':
                indices = [0, 4] + list(range(23, len(pred_df.columns)))
                updated_pred_df = pred_df.iloc[:, indices]

        if len(updated_pred_df.columns) > 0:
            updated_pred_df.iloc[:, 1:] = updated_pred_df.iloc[:, 1:].round(2)

        print('---------updated_pred_df', updated_pred_df)
        updated_pred_df = updated_pred_df.iloc[:int(selected_day[0])]

        date_rows = updated_pred_df['Date'].tolist()
        print('---------date_rows', date_rows)
        # Assuming last_180_rows_date is a list of date strings like ["2023-06-01", "2023-06-02", ...]
        date_x_axis_labels = []

        # You can decide the interval of labels here. For example, 7 for weekly.
        label_interval = 1
        for i, date_str in enumerate(date_rows):
            if i % label_interval == 0:  # This will only add a label every 'label_interval' days
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_x_axis_labels.append(date_obj.strftime("%b %d"))
            else:
                date_x_axis_labels.append(None)  # Skip other labels

        y_axis_data = []

        for column in updated_pred_df:
            if column != 'Date':
                print('================', updated_pred_df[column].tolist())
                # col_obj[column]= updated_pred_df[column].values.toList()
                y_axis_data.append({
                    'name': column,
                    'data': updated_pred_df[column].tolist()
                })
        print('y_axis_data', y_axis_data)

        prediction_chart_data = {
            'x': date_x_axis_labels,
            'y': y_axis_data,
            'algo_model': algo_model
        }

        print('prediction_chart_data', prediction_chart_data)

        # --------------------Political---------------------
        df['Partisan Investing'] = df['Partisan Investing'].apply(lambda x: round(x, 2))
        public_opinion = ''
        if (df['Partisan Investing'].iloc[-1] > 0.5):
            public_opinion = 'Republic'

        if (df['Partisan Investing'].iloc[-1] <= 0.5):
            public_opinion = 'Democratic'

        political_chart_data = {
            'value': df['Partisan Investing'].iloc[-1],
            'public_opinion': public_opinion
        }

        # --------------------Price---------------------
        stock_price_filtered_data = last_180_rows['Price'].apply(lambda x: round(x, 2))
        stock_price = stock_price_filtered_data.tolist()

        # Assuming last_180_rows_date is a list of date strings like ["2023-06-01", "2023-06-02", ...]
        x_axis_labels = []

        # You can decide the interval of labels here. For example, 7 for weekly.
        label_interval = 30
        for i, date_str in enumerate(last_180_rows_date):
            if i % label_interval == 0:  # This will only add a label every 'label_interval' days
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                x_axis_labels.append(date_obj.strftime("%b %d"))
            else:
                x_axis_labels.append(None)  # Skip other labels

        price_chart_data = {
            'x': x_axis_labels,
            'y': [
                {
                    'name': 'Price',
                    'data': stock_price
                }
            ]
        }

        print('price_chart_data', price_chart_data)

        # --------------------Economic---------------------

        last_180_rows['Job Postings'] = last_180_rows['Job Postings'].apply(lambda x: round(x, 2))
        last_180_rows['Equity Investment'] = last_180_rows['Equity Investment'].apply(lambda x: round(x, 2))
        last_180_rows['Risk Free Rate'] = last_180_rows['Risk Free Rate'].apply(lambda x: round(x, 2))

        economic1 = last_180_rows['Job Postings'].tolist()
        economic2 = last_180_rows['Equity Investment'].tolist()
        economic3 = last_180_rows['Risk Free Rate'].tolist()

        economic_chart_data = {
            'x': x_axis_labels,
            'y': [
                {
                    'name': 'Job Postings',
                    'data': economic1
                },
                {
                    'name': 'Equity Investment',
                    'data': economic2
                },
                {
                    'name': 'Risk Free Rate',
                    'data': economic3
                },
            ]
        }

        # --------------------Technological---------------------

        df['Investment in R&D'] = df['Investment in R&D'].apply(lambda x: round(x, 2))
        df['Peer Investments in R&D'] = df['Peer Investments in R&D'].apply(lambda x: round(x, 2))

        technological_chart_data = {
            'technological': round(df['Investment in R&D'].iloc[-1] * 10, 2),
            'technologicalPeers': round(df['Peer Investments in R&D'].iloc[-1] * 10, 2)
        }

        # --------------------Environmental---------------------

        df['Social Score'] = df['Social Score'].apply(lambda x: round(x, 2))
        df['Peer Social Score'] = df['Peer Social Score'].apply(lambda x: round(x, 2))

        df['Governance Score'] = df['Governance Score'].apply(lambda x: round(x, 2))
        df['Peer Governance Score'] = df['Peer Governance Score'].apply(lambda x: round(x, 2))

        df['Environmental Score'] = df['Environmental Score'].apply(lambda x: round(x, 2))
        df['Peer Environmental Scores'] = df['Peer Environmental Score'].apply(lambda x: round(x, 2))

        environmental_chart_data = {
            'Social Score': df['Social Score'].iloc[-1],
            'Peer Social Score': df['Peer Social Score'].iloc[-1],
            'Governance Score': df['Governance Score'].iloc[-1],
            'Peer Governance Score': df['Peer Governance Score'].iloc[-1],
            'Environmental Score': df['Environmental Score'].iloc[-1],
            'Peer Environmental Scores': df['Peer Environmental Scores'].iloc[-1]
        }

        print('environemntal', environmental_chart_data)

        # --------------------Social---------------------

        df['Twitter Posts'] = df['Twitter Posts'].apply(lambda x: round(x, 2))
        df['Peer Twitter Posts'] = df['Peer Twitter Posts'].apply(lambda x: round(x, 2))

        df['Twitter Impressions'] = df['Twitter Impressions'].apply(lambda x: round(x, 2))
        df['Peer Twitter Impressions'] = df['Peer Twitter Impressions'].apply(lambda x: round(x, 2))

        df['Twitter Sentiment'] = df['Twitter Sentiment'].apply(lambda x: round(x, 2))
        df['Peer Twitter Sentiment'] = df['Peer Twitter Sentiment'].apply(lambda x: round(x, 2))

        social_chart_data = {
            'Twitter Posts': df['Twitter Posts'].iloc[-1],
            'Peer Twitter Posts': df['Peer Twitter Posts'].iloc[-1],
            'Twitter Impressions': df['Twitter Impressions'].iloc[-1],
            'Peer Twitter Impressions': df['Peer Twitter Impressions'].iloc[-1],
            'Twitter Sentiment': df['Twitter Sentiment'].iloc[-1],
            'Peer Twitter Sentiment': df['Peer Twitter Sentiment'].iloc[-1]
        }

        print('social_chart_data-----------', social_chart_data)

        # --------------------Economic---------------------

        last_180_rows['Bipartisan Buying'] = last_180_rows['Bipartisan Buying'].apply(lambda x: round(x, 2))
        last_180_rows['Bipartisan Selling'] = last_180_rows['Bipartisan Selling'].apply(lambda x: round(x, 2))

        legal1 = last_180_rows['Bipartisan Buying'].tolist()
        legal2 = last_180_rows['Bipartisan Selling'].tolist()
        legal_chart_data = {
            'x': x_axis_labels,
            'y': [
                {
                    'name': 'Bipartisan Buying',
                    'data': legal1,
                    'color': '#00FF00'
                },
                {
                    'name': 'Bipartisan Selling',
                    'data': legal2,
                    'color': '#ff0000'
                }
            ]
        }

        return render(request, 'pages/stocks.html', {'csv_files': csv_files,
                                                     'selected_csv': selected_csv,
                                                     'day_range': day_range,
                                                     'selected_days': selected_day,
                                                     'selected_algoModel': algo_model,
                                                     'political_chart_data': json.dumps(political_chart_data),
                                                     'economic_chart_data': json.dumps(economic_chart_data),
                                                     'technological_chart_data': json.dumps(technological_chart_data),
                                                     'environmental_chart_data': json.dumps(environmental_chart_data)
            , 'legal_chart_data': json.dumps(legal_chart_data),
                                                     'price_chart_data': json.dumps(price_chart_data),
                                                     'social_chart_data': json.dumps(social_chart_data),
                                                     'prediction_chart_data': json.dumps(prediction_chart_data)
                                                     })

    return render(request, 'pages/stocks.html', {'csv_files': csv_files, 'day_range': day_range})


@csrf_exempt
def get_data_for_csv(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        csv_file = data.get('csv_file')

        # Fetch or generate the data based on the csv_file
        # Example: querying a model or processing a file
        csv_options = {
            'AAPL': os.getcwd() + '/results' + '/AAPL_results' + '/AAPL_7day_predictions.csv',
            'AMZN': os.getcwd() + '/results' + '/AMZN_results' + '/AMZN_7day_predictions.csv',
            'GOOGL': os.getcwd() + '/results' + '/GOOGL_results' + '/GOOGL_7day_predictions.csv',
            'META': os.getcwd() + '/results' + '/META_results' + '/META_7day_predictions.csv',
            'MSFT': os.getcwd() + '/results' + '/MSFT_results' + '/MSFT_7day_predictions.csv',
            'NVDA': os.getcwd() + '/results' + '/NVDA_results' + '/NVDA_7day_predictions.csv',
            'TSLA': os.getcwd() + '/results' + '/TSLA_results' + '/TSLA_7day_predictions.csv',

            # Add more mappings as needed
        }
        apple_file_path = os.getcwd() + '/AAPL_results' + '/AAPL_7day_predictions.csv'
        file_path = csv_options.get(csv_file, apple_file_path)
        df = pd.read_csv(file_path)
        print(df.columns)
        dynamic_options = [
            {'value': df.columns[1], 'text': df.columns[1]},
            {'value': df.columns[2], 'text': df.columns[2]},
            {'value': df.columns[3], 'text': df.columns[3]},
            {'value': df.columns[4], 'text': df.columns[4]},
        ]

        return JsonResponse(dynamic_options, safe=False)
