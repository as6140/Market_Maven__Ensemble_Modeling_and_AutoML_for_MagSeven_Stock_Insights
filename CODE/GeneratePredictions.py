from df_builder import combine_stock_data
import pandas as pd
import pycaret.time_series as pyc
import numpy as np
import os

# start dev
##### Choose optimizer metric
optimizer = 'RMSE'
data_start_date = '2022-01-01'
data_end_date = '2024-01-07'

# FUNCTION INPUTS
list_of_tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "META", "NVDA"]

list_of_day7_preds = []

# FOR LOOP FOR PYCARET
for ticker in list_of_tickers:
    print(f"Stock being modelled: {ticker}")
    stocks = pd.read_csv('./data/stocks/'+ticker+'.csv', index_col=0, parse_dates=True)
    # SETUP
    print(f"Setup Step for: {ticker}")
    s = pyc.setup(stocks, fh=7, session_id=123, target='close', seasonal_period='D', \
                  numeric_imputation_target='ffill', numeric_imputation_exogenous='ffill', \
                  fold=5)  # , normalize=True, transformation=True)
    print("Setup Complete")
    # COMPARE MODELS
    print(f"Compare Models Step for: {ticker}; Optimizing: {optimizer}")
    comparison = pyc.compare_models(
        include=['arima', 'br_cds_dt', 'gbr_cds_dt', 'croston', 'en_cds_dt', 'rf_cds_dt', 'ada_cds_dt', \
                 'lightgbm_cds_dt'], \
        sort=optimizer, n_select=3)
    comparison_df = pyc.pull()
    print("Compare Models Complete")

    # create models separately
    print(f"Create Individual Models Step for: {ticker}")
    print(f"Model 1: {comparison_df.index[0]} :: {comparison_df['Model'][0].split(' w/')[0]}")
    model1 = pyc.create_model(comparison_df.index[0], return_train_score=True)
    print(f"Model 2: {comparison_df.index[1]} :: {comparison_df['Model'][1].split(' w/')[0]}")
    model2 = pyc.create_model(comparison_df.index[1], return_train_score=True)
    print(f"Model 3: {comparison_df.index[2]} :: {comparison_df['Model'][2].split(' w/')[0]}")
    model3 = pyc.create_model(comparison_df.index[2], return_train_score=True)
    print("Create Individual Models Complete")

    # TUNE MODELS
    print(f"Tune Individual Models Step for: {ticker}")
    print(f"Tuning Model 1: {comparison_df.index[0]} :: {comparison_df['Model'][0].split(' w/')[0]}")
    model1_tuned = pyc.tune_model(model1, optimize=optimizer, choose_better=True, n_iter=5)  # default: n_iter=10,
    print(f"Tuning Model 2: {comparison_df.index[1]} :: {comparison_df['Model'][1].split(' w/')[0]}")
    model2_tuned = pyc.tune_model(model2, optimize=optimizer, choose_better=True, n_iter=5)  # default: n_iter=10
    print(f"Tuning Model 3: {comparison_df.index[2]} :: {comparison_df['Model'][2].split(' w/')[0]}")
    model3_tuned = pyc.tune_model(model3, optimize=optimizer, choose_better=True, n_iter=5)  # default: n_iter=10
    print("Tune Individual Models Complete")

    # blend models
    print(f"Blending Models Step for: {ticker}")
    blender = pyc.blend_models([model1_tuned, model2_tuned, model3_tuned], fold=5, method='mean')
    print("Blending Models Complete")

    # tune the blended weights
    print(f"Tuning Weights of Blended Model Step for: {ticker}")
    tuned_blender = pyc.tune_model(blender)
    print("Tuning Weights of Blended Model Complete")

    # PREDICTIONS
    stocks_reversed = stocks.iloc[::-1]
    std = stocks_reversed['standardDeviation'].iloc[0]
    std2 = std * 2
    std3 = std * 3

    holdout_pred = pyc.predict_model(tuned_blender)
    model1_pred = pyc.predict_model(model1_tuned)
    model1_holdout_score_grid = pyc.pull()
    model2_pred = pyc.predict_model(model2_tuned)
    model2_holdout_score_grid = pyc.pull()
    model3_pred = pyc.predict_model(model3_tuned)
    model3_holdout_score_grid = pyc.pull()

    # ADD ALL the Y_PREDS to holdout_pred
    holdout_pred.rename(columns={"y_pred": "y_pred_tuned_blended"}, inplace=True)
    model1_pred.rename(columns={"y_pred": f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"}, inplace=True)
    model2_pred.rename(columns={"y_pred": f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"}, inplace=True)
    model3_pred.rename(columns={"y_pred": f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"}, inplace=True)
    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] = model1_pred[
        f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"]
    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] = model2_pred[
        f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"]
    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] = model3_pred[
        f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"]

    # Add STD DEVIATIONS
    numeric_index = np.arange(len(holdout_pred))

    # Apply the adjusted standard deviation
    holdout_pred["y_pred_tuned_blended_plus_1std"] = holdout_pred['y_pred_tuned_blended'] + (numeric_index + 1) * std
    holdout_pred["y_pred_tuned_blended_minus_1std"] = holdout_pred['y_pred_tuned_blended'] - (numeric_index + 1) * std
    holdout_pred["y_pred_tuned_blended_plus_2std"] = holdout_pred['y_pred_tuned_blended'] + 2 * (
                numeric_index + 1) * std
    holdout_pred["y_pred_tuned_blended_minus_2std"] = holdout_pred['y_pred_tuned_blended'] - 2 * (
                numeric_index + 1) * std
    holdout_pred["y_pred_tuned_blended_plus_3std"] = holdout_pred['y_pred_tuned_blended'] + 3 * (
                numeric_index + 1) * std
    holdout_pred["y_pred_tuned_blended_minus_3std"] = holdout_pred['y_pred_tuned_blended'] - 3 * (
                numeric_index + 1) * std

    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}_plus_1std"] = model1_pred[
                                                                                        f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] + std * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}_minus_1std"] = model1_pred[
                                                                                         f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] - std * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}_plus_2std"] = model1_pred[
                                                                                        f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] + std2 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}_minus_2std"] = model1_pred[
                                                                                         f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] - std2 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}_plus_3std"] = model1_pred[
                                                                                        f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] + std3 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}_minus_3std"] = model1_pred[
                                                                                         f"y_pred_{comparison_df['Model'][0].split(' w/')[0]}"] - std3 * (
                numeric_index + 1)

    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}_plus_1std"] = model2_pred[
                                                                                        f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] + std * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}_minus_1std"] = model2_pred[
                                                                                         f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] - std * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}_plus_2std"] = model2_pred[
                                                                                        f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] + std2 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}_minus_2std"] = model2_pred[
                                                                                         f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] - std2 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}_plus_3std"] = model2_pred[
                                                                                        f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] + std3 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}_minus_3std"] = model2_pred[
                                                                                         f"y_pred_{comparison_df['Model'][1].split(' w/')[0]}"] - std3 * (
                numeric_index + 1)

    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}_plus_1std"] = model3_pred[
                                                                                        f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] + std * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}_minus_1std"] = model3_pred[
                                                                                         f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] - std * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}_plus_2std"] = model3_pred[
                                                                                        f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] + std2 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}_minus_2std"] = model3_pred[
                                                                                         f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] - std2 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}_plus_3std"] = model3_pred[
                                                                                        f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] + std3 * (
                numeric_index + 1)
    holdout_pred[f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}_minus_3std"] = model3_pred[
                                                                                         f"y_pred_{comparison_df['Model'][2].split(' w/')[0]}"] - std3 * (
                numeric_index + 1)

    # holdout_pred.to_csv(f"./data/results/{ticker}_7day_predictions.csv")
    if not os.path.exists(f"./stock-dashboard/results/{ticker}_results/"):
        os.makedirs(f"./stock-dashboard/results/{ticker}_results/")
    holdout_pred.to_csv(f"./stock-dashboard/results/{ticker}_results/{ticker}_7day_predictions.csv", index_label='Date')

