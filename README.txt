DESCRIPTION:
CompoundIndicators.py contains the file for the compound technical indicators, such as Ichimoku Cloud and Bollinger Bands, which were implemented by hand rather than through an API.

df_builder contains all API calls and supporting code to build the core dataframe containing our technical information used for prediction and building the dataframes for visualization.

DownloadData.py contains code to download data for a set of stocks and then process the data into one dataframe formatted for visualization and another for use in prediction.

GeneratePredictions.py runs the PyCaret AutoML pipeline to predict the future prices of the specified stocks.

The data folder contains a few example dataframes.

The stock-dashboard folder contains the code for our front end visualizations.

INSTALLATION:
$ cd ./stock-dashboard
$ python -m virtualenv env # Optional
$ Mac/linux: source env/bin/activate # Optional
or
$ Windows : env/Scripts/activate # Optional
$ pip install -r requirements.txt # Required

EXECUTION:
First run DownloadData.py to pull data from the API. The stock_list can be edited for the desired stocks.

Next run GeneratePredictions.py to run our PyCaret AutoML pipeline for any given stock(s) which have been downloaded.

Change the terminal directory to the stock-dashboard folder and run the following command:
"python manage.py runserver"

The server will now run locally and can be accessed from http://127.0.0.1:8000/

VIDEO TUTORIAL:
https://youtu.be/hGqtY2BTxpY