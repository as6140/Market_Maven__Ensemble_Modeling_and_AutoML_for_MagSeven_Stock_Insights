# market-maven
Project repository for Team 128 - Market Maven: Ensemble Modeling for Strategic Stock Insights

API Document:\
https://site.financialmodelingprep.com/developer/docs

## Steps to run  

> First run DownloadData.py to pull data from the API.  The stock_list can be edited for the desired stocks.

> Next run GeneratePredictions.py to run our PyCaret AutoML pipeline for any given stock(s) which have been downloaded.

> Install modules via `python venv` in order to see outputs based on visualization and data. (Optional)</br>  Check paths on input data. 

```bash
$ cd ./stock-dashboard
$ python -m virtualenv env # Optional
$ Mac/linux: source env/bin/activate # Optional
or
$ Windows : env/Scripts/activate # Optional
$ pip install -r requirements.txt # Required
```

> Start the app

```bash
$ python manage.py runserver
```

At this point, the app runs at `http://127.0.0.1:8000/`. 

<br />
---
