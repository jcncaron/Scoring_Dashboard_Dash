# Scoring_Dashboard_Dash

root directory:
---------------------
* main.py: app
  - Importing modules from dash library
  - Importing functions from dashboard_functions/functions.py and dashboard_functions/figures.py
  - Initializing app with Dash
  - Prepare dataset and predict with model_lgbm_1.joblib and test_df_1000
  - Define different figures for the dashboard : predict_proba score, local and global feature importances, feature distributions according to the predicted classes
  - Application layout with dash
  - @app.callback for the interactive part of the dashboard
* Procfile: specifies the commands that are executed by the app
  - declare the app's web server
* requirements.txt: keeps track of the modules and packages used in the project
  - created with 'pip freeze > requirements.txt command' in terminal
* runtime.txt: declares the exact python version number to use (for Heroku deployment for example)

assets directory:
-----------------
* bootstrap.css: template for the dash dashboard
* header.css: template for the headers of the dashboard
* typo.css: specifies the typography of body and title texts

dashboard_functions directory:
------------------------------
* functions.py : functions used in main.py
* figures.py : functions to define figures to plot in main.py
  - with plotly.express and plotly.graph_objects

data directory:
----------------
* model_lgbm_1.joblib: lgbmclassifier serialized model in joblib format
* test_df_1000: dataset with 1000 customers only
