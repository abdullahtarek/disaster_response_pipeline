# Disaster Response Pipeline Project

### Table of contents
- [Introduction](#Introduction)     
- [Requirements](#Requirements)     
- [Instructions](#Instructions)   
- [File description](#File_description)


### Introduction        
This project uses the [Figure 8](https://www.figure-eight.com/) dataset to make an API that classifies disaster messages using NLP and machine learning. This helps to filter out the important messages in case of disaster when systems are least capable of going through those messages to find important ones.

### Requirements
* python >=3.6
* sklearn
* pandas
* plotly
* flask
* nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File_description:
* **ETL_Pipeline_Preparation.ipynb**: Notebook to explore the dataset and explore how to wrangle and clean the data
* **ML_Pipeline_Preparation.ipynb**: Notebook for exploring different models and paramters on the wrangled dataset. and using grid search to find optimal paramters 
* **data/proccess_data.py**: python file that contains clean code of an ETL pipeline that can be run on the figure 8 dataset. like the above example
* **model/train_classifier.py**: python file that contains clean code of a machine learning pipeline that we can tun on the database extracted from the proccess_data.py step
* **app/run.py**: python file that runs the flask backend of the website.
* **app/templates/master.html**: html page that uses plotly to show a visualization on the webpage and takes input from the user for message classification
* **app/templates.go.html**: html page that displays the result of the classification


