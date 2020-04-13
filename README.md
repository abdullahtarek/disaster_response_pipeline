# Disaster Response Pipeline Project

### Table of contents
- [Introduction](#Introduction)     
- [Requirements](#Requirements)     
- [Instructions](#Instructions)      


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
