import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    This file reads in a database file path and returns the approporiate columns for the message and the expected output
    
    Args:
    database_filepath: string the path that contains the database file
    
    Returns:
    X: message for every row in the database    
    y: expected output for each row in the database 
    column_names: the column names for every output column
    
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages',engine.connect())
    
   
    output_column_names=list(set(df.columns)-set(["id",'message',"original","genre"]))
    df[output_column_names]=df[output_column_names].astype("int32")
    
    
    X = df["message"]
    y = df[output_column_names]
    
    
    
    return X,y,output_column_names

def tokenize(text):
    """
    This function takes in a sting cleans it and tokenize it.
    
    Args:
    text: string  The input text that is going to be cleaned and tokenized
    
    Returns:
    list: list of tokens after cleaning.
    """
    
    #initialize word lemmatizer
    lemmatizer = WordNetLemmatizer() 
    #remove punctuation and uwanted characters
    text=re.sub("r[^a-zA-Z0-9]"," ",text)
    #tokenzie the words
    words= nltk.word_tokenize(text)
    #lower case the words and lemmantize them
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    #remove stop words
    words= [word for word in words if word not in stopwords.words('english')]
    
    return words


def build_model():
    """
    This function builds the pipeline that we are going to train
    
    Returns:
    sklearn pipeline object: the pipeline that we are going to train
    """
    
    # instantiate a pipeline with Bag of words TFIDF then KNN classifier wrapped with multioutput classifier
    pipeline = Pipeline([
    ('bow',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultiOutputClassifier(KNeighborsClassifier())) 
])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function prints out the evaluation of model provided in the params. It calculates the percision, recall and F1 score
    
    Args:
    model: the sklearn model that we ant to evaluate against out testing set
    X_test: list the messages of the testing set
    Y_test: list the expected output for each message
    category_names: list the column names for each output class in the Y_test
    
    
    Returns:
    None
    prints an evaluation of the model
    
    """
    ypred= model.predict(X_test)
    print(classification_report(np.array(Y_test,ndmin=2),np.array(ypred,ndmin=2), labels=list(range(36)),target_names= category_names ))


def save_model(model, model_filepath):
    """
    save a model to a pkl file 
    
    Args:
    model: sklearn model
    model_filepath: string the path to file we are saving thee model to
    
    """
    filename = model_filepath.strip()
    if filename[-4:]!='.pkl':
        filename+='.pkl'
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()