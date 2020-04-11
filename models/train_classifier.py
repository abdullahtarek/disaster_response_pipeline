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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages',engine.connect())
    
   
    #REMOVE THIS and fix from the data wrangling part
    output_column_names=list(set(df.columns)-set(["id",'message',"original","genre"]))
    df[output_column_names]=df[output_column_names].astype("int32")
    
    df = df[df["related"]<2]
    
    X = df["message"].to_numpy()
    y = df[output_column_names]
    y=y.to_numpy()
    
    
    
    return X,y,output_column_names

def tokenize(text):
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
    pipeline = Pipeline([
    ('bow',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultiOutputClassifier(KNeighborsClassifier())) 
])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    ypred= model.predict(X_test)
    print(classification_report(np.array(Y_test,ndmin=2),np.array(ypred,ndmin=2), labels=list(range(36)),target_names= category_names ))


def save_model(model, model_filepath):
    filename = model_filepath.strip()
    if filename[-4:]!='.sav':
        filename+='.sav'
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