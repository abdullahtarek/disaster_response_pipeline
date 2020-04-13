import sys
from sqlalchemy import create_engine
import pandas  as pd
import numpy as np
import re

def load_data(messages_filepath, categories_filepath):
    """
    Reads in data from a csv and merge them together. 
    
    Args:
    message_filepath: string path to the csv file that has messages data
    categories_filepath: string path to the csv file that has the catgories for each message 
    
    Returns:
    dataframe: the merged data
    
    
    """
    
    
    # read datasets from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge the two tables
    df = pd.merge(messages,categories,how='inner',on=['id'] )
    
    return df
    
def clean_data(df):
    """
    Cleans and wrangels the data into a format that can be easily put into a machine learning algorithm
    
    Args: 
    df : dataframe containig the data from the load data function
    
    Returns:
    df: dataframe containing the data after cleaning and wrangling
    """
    
    #transform the column that have th message category into a format that will be easily used for ML algorithms
    category_colnames= re.sub('[\-0-1]','',df.loc[0,'categories']).split(';') 
    
    categories = df["categories"].str.split( ";",n=36, expand=True)
    
    categories.columns = category_colnames
    
    categories= categories[categories["related"]!="related-2"]
    
    #make the columns zero or  1 only
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    
    #merge the expanded categories with the original dataframe
    df = df.drop(columns=['categories'])
    df = pd.concat([df,categories],join="inner", axis=1)
    
    #drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    load the data from a dataframe to a database
    
    Args:
    df: the dataframe to save as a database
        
    databse_filename:string the path to save the database in
        
    Returns:
    None
    saves the dataframe as a database at the specified location
    
    
    """
    
    database_name="disaster_messages"
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_name, engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()