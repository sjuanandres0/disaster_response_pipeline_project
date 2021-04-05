import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge data from 2 filepaths
    INPUT:
        messages_filepath (str) - path to csv file
        categories_filepath (str) - path to csv file
    OUTPUT:
        df (pd.DataFrame) - DataFrame with messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath')
    df = messages.merge(categories)
    return df, categories


def clean_data(df, categories):
    """Clean the dataframe
    INPUT:
        df (pd.DataFrame) - Dataframe from load_data
    OUTPUT:
        df (pd.DataFrame) - Clean Dataframe
    """
    categories_id = categories['id']
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    categories = pd.concat([categories_id, categories], axis=1)
    df.drop('categories', axis=1, inplace=True)
    df = df.merge(categories, on='id')
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Saves dataframe into a database
    INPUT:
        df (pd.DataFrame) - Clean Dataframe
        database_filename (str): filename from user
    OUTPUT: None
    """
    engine_route = 'sqlite:///' + str(database_filename)
    engine = create_engine(engine_route)
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)

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
