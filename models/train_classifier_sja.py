import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle

def load_data(database_filepath):
    """Loads data previously saved in a database
    INPUT:
        database_filepath (str) - path to SQLite database
    OUTPUT:
        X (pd.Series) - Series object of message data
        y (pd.DataFrame) - DataFrame of categories (columns to be predicted)
    """
    engine_route = 'sqlite:///' + str(database_filepath)
    engine = create_engine(engine_route)
    table_name = str(engine.table_names()[0])
    print('DB table names', engine.table_names())
    df = pd.read_sql(table_name, con=engine)
    X = df['message']
    y = df.iloc[:,-36:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """Tokenize function processes message text data:
    lowercase, no punctuation, lemmatization, URLs
    INPUT:
        text (str) - text to tokenize
    OUTPUT:
        cleaned_tokens (list) - list of tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model(grid_search=False):
    """Builds the Machine Learning Pipeline.
    INPUT:
        grid_search (boolean): if true, a grid search is performed to find the
            most optimal model
    OUTPUT:
        Sklearn Pipeline (with or without GridSearch)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
                    min_samples_split=4, n_estimators=100)
            ,n_jobs=-1))
    ])

    if grid_search == True:
        parameters = {
                'clf__estimator__n_estimators': [50, 100, 500, 1000]
                ,'clf__estimator__min_samples_split': [2, 3, 57]
#               ,vect__ngram_range': ((1, 1), (1, 2))
#               ,'vect__max_df': (0.5, 0.75, 1.0)
#               ,'vect__max_features': (None, 5000, 10000)
#               ,'tfidf__use_idf': (True, False)
        }
        cv = GridSearchCV(pipeline, param_grid=parameters)
        return cv

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluate the model for each category and save results in csv file.
    INPUT:
        model (sklearn Pipeline) - Name of model
        X_test (pd.DataFrame) - Column with messages
        y_test (pd.DataFrame) - Encoded columns of categories
        model_filepath (str) - path of model (used to save results under same
            name as Pickle file)
        category_names (list) - list of names of target categories (labels)
    OUTPUT:
        None
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(data=y_pred, columns=category_names)

    results_dic = {}
    reports = []
    for column in category_names:
        report = classification_report(y_test[column], y_pred[column])
        reports.append(report)
        print('--------------\n\nClassification report for {}'
            .format(column.upper()))
        print(report)
        results_dic[column] = classification_report(y_test[column],
            y_pred[column], output_dict=True)['weighted avg']
    pd.DataFrame(results_dic).transpose().to_csv('results.csv')


def save_model(model, model_filepath):
    """Save model as a pickle file
    INPUT:
      model (sklearn Pipeline): name of model
      model_filepath (str): specified path
    OUTPUT: None
    """
    #Reference: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    pickle_out = open(model_filepath,'wb')
    pickle.dump(model, pickle_out)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model(grid_search=False)

        print('Training model, it can take a while...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)
        print('Evaluation results saved in {}'.format('results.csv')

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
