import ssl

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def cleaning_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean the dataset given a data_frame.
    This function removes duplicates and empty rows.
    :param data_frame: The data frame (dataset) to clean
    :return: The cleaned data frame (dataset)
    """
    data_frame.drop_duplicates(inplace=True)

    # Removing empty rows
    data_frame.dropna(inplace=True)

    # We prefer to work with lowercase text
    data_frame["Resume_str"] = data_frame["Resume_str"].apply(lambda x: x.lower())

    return data_frame


def data_vectorization(data_frame: pd.DataFrame):
    """
    Function to vectorize the data frame (dataset) to return only what I need.
    :param data_frame: The data frame (dataset) to vectorize
    :return: The vectorized data frame (dataset)
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    processed_data_frame = vectorizer.fit_transform(data_frame["Resume_str"].apply(lambda x: np.str_(x)))

    return processed_data_frame, vectorizer


def process_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function to process the data frame (dataset) to return only what I need.
    :param data_frame: The data frame (dataset) to process
    :return: The processed data frame (dataset)
    """
    # Only keep resume_str and category columns
    processed_data_frame = data_frame[["Resume_str", "Category"]]

    return processed_data_frame


def tokenize_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function to tokenize the data frame (dataset) to return only what I need.
    :param data_frame: The data frame (dataset) to process
    :return: The processed data frame (dataset)
    """
    data_frame["Resume_str"] = data_frame["Resume_str"].apply(nltk.word_tokenize)

    # We only keep the resume_str column as we return a data frame
    return data_frame[["Resume_str"]]


def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download()

    # Variable containing the path to PDF CVs we want to analyze
    # folder_to_analyze: str = "./dataset/data/DIGITAL-MEDIA"

    # Opening the resumes.csv file and inserting it into a pandas DataFrame
    data_frame: pd.DataFrame = pd.read_csv("dataset/resumes/resumes.csv")

    data_frame: pd.DataFrame = cleaning_data(data_frame)

    Y = data_frame["Category"]

    data_frame: pd.DataFrame = process_data(data_frame)

    data_frame = tokenize_data(data_frame)

    # We vectorize the data frame to make it easier for the machine learning algorithm
    X, vectorizer = data_vectorization(data_frame)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Print the classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    #    print(processed_data_frame)
    print(vectorizer)


if __name__ == '__main__':
    main()
