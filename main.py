import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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
    # We need, punkt to tokenize the text, so we check if it's already downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Variable containing the path to PDF CVs we want to analyze
    # folder_to_analyze: str = "./dataset/data/DIGITAL-MEDIA"

    # Opening the resumes.csv file and inserting it into a pandas DataFrame
    data_frame: pd.DataFrame = pd.read_csv("dataset/resumes/resumes.csv")

    data_frame: pd.DataFrame = cleaning_data(data_frame)
    data_frame: pd.DataFrame = process_data(data_frame)
    data_frame = tokenize_data(data_frame)

    # We join the tokens to get the resume as a string
    data_frame["Resume_str"] = data_frame["Resume_str"].apply(lambda x: ' '.join(x))

    # We initialize the TF-IDF vectorizer to convert the text into a matrix of TF-IDF features
    tfidf_vectorizer = TfidfVectorizer()

    # We fit and transform the data frame (dataset) to get the TF-IDF features
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame["Resume_str"])


if __name__ == '__main__':
    main()
