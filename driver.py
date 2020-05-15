# Written by: Intsar Saeed

import os
import sys
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    # This function process the given data in a directory generate *.csv file
    # The file is already generated for testing (imdb_tr.csv) and testing (imdb_tr.csv)
    # Use this function ONLY when you need to generate the Testing and Training .csv for a new set of data

    # Initialize empty data-frame
    data_frame = pd.DataFrame(columns=["text", "polarity"])

    # Read files
    pos_path = os.path.join(inpath + "pos/")
    neg_path = os.path.join(inpath + "neg/")
    pos_files = [file for file in os.listdir(pos_path)]
    neg_files = [file for file in os.listdir(neg_path)]

    for i in range(len(pos_files)):
        data_pos = open(pos_path + pos_files[i], 'r', encoding="utf8").read()
        data_neg = open(neg_path + neg_files[i], 'r', encoding="utf8").read()
        data_frame = data_frame.append({'text': data_pos, 'polarity': 1}, ignore_index=True)
        data_frame = data_frame.append({'text': data_neg, 'polarity': 0}, ignore_index=True)
    data_frame.to_csv(name)


if __name__ == "__main__":

    # Path to the inputs
    train_path = os.path.join(sys.path[0], "imdb_tr.csv")
    test_path = os.path.join(sys.path[0], "imdb_te.csv")

    # Data processing
    # imdb_data_preprocess(inpath=train_path)  # Only use when required

    # Open processed data file
    training_data = pd.read_csv(train_path, skiprows=0, encoding="ISO-8859-1")
    testing_data = pd.read_csv(test_path, skiprows=0, encoding="ISO-8859-1")
    corpus = training_data["text"]

    # High-frequency words that do not play a big role when training the model
    with open(os.path.join(sys.path[0], "stopwords.en.txt")) as f:
        stopwords = set([item.strip() for item in f])

    # Traina SGD classifier using unigram representation and predict sentiments
    vectorizer = CountVectorizer(stop_words=stopwords)
    X_train = vectorizer.fit_transform(corpus)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(X_train, training_data["polarity"])
    # Predict
    X_test = vectorizer.transform(testing_data["text"])
    f = open("unigram.output.txt", 'w')
    for x in X_test:
        f.write(str(int(clf.predict(x))) + "\n")
    f.close()

    # Train SGD classifier using bigram representation and predict sentiments
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
    X_train = vectorizer.fit_transform(corpus)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(X_train, training_data["polarity"])
    # Predict
    X_test = vectorizer.transform(testing_data["text"])
    f = open("bigram.output.txt", 'w')
    for x in X_test:
        f.write(str(int(clf.predict(x))) + "\n")
    f.close()

    # Train a SGD classifier using unigram representation with tf-idf and predict sentiments
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X_train = vectorizer.fit_transform(corpus)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(X_train, training_data["polarity"])
    # Predict
    X_test = vectorizer.transform(testing_data["text"])
    f = open("unigramtfidf.output.txt", 'w')
    for x in X_test:
        f.write(str(int(clf.predict(x))) + "\n")
    f.close()

    # Train a SGD classifier using bigram representation with tf-idf and predict sentiments
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords)
    X_train = vectorizer.fit_transform(corpus)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(X_train, training_data["polarity"])
    # Predict
    X_test = vectorizer.transform(testing_data["text"])
    f = open("bigramtfidf.output.txt", 'w')
    for x in X_test:
        f.write(str(int(clf.predict(x))) + "\n")
    f.close()
