import sklearn.datasets as datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as NaiveBayes


count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


def load_training_data():
    raw_text = datasets.load_files('datasets/train_data')
    training_label = raw_text.target_names
    training_count = count_vect.fit_transform(raw_text.data)
    bag_of_words_training = tfidf_transformer.fit_transform(training_count)
    return bag_of_words_training, training_label


def load_test_data():
    test_data = datasets.load_files('datasets/test_data')
    test_count = count_vect.transform(test_data.data)
    bag_of_word_test = tfidf_transformer.transform(test_count)
    return bag_of_word_test


def classifier():
    bayes = NaiveBayes()
    training_set, labels = load_training_data()
    bayes.fit(training_set, labels)
    return bayes
